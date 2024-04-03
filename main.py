import dgl
import torch
import argparse
from model import GPR_ATT
from utils import setup_seed, get_data, normalize, gen_dgl_graph, get_classification_loss
import warnings
import random
warnings.filterwarnings("ignore", category=UserWarning)


def Set_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='STRING',
                        choices=['STRING', 'Non-Specific', 'Specific', 'Lofgof'])
    parser.add_argument('--dataset_name', type=str, default='hESC',
                        choices=['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L'])
    parser.add_argument('--top', type=str, default='500',
                        choices=['500', '1000'])
    parser.add_argument('--ratio', type=int, default=1,
                        help="positive and negtive ratio")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed of dataset and model")
    parser.add_argument('--k', type=int, default=10,
                        help="The number of neighbors in the KNN model")
    parser.add_argument('--hid_dim', type=int, default=32,
                        help="Hidden channels of the model")
    parser.add_argument('--num_layer', type=int, default=2,
                        help="The number of layers in the model")

    parser.add_argument('--x', type=int, default=250,
                        help="Update adjacency matrix every x epoch of training")
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--early_stop', type=int, default=1000)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    arg = Set_parameters()
    setup_seed(arg.seed)

    features, nfeats, labels, train_mask, val_mask, test_mask, g = get_data(arg)
    Adj = normalize(g.adj())
    g = gen_dgl_graph(Adj.indices()[0],
                      Adj.indices()[1],
                      edge_w=Adj.values(),
                      ndata=features,
                      )
    update_adj_epoch = arg.x

    model = GPR_ATT(in_channels=nfeats, hidden_channels=arg.hid_dim, out_channels=2, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)

    model = model.to("cuda")
    train_mask = train_mask.to("cuda")
    val_mask = val_mask.to("cuda")
    test_mask = test_mask.to("cuda")
    labels = labels.to("cuda")
    g = g.to("cuda")

    best_val = None
    early_stop = 0
    for epoch in range(1, arg.epoch):
        model.train()
        optimizer.zero_grad()

        loss, train_res = get_classification_loss(model, train_mask, g.ndata['h'], labels, g)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_loss, val_res = get_classification_loss(model, val_mask, g.ndata['h'], labels, g)
            early_stop += 1
            if best_val is None or val_res.auc > best_val.auc:
                best_val = val_res
                # print("epoch {}, Val Loss {:.4f}, Val Auc {:.4f}, Val AUPR {:.4f}".format(
                #     epoch, val_loss, best_val.auc, best_val.aupr))
                # with open(f"../result/{arg.dataset_type}_{arg.dataset_name}_{arg.top}.txt", "a+") as file:
                #     file.write("epoch {}, Val Loss {:.4f}, Val Auc {:.4f}, Val AUPR {:.4f} \n".format(
                #         epoch, val_loss, best_val.auc, best_val.aupr))

                test_loss_, test_res = get_classification_loss(model, test_mask, g.ndata['h'], labels, g)
                print("epoch {}, Test Loss {:.4f}, Test Auc {:.4f}, Test AUPR {:.4f}".format(
                    epoch, test_loss_, test_res.auc, test_res.aupr))
                # with open(f"../result/{arg.dataset_type}_{arg.dataset_name}_{arg.top}.txt", "a+") as file:
                #     file.write("epoch {}, Test Loss {:.4f}, Test Auc {:.4f}, Test AUPR {:.4f} \n".format(
                #         epoch, test_loss_, test_res.auc, test_res.aupr))
                early_stop = 0

        if epoch % update_adj_epoch == 0:
            # print('开始更新边')
            x_h = model.gen_node_emb(g.ndata['h'], g)
            knn_g = dgl.knn_graph(x_h, arg.k, algorithm='bruteforce-sharemem', dist='cosine', exclude_self=False)
            new_edges = knn_g.edges()
            g = gen_dgl_graph(torch.cat((g.edges()[0], new_edges[0])),
                              torch.cat((g.edges()[1], new_edges[1])),
                              ndata=g.ndata['h'],
                              ).to('cpu')
            g = dgl.to_simple(g)
            Adj = normalize(g.adj())
            g = gen_dgl_graph(Adj.indices()[0],
                              Adj.indices()[1],
                              edge_w=Adj.values(),
                              ndata=g.ndata['h'].to('cpu'),
                              )
            g = g.to("cuda")
            # print('更新完毕')
        if early_stop == arg.early_stop or epoch == arg.epoch - 1:
            print("Best Model Result:")
            # with open(f"../result/{arg.dataset_type}_{arg.dataset_name}_{arg.top}.txt", "a+") as file:
            #     file.write("Best Model Result: \n")
            print("epoch {}, Test Loss {:.4f}, Test Auc {:.4f}, Test AUPR {:.4f}".format(
                epoch - early_stop, test_loss_, test_res.auc, test_res.aupr))
            # with open(f"../result/{arg.dataset_type}_{arg.dataset_name}_{arg.top}.txt", "a+") as file:
            #     file.write("epoch {}, Test Loss {:.4f}, Test Auc {:.4f}, Test AUPR {:.4f} \n".format(
            #         epoch - early_stop, test_loss_, test_res.auc, test_res.aupr))
            # # with open(f"../experiment/aaa_total.txt", "a+") as file:
            # with open(f"../result/aaa_total.txt", "a+") as file:
            #     file.write("{}_{}_{} Auc {:.4f}, AUPR {:.4f} \n".format(
            #         arg.dataset_type, arg.dataset_name, arg.top, test_res.auc, test_res.aupr))
            with open(f"../result.txt", "a+") as file:
                file.write("Test Auc {:.4f}, Test AUPR {:.4f}\n".format(test_res.auc, test_res.aupr))
            break
