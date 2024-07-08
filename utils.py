import dgl
import torch
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.nn.functional as F
from collections import namedtuple
from torch_geometric.data import Data
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
Evaluation_Metrics = namedtuple('Evaluation_Metrics', ['auc', 'aupr'])


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(data_type, data_name, gene_top):
    file_path = f"Data/{data_type} Dataset/{data_name}/TFs+{gene_top}"
    data = pd.read_csv(f"{file_path}/BL--ExpressionData.csv", index_col=0)
    edge = pd.read_csv(f"{file_path}/BL--network.csv")[["Gene1", 'Gene2']].values.tolist()
    return data, edge


def create_negative_samples(edge, gene_list, feature, neg_ratio):
    new_node, label = [], []
    count = 0
    for a, b in edge:
        count += 1
        new_node.append(feature[gene_list.index(a)] + feature[gene_list.index(b)])
        label.append(1)
        for i in range(neg_ratio):
            count += 1
            while True:
                x = np.random.randint(len(gene_list))
                y = np.random.randint(len(gene_list))
                if x != y and [x, y] not in edge:
                    new_node.append(feature[x] + feature[y])
                    label.append(0)
                    break
    return new_node, label


def get_data(arg):
    print('get dataset:{} {} {}'.format(arg.dataset_type, arg.dataset_name, arg.top))
    data, edge = load_data(arg.dataset_type, arg.dataset_name, arg.top)
    print(len(data), len(edge))

    gene_list = data.index.tolist()
    feature = data.values
    random.shuffle(edge)

    ratio = arg.ratio
    new_node, label = create_negative_samples(edge, gene_list, feature, neg_ratio=ratio)

    num_data = len(edge)
    train_num, val_num, test_num = [int(num_data * frac) * (ratio+1) for frac in [0.6, 0.2, 0.2]]
    train_num = int(train_num)

    print("sum node：", len(new_node), "negative ratio：", ratio)
    print(train_num, val_num, test_num)

    train_mask = np.zeros(len(label), dtype=bool)
    train_mask[:train_num] = True
    val_mask = np.zeros(len(label), dtype=bool)
    val_mask[train_num: train_num + val_num] = True
    test_mask = np.zeros(len(label), dtype=bool)
    test_mask[train_num + val_num:] = True
    train_mask, val_mask, test_mask = [torch.tensor(mask, dtype=torch.bool) for mask in
                                       [train_mask, val_mask, test_mask]]

    new_node = torch.tensor(new_node).cuda()
    label = torch.tensor(label).cuda()

    print('run KNN')
    knn_g = dgl.knn_graph(new_node, 5, algorithm='bruteforce-sharemem', dist='cosine', exclude_self=False)
    edge_index = torch.LongTensor(np.concatenate([[knn_g.edges()[0].tolist()], [knn_g.edges()[1].tolist()]], axis=0))

    # # attack ratio
    # a = 0.1
    # num_edges = edge_index.size(1)
    # num_edges_to_remove = int(num_edges * a)
    # edges_to_remove = random.sample(range(num_edges), num_edges_to_remove)
    # remaining_edges = [i for i in range(num_edges) if i not in edges_to_remove]
    # new_edge_index = edge_index[:, remaining_edges]
    #
    # noise_edge = []
    # num = int(len(edge_index[0]) * a)
    # for i in range(num):
    #     while True:
    #         x = np.random.randint(len(new_node))
    #         y = np.random.randint(len(new_node))
    #         if x != y:
    #             noise_edge.append([x, y])
    #             break
    # edge_index = torch.cat([new_edge_index, torch.LongTensor(noise_edge).T], dim=1)

    data = Data(x=new_node, edge_index=edge_index, y=label, train_mask=train_mask, val_mask=val_mask,
                test_mask=test_mask)
    print(data)

    graph = dgl.graph((data.edge_index[0], data.edge_index[1]))
    graph.ndata['feature'] = data.x.cpu()
    graph.ndata['label'] = data.y.type(torch.LongTensor).cpu()
    train_mask, val_mask, test_mask = data['train_mask'], data['val_mask'], data['test_mask']
    graph.ndata['feature'] = torch.tensor(normalize_features(graph.ndata['feature']), dtype=torch.float)

    return graph.ndata['feature'], graph.ndata['feature'].size()[-1], graph.ndata['label'], \
           train_mask, val_mask, test_mask, graph


def normalize_features(mx):
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize(adj):
    adj = torch.sparse_coo_tensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                                  torch.FloatTensor(adj.val),
                                  torch.Size(adj.shape))
    adj = adj.coalesce()
    inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=(1,)).values()))
    D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
    new_values = adj.values() * D_value

    return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size()).coalesce()


def gen_dgl_graph(index1, index2, edge_w=None, ndata=None):
    g = dgl.graph((index1, index2))
    if edge_w is not None:
        g.edata['w'] = edge_w
    if ndata is not None:
        g.ndata['h'] = ndata
    return g


def evaluation_model_prediction(pred_logit, label):
    pred_label = np.argmax(pred_logit, axis=1)
    pred_logit = pred_logit[:, 1]

    AUC = roc_auc_score(label, pred_logit)
    precision, recall, _ = precision_recall_curve(label, pred_label)
    aupr = auc(recall, precision)
    return Evaluation_Metrics(auc=AUC, aupr=aupr)


def get_classification_loss(model, mask, features, labels, g=None):
    logits = model(features, g)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
    eval_res = evaluation_model_prediction(logp[mask].detach().cpu().numpy(), labels[mask].cpu().numpy())
    return loss, eval_res
