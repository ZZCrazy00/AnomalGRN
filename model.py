import dgl
import torch
import numpy as np
import torch.nn.functional as F


class GPR_ATT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GPR_ATT, self).__init__()

        self.inlinear = torch.nn.Linear(in_channels, hidden_channels)
        self.outlinear = torch.nn.Linear(hidden_channels, out_channels)

        torch.nn.init.xavier_uniform_(self.inlinear.weight)
        torch.nn.init.xavier_uniform_(self.outlinear.weight)

        self.gnn = GPR_sparse(hidden_channels, num_layers)
        self.extractor = ExtractorMLP(hidden_channels)

    def gen_node_emb(self, x, g=None):
        with g.local_scope():
            h = self.inlinear(x)
            h_gnn = self.gnn.forward(h, g)
            h_gnn = self.extractor.feature_extractor(h_gnn)
            return h_gnn

    def forward(self, x, g=None):
        with g.local_scope():
            h = self.inlinear(x)
            h_gnn = self.gnn.forward(h, g)
            g.edata['attn'] = self.extractor(h_gnn, g.edges())
            h_gnn = self.gnn.forward(h, g, edge_attn=True)
            x = self.outlinear(h_gnn)
        return x


class ExtractorMLP(torch.nn.Module):
    def __init__(self, hidden_size):
        super(ExtractorMLP, self).__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Dropout(p=0.2),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self._init_weight(self.feature_extractor)

    @staticmethod
    def _init_weight(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, emb, edge_index):
        col, row = edge_index
        f1, f2 = emb[col], emb[row]
        attn_logits = self.cos(self.feature_extractor(f1), self.feature_extractor(f2))
        return attn_logits


class GCNConv_dgl_attn(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl_attn, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x, g):
        g.ndata['h'] = self.linear(x)
        g.update_all(dgl.function.u_mul_e('h', 'w', 'm'), dgl.function.sum(msg='m', out='h'))
        return g.ndata['h']


class GPR_sparse(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super(GPR_sparse, self).__init__()

        self.layers = torch.nn.ModuleList(
            [GCNConv_dgl_attn(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        alpha = 0.1
        temp = alpha * (1 - alpha) ** np.arange(num_layers + 1)
        temp[-1] = (1 - alpha) ** num_layers
        self.temp = torch.nn.Parameter(torch.from_numpy(temp))

    def forward(self, x, g, edge_attn=False):
        if edge_attn:
            g.edata['w'] = g.edata['w'] * g.edata['attn']
        g.edata['w'] = F.dropout(g.edata['w'], p=0.5, training=self.training)
        hidden = x * self.temp[0]
        for i, conv in enumerate(self.layers):
            x = conv(x, g)
            x = F.relu(x)
            x = F.dropout(x, p=0.7, training=self.training)
            hidden += x * self.temp[i + 1]
        return hidden
