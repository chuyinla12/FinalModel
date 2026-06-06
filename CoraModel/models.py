import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import dense_to_sparse
except Exception:
    GCNConv = None
    dense_to_sparse = None


class GCNEncoderPyg(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        if GCNConv is None or dense_to_sparse is None:
            raise RuntimeError("torch_geometric is required for gcn_impl=pyg")
        self.dropout = float(dropout)
        self.conv1 = GCNConv(int(in_dim), int(hidden_dim), add_self_loops=False, normalize=True)
        self.conv2 = GCNConv(int(hidden_dim), int(out_dim), add_self_loops=False, normalize=True)

    def forward(self, x, adj_mp):
        edge_index, _ = dense_to_sparse(adj_mp)
        x = F.dropout(x, p=self.dropout, training=self.training)
        z1 = self.conv1(x, edge_index)
        z1 = F.elu(z1)
        z1 = F.normalize(z1, p=2, dim=1)
        z1 = F.dropout(z1, p=self.dropout, training=self.training)
        z2 = self.conv2(z1, edge_index)
        z2 = F.normalize(z2, p=2, dim=1)
        return z2, z1


class GCNEncoder(nn.Module):
    """双层典型 GCN：Linear -> ELU -> normalize -> Linear -> normalize。"""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.dropout = float(dropout)
        self.layer1 = nn.Linear(int(in_dim), int(hidden_dim), bias=True)
        self.layer2 = nn.Linear(int(hidden_dim), int(out_dim), bias=True)

    def _norm_adj(self, adj):
        A = adj.to(torch.float32)
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        return deg_inv_sqrt.view(-1, 1) * A * deg_inv_sqrt.view(1, -1)

    def forward(self, x, adj_mp):
        A = self._norm_adj(adj_mp)
        x = F.dropout(x, p=self.dropout, training=self.training)

        h = torch.matmul(A, x)
        h = self.layer1(h)
        h = F.elu(h)
        h = F.normalize(h, p=2, dim=1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h_penultimate = h

        h = torch.matmul(A, h)
        h = self.layer2(h)
        h = F.normalize(h, p=2, dim=1)
        return h, h_penultimate


class DualViewGCN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        class_num,
        view_num=2,
        gcn_dropout=0.1,
        gcn_impl="dense",
        classifier_hidden=None,
    ):
        super().__init__()
        self.class_num = int(class_num)
        self.view_num = int(view_num)
        gcn_impl = str(gcn_impl).strip().lower()
        if gcn_impl == "pyg":
            self.gcn = GCNEncoderPyg(input_dim, hidden_dim, output_dim, dropout=gcn_dropout)
        else:
            self.gcn = GCNEncoder(input_dim, hidden_dim, output_dim, dropout=gcn_dropout)

        classifier_hidden = [] if classifier_hidden is None else [int(x) for x in classifier_hidden]
        head_layers = []
        in_dim = int(output_dim)
        for h in classifier_hidden:
            head_layers.append(nn.Linear(in_dim, int(h)))
            head_layers.append(nn.ReLU())
            in_dim = int(h)
        head_layers.append(nn.Linear(in_dim, self.class_num))
        self.cluster_head = nn.Sequential(*head_layers)
        self.fuse_proj = nn.Linear(int(output_dim) * self.view_num, int(output_dim), bias=False)

    def forward(self, xs, adjs_mp, adjs_labels, weights_h):
        if not torch.is_tensor(weights_h):
            weights_h = torch.tensor(weights_h, device=xs[0].device, dtype=torch.float32)
        weights_h = weights_h.to(xs[0].device).to(torch.float32)

        hs, cluster_logits, adjs = [], [], []
        for v in range(self.view_num):
            h, _ = self.gcn(xs[v], adjs_mp[v])
            hs.append(h)
            adjs.append(adjs_labels[v].to(torch.float32))
            cluster_logits.append(self.cluster_head(h))

        w_sum = weights_h.sum() + 1e-12
        h_cat = torch.cat([hs[v] * (weights_h[v] / w_sum) for v in range(self.view_num)], dim=1)
        h_all = self.fuse_proj(h_cat)
        h_all = F.normalize(h_all, p=2, dim=-1)

        cluster_all_logits = self.cluster_head(h_all)
        cluster_q = [F.softmax(cluster_logits[v], dim=1) for v in range(self.view_num)]
        cluster_all = F.softmax(cluster_all_logits, dim=1)

        return {
            "hs": hs,
            "h_all": h_all,
            "adjs": adjs,
            "cluster_logits": cluster_logits,
            "cluster_all_logits": cluster_all_logits,
            "cluster_q": cluster_q,
            "cluster_all": cluster_all,
        }
