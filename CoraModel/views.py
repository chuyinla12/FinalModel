import numpy as np
import torch
import networkx as nx


def build_knn_adj(x: torch.Tensor, k: int, metric: str = "cosine"):
    metric = str(metric).strip().lower()
    N = int(x.size(0))
    k = max(1, min(int(k), N - 1))
    if metric != "cosine":
        raise ValueError("Only cosine metric is supported")
    z = torch.nn.functional.normalize(x, p=2, dim=1)
    sim = torch.mm(z, z.t())
    _, idx = torch.topk(sim, k=k + 1, dim=1)
    mask = torch.zeros((N, N), device=x.device, dtype=torch.bool)
    row = torch.arange(N, device=x.device).unsqueeze(1).expand_as(idx)
    mask[row, idx] = True
    mask.fill_diagonal_(False)
    mask = mask | mask.t()
    return mask.to(torch.float32)


def prune_low_degree_edges(adj: torch.Tensor, ratio: float, score: str = "avg"):
    ratio = float(ratio)
    if ratio <= 0:
        return adj
    A = (adj > 0).to(torch.float32)
    A = torch.triu(A, diagonal=1)
    idx = A.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return adj
    u = idx[:, 0]
    v = idx[:, 1]
    deg = (adj > 0).to(torch.float32).sum(dim=1)
    du = deg[u]
    dv = deg[v]
    score = str(score).strip().lower()
    if score == "sum":
        s = du + dv
    elif score == "avg":
        s = (du + dv) * 0.5
    else:
        s = torch.minimum(du, dv)
    m = int(idx.size(0))
    rm = int(round(m * ratio))
    rm = max(0, min(rm, m))
    if rm <= 0:
        return adj
    _, order = torch.sort(s, descending=False)
    remove = idx[order[:rm]]
    A = (adj > 0).to(torch.bool)
    A[remove[:, 0], remove[:, 1]] = False
    A[remove[:, 1], remove[:, 0]] = False
    A.fill_diagonal_(False)
    return A.to(torch.float32)


def prune_high_ebc_edges(adj: torch.Tensor, ratio: float, approx_k: int = 256, seed: int = 0):
    ratio = float(ratio)
    if ratio <= 0:
        return adj
    A = (adj > 0).to(torch.bool).detach().cpu()
    N = int(A.size(0))
    edges = torch.triu(A, diagonal=1).nonzero(as_tuple=False).numpy()
    if edges.shape[0] == 0:
        return adj
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from([(int(u), int(v)) for u, v in edges])
    k = int(approx_k)
    if k <= 0 or k >= N:
        k = None
    if k is not None:
        bc = nx.edge_betweenness_centrality(G, k=k, seed=int(seed))
    else:
        bc = nx.edge_betweenness_centrality(G)
    m = len(bc)
    rm = int(round(m * ratio))
    rm = max(0, min(rm, m))
    if rm <= 0:
        return adj
    sorted_edges = sorted(bc.items(), key=lambda kv: kv[1], reverse=True)
    to_remove = [e for e, _ in sorted_edges[:rm]]
    G.remove_edges_from(to_remove)
    A2 = nx.to_numpy_array(G, dtype=np.float32)
    np.fill_diagonal(A2, 0.0)
    return torch.from_numpy(A2).to(adj.device)


def make_message_passing_adj(adj_label: torch.Tensor):
    A = (adj_label > 0).to(torch.float32)
    A.fill_diagonal_(1.0)
    return A


def compute_ppr_adj(adj_label: torch.Tensor, alpha: float = 0.15, threshold: float = 1e-4):
    """
    计算 Personalized PageRank (PPR) 矩阵，并进行阈值稀疏化。
    S = alpha * (I - (1-alpha) * D^-1/2 * A * D^-1/2)^-1
    """
    alpha = float(alpha)
    A = (adj_label > 0).to(torch.float32)
    N = A.size(0)
    
    # 对称归一化 A
    D = A.sum(dim=1)
    D_inv_sqrt = torch.pow(D + 1e-12, -0.5)
    A_norm = D_inv_sqrt.view(-1, 1) * A * D_inv_sqrt.view(1, -1)
    
    # S = alpha * (I - (1-alpha) * A_norm)^-1
    # 在 CPU 上计算逆矩阵以节省显存
    I = torch.eye(N, device="cpu")
    M = I - (1.0 - alpha) * A_norm.cpu()
    S = alpha * torch.inverse(M)
    
    # 阈值稀疏化
    if threshold > 0:
        S[S < threshold] = 0
    
    return S.to(adj_label.device)


def build_knn_view(features: torch.Tensor, k: int, p_low_deg: float, p_high_ebc: float, ebc_approx_k: int, seed: int):
    # 1. 构建基础 KNN
    adj_knn = build_knn_adj(features, k=k)
    # 2. 先进行高边介数剪枝 (这部分通常是静态的)
    adj_knn = prune_high_ebc_edges(adj_knn, ratio=p_high_ebc, approx_k=ebc_approx_k, seed=seed)
    # 3. 进行低度节点剪枝
    adj_knn = prune_low_degree_edges(adj_knn, ratio=p_low_deg, score="avg")
    
    adj_knn = torch.clamp(adj_knn, 0, 1)
    adj_knn.fill_diagonal_(0.0)
    return adj_knn


def apply_dynamic_pruning(adj_base: torch.Tensor, p_low_deg: float):
    """仅应用低度节点剪枝，用于退火过程中的快速更新"""
    adj = prune_low_degree_edges(adj_base, ratio=p_low_deg, score="avg")
    adj = torch.clamp(adj, 0, 1)
    adj.fill_diagonal_(0.0)
    return adj
