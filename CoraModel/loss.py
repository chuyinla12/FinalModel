import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_loss(qi, qj, mask, temperature, weight):
    qi = F.normalize(qi, p=2, dim=1)
    qj = F.normalize(qj, p=2, dim=1)

    sim_mm = torch.exp(torch.clamp(torch.mm(qi, qi.t()) / temperature, -50, 50))
    sim_mn = torch.exp(torch.clamp(torch.mm(qi, qj.t()) / temperature, -50, 50))

    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
    pos_mm = (mask * sim_mm * logits_mask).sum(dim=1)
    pos_mn = (mask * sim_mn).sum(dim=1)
    neg_mm = sim_mm.sum(dim=1)
    neg_mn = sim_mn.sum(dim=1)

    loss_val = (pos_mm + pos_mn) / (neg_mm + neg_mn + 1e-10)
    loss_val = torch.clamp(loss_val, min=1e-10)
    return -weight * torch.mean(torch.log(loss_val))


def sample_level_loss(adj, h_view, h_all, temperature, weight=1.0):
    mask = adj.to(torch.float32)
    mask = torch.nan_to_num(mask, nan=0.0)
    mask = torch.clamp(mask, min=0.0)
    mask = mask.clone()
    mask.fill_diagonal_(0.0)
    return contrastive_loss(h_view, h_all, mask, temperature, weight)


def cluster_level_loss(cluster_q, cluster_all, temperature, weight=1.0):
    n_clusters = cluster_q.size(1)
    mask = torch.eye(n_clusters, device=cluster_q.device)
    qi = cluster_q.t()
    qj = cluster_all.t()
    return contrastive_loss(qi, qj, mask, temperature, weight)


def centroid_level_loss(u_view, u_all, temperature, weight=1.0):
    """
    仿照簇级别对比损失，对簇中心 (centroids) 进行对比。
    u_view, u_all: [K, D]
    """
    n_clusters = u_view.size(0)
    mask = torch.eye(n_clusters, device=u_view.device)
    return contrastive_loss(u_view, u_all, mask, temperature, weight)


def instance_centroid_loss(h, centroids, labels, temperature, weight=1.0):
    """
    实例-簇心级别损失 (Instance-Centroid Contrastive Loss)。
    让节点嵌入 h 与其对应的簇中心（如 Soft Centroids）更紧密。
    h: [N, D]
    centroids: [K, D]
    labels: [N] (伪标签/分配)
    """
    if weight <= 0 or centroids is None:
        return torch.tensor(0.0, device=h.device)
    
    h = F.normalize(h, p=2, dim=1)
    centroids = F.normalize(centroids, p=2, dim=1)
    
    # 计算节点与所有簇中心的相似度: [N, K]
    logits = torch.mm(h, centroids.t()) / temperature
    
    # InfoNCE 等价于交叉熵损失，其中正样本是伪标签对应的中心
    return weight * F.cross_entropy(logits, labels)


def feature_level_loss(h_view, h_all, temperature, weight=1.0):
    """
    维度级对比损失 (Feature-level Contrastive Loss)。
    通过对特征矩阵转置 (H^T)，让不同的特征维度之间互为负样本，防止特征坍塌。
    h_view, h_all: [N, D] -> 转置后 [D, N]
    """
    if weight <= 0:
        return torch.tensor(0.0, device=h_view.device)
    D = h_view.size(1)
    mask = torch.eye(D, device=h_view.device)
    hi = h_view.t() # [D, N]
    hj = h_all.t()  # [D, N]
    return contrastive_loss(hi, hj, mask, temperature, weight)


class NegativeEntropyLoss(nn.Module):
    """ClusterLoss 中的负熵 (NE) 项。"""

    def __init__(self, ne_weight=1.0):
        super().__init__()
        self.ne_weight = float(ne_weight)

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i = p_i / (p_i.sum() + 1e-10)
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i + 1e-10)).sum()

        p_j = c_j.sum(0).view(-1)
        p_j = p_j / (p_j.sum() + 1e-10)
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j + 1e-10)).sum()
        return self.ne_weight * (ne_i + ne_j)


def soft_cluster_centroids(cluster_q, h):
    """u_c = sum_i q_{ic} * h_i，q 为 softmax 后的簇隶属，h 为未 softmax 的节点嵌入。"""
    u = torch.matmul(cluster_q.t(), h)
    mass = cluster_q.sum(dim=0, keepdim=True).t().clamp(min=1e-10)
    return u / mass


def student_t_distribution(z, u, alpha=1.0):
    dist = torch.sum(torch.pow(z.unsqueeze(1) - u.unsqueeze(0), 2), dim=2)
    q = 1.0 / (1.0 + dist / alpha)
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / (torch.sum(q, dim=1) + 1e-10)).t()
    return q


def target_distribution(q):
    weight = q ** 2 / (q.sum(0) + 1e-10)
    return (weight.t() / (weight.sum(1) + 1e-10)).t()


def kl_loss_soft_centroids(cluster_q_views, hs, cluster_all, h_all, epoch, kl_max, kl_anneal_epochs, alpha=1.0):
    """DEC 风格 KL：u 由 softmax(H) 对节点嵌入 h 加权求和得到，不再使用 KMeans 固定中心。"""
    l = min(float(kl_max), float(epoch + 1) / float(max(1, int(kl_anneal_epochs))) * float(kl_max))

    qgs = []
    for q_prob, h in zip(cluster_q_views, hs):
        u = soft_cluster_centroids(q_prob, h)
        qgs.append(student_t_distribution(h, u, alpha=alpha))
    u_all = soft_cluster_centroids(cluster_all, h_all)
    qgs.append(student_t_distribution(h_all, u_all, alpha=alpha))

    pgh = target_distribution(qgs[-1].detach())
    loss = F.kl_div((qgs[-1] + 1e-10).log(), pgh, reduction="batchmean")
    for v in range(len(qgs) - 1):
        pg = target_distribution(qgs[v].detach())
        loss = loss + F.kl_div((qgs[v] + 1e-10).log(), pg, reduction="batchmean")
        loss = loss + F.kl_div((qgs[v] + 1e-10).log(), pgh, reduction="batchmean")
    return l * loss
