import argparse
import os
import json

import numpy as np
import torch
from sklearn.cluster import KMeans

from data import default_data_dir, load_cora
from loss import (
    cluster_level_loss,
    kl_loss_soft_centroids,
    NegativeEntropyLoss,
    sample_level_loss,
    centroid_level_loss,
    soft_cluster_centroids,
    instance_centroid_loss,
    contrastive_loss,
    feature_level_loss,
)
from models import DualViewGCN
from utils import cal_homo_ratio_fast, ensure_dir, eva, get_device, set_seed, drop_feature
from views import build_knn_view, make_message_passing_adj, compute_ppr_adj, apply_dynamic_pruning, build_knn_adj, prune_high_ebc_edges


def build_args():
    # 预解析 dataset，用于加载配置文件
    temp_p = argparse.ArgumentParser(add_help=False)
    temp_p.add_argument("--dataset", type=str, default="cora")
    temp_args, _ = temp_p.parse_known_args()

    dataset = temp_args.dataset
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", f"{dataset}_best.json")
    
    defaults = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                defaults = json.load(f)
            print(f"Loaded best config for {dataset} from {config_path}")
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")

    p = argparse.ArgumentParser(description="Dual-view GCN clustering")
    p.add_argument("--dataset", type=str, default="cora", help="数据集名称")
    p.add_argument("--data_dir", type=str, default=None, help="数据目录")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=defaults.get("seed", 0))
    p.add_argument("--cuda", type=int, default=1)
    p.add_argument("--epochs", type=int, default=defaults.get("epochs", 200))
    p.add_argument("--lr", type=float, default=defaults.get("lr", 5e-4))
    p.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 1e-3))
    p.add_argument("--grad_clip", type=float, default=defaults.get("grad_clip", 5.0))
    p.add_argument("--eval_interval", type=int, default=defaults.get("eval_interval", 10))
    p.add_argument("--print_interval", type=int, default=defaults.get("print_interval", 10))
    p.add_argument("--kmeans_n_init", type=int, default=defaults.get("kmeans_n_init", 20))
    p.add_argument("--update_weights", type=int, default=defaults.get("update_weights", 1))
    p.add_argument("--update_weights_interval", type=int, default=defaults.get("update_weights_interval", 10))
    p.add_argument("--weights_momentum", type=float, default=defaults.get("weights_momentum", 0.9))
    p.add_argument("--weights_min", type=float, default=defaults.get("weights_min", 0.05))
    p.add_argument("--knn_k", type=int, default=defaults.get("knn_k", 20))
    p.add_argument("--p_low_deg", type=float, default=defaults.get("p_low_deg", 0.4))
    p.add_argument("--p_low_deg_anneal_epochs", type=int, default=defaults.get("p_low_deg_anneal_epochs", 150), help="p_low_deg 退火到 0 的周期")
    p.add_argument("--p_high_ebc", type=float, default=defaults.get("p_high_ebc", 0.4))
    p.add_argument("--ebc_approx_k", type=int, default=defaults.get("ebc_approx_k", 256))
    p.add_argument("--v1_prop", type=str, default=defaults.get("v1_prop", "ppr"), choices=["gcn", "ppr"], help="视图 1 的传播方式")
    p.add_argument("--v2_prop", type=str, default=defaults.get("v2_prop", "gcn"), choices=["gcn", "ppr"], help="视图 2 的传播方式")
    p.add_argument("--ppr_alpha", type=float, default=defaults.get("ppr_alpha", 0.15), help="PPR 传送概率")
    p.add_argument("--ppr_threshold", type=float, default=defaults.get("ppr_threshold", 1e-4), help="PPR 稀疏化阈值")
    p.add_argument("--hidden_dim", type=int, default=defaults.get("hidden_dim", 256))
    p.add_argument("--output_dim", type=int, default=defaults.get("output_dim", 64))
    p.add_argument("--gcn_dropout", type=float, default=defaults.get("gcn_dropout", 0.1))
    p.add_argument("--gcn_impl", type=str, default=defaults.get("gcn_impl", "pyg"), choices=["dense", "pyg"])
    p.add_argument("--classifier_hidden", type=int, nargs="*", default=defaults.get("classifier_hidden", [128, 64]))
    p.add_argument("--w_sample", type=float, default=defaults.get("w_sample", 1.0))
    p.add_argument("--w_cluster", type=float, default=defaults.get("w_cluster", 2.0))
    p.add_argument("--w_centroid", type=float, default=defaults.get("w_centroid", 1.0))
    p.add_argument("--w_inst_cent", type=float, default=defaults.get("w_inst_cent", 1.0))
    p.add_argument("--w_feature", type=float, default=defaults.get("w_feature", 1.0), help="维度级对比损失权重")
    p.add_argument("--w_ne", type=float, default=defaults.get("w_ne", 4.0))
    p.add_argument("--w_kl", type=float, default=defaults.get("w_kl", 1.0))
    p.add_argument("--tau_sample", type=float, default=defaults.get("tau_sample", 0.5))
    p.add_argument("--tau_cluster", type=float, default=defaults.get("tau_cluster", 0.2))
    p.add_argument("--kl_max", type=float, default=defaults.get("kl_max", 0.5))
    p.add_argument("--kl_anneal_epochs", type=int, default=defaults.get("kl_anneal_epochs", 50))
    p.add_argument("--early_stop_patience", type=int, default=defaults.get("early_stop_patience", 0))
    return p.parse_args()


def main():
    args = build_args()
    set_seed(args.seed)

    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or default_data_dir()
    save_dir = args.save_dir or os.path.join(root, "runs", args.dataset)
    ensure_dir(save_dir)

    # 简单的泛化：根据 dataset 名称选择加载函数
    if args.dataset.lower() == "cora":
        labels, _, features, adj_label = load_cora(data_dir)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported yet.")
    device = get_device(bool(args.cuda))
    labels = labels.to(device)
    features = features.to(device)
    adj_label = adj_label.to(device).to(torch.float32)

    class_num = int(labels.max().item()) + 1
    input_dim = int(features.size(1))
    weights_h = torch.ones(2, device=device) / 2.0
    homo_rate = [0.5, 0.5]

    # 视图 1 传播矩阵
    if args.v1_prop == "ppr":
        print(f"Computing PPR matrix for View 1 (alpha={args.ppr_alpha}, threshold={args.ppr_threshold})...")
        adj_raw_mp = compute_ppr_adj(adj_label, alpha=args.ppr_alpha, threshold=args.ppr_threshold).to(device)
    else:
        adj_raw_mp = make_message_passing_adj(adj_label).to(device)

    # 预计算 KNN 的静态部分 (EBC 剪枝很慢，只在最开始算一次)
    print("Computing static base KNN view (this may take a few seconds)...")
    adj_knn_base = build_knn_adj(features, k=args.knn_k)
    adj_knn_base = prune_high_ebc_edges(adj_knn_base, ratio=args.p_high_ebc, approx_k=args.ebc_approx_k, seed=args.seed)
    
    # 应用初始的 p_low_deg
    adj_knn = apply_dynamic_pruning(adj_knn_base, p_low_deg=args.p_low_deg).to(device)
    
    # 视图 2 传播矩阵
    if args.v2_prop == "ppr":
        print(f"Computing PPR matrix for View 2 (alpha={args.ppr_alpha}, threshold={args.ppr_threshold})...")
        adj_knn_mp = compute_ppr_adj(adj_knn, alpha=args.ppr_alpha, threshold=args.ppr_threshold).to(device)
    else:
        adj_knn_mp = make_message_passing_adj(adj_knn).to(device)

    model = DualViewGCN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        class_num=class_num,
        gcn_dropout=args.gcn_dropout,
        gcn_impl=args.gcn_impl,
        classifier_hidden=args.classifier_hidden,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion_ne = NegativeEntropyLoss(ne_weight=args.w_ne).to(device)

    best_acc = -1.0
    best_nmi = -1.0
    best_state = None
    y_pred = None
    eval_centers = None
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()

        # p_low_deg 退火逻辑：每 5 个 epoch 更新一次，且仅应用轻量级的动态剪枝
        if args.p_low_deg_anneal_epochs > 0 and epoch < args.p_low_deg_anneal_epochs and epoch % 5 == 0:
            curr_p_low_deg = max(0.0, args.p_low_deg * (1.0 - epoch / args.p_low_deg_anneal_epochs))
            
            # 使用预计算好的 adj_knn_base，仅重新应用低度节点剪枝
            adj_knn = apply_dynamic_pruning(adj_knn_base, p_low_deg=curr_p_low_deg).to(device)
            
            # 根据配置更新传播矩阵 (如果是 PPR 则在 CPU 计算逆矩阵，防止显存爆炸)
            if args.v2_prop == "ppr":
                adj_knn_mp = compute_ppr_adj(adj_knn, alpha=args.ppr_alpha, threshold=args.ppr_threshold)
            else:
                adj_knn_mp = make_message_passing_adj(adj_knn)
            adj_knn_mp = adj_knn_mp.to(device)

        # 为视图一 (v1) 加入 10% 的特征扰动
        # features_v1 = drop_feature(features, drop_prob=0)
        
        out = model(
            xs=[features, features],
            adjs_mp=[adj_raw_mp, adj_knn_mp],
            adjs_labels=[adj_label, adj_knn],
            weights_h=weights_h,
        )
        hs = out["hs"]
        h_all = out["h_all"]
        cluster_q = out["cluster_q"]
        cluster_all = out["cluster_all"]
        adjs = out["adjs"]

        loss_sample = 0.0
        loss_cluster = 0.0
        loss_centroid = 0.0
        loss_feature = 0.0
        loss_inst_cent = 0.0
        loss_ne = 0.0
        
        # 视图间的对比损失 (v1 vs v2)
        u_v = [soft_cluster_centroids(cluster_q[v], hs[v]) for v in range(2)]
        u_all = soft_cluster_centroids(cluster_all, h_all)
        
        # 1. 视图间样本对比 (align same nodes)
        mask_node = torch.eye(hs[0].size(0), device=device)
        loss_sample_cross = contrastive_loss(hs[0], hs[1], mask_node, args.tau_sample, args.w_sample)
        
        # 2. 视图间簇对比 (align same clusters)
        loss_cluster_cross = cluster_level_loss(cluster_q[0], cluster_q[1], args.tau_cluster, args.w_cluster)
        
        # 3. 视图间簇心对比 (align same centroids)
        loss_centroid_cross = centroid_level_loss(u_v[0], u_v[1], args.tau_cluster, args.w_centroid)

        # 4. 视图间维度对比 (align same feature dimensions)
        loss_feature_cross = feature_level_loss(hs[0], hs[1], args.tau_sample, args.w_feature)

        # 视图与融合视图的对比 (v vs all)
        for v in range(2):
            loss_sample = loss_sample + sample_level_loss(adjs[v], hs[v], h_all, args.tau_sample, args.w_sample)
            loss_cluster = loss_cluster + cluster_level_loss(cluster_q[v], cluster_all, args.tau_cluster, args.w_cluster)
            loss_centroid = loss_centroid + centroid_level_loss(u_v[v], u_all, args.tau_cluster, args.w_centroid)
            loss_feature = loss_feature + feature_level_loss(hs[v], h_all, args.tau_sample, args.w_feature)
            loss_ne = loss_ne + criterion_ne(cluster_q[v], cluster_all)

        # 实例-簇心级别损失 (使用模型预测的 soft 中心和伪标签)
        labels_soft = torch.argmax(cluster_all, dim=1).detach()
        loss_inst_cent = instance_centroid_loss(h_all, u_all, labels_soft, args.tau_cluster, args.w_inst_cent)
            
        loss_sample = (loss_sample / 2.0 + loss_sample_cross) / 2.0
        loss_cluster = (loss_cluster / 2.0 + loss_cluster_cross) / 2.0
        loss_centroid = (loss_centroid / 2.0 + loss_centroid_cross) / 2.0
        loss_feature = (loss_feature / 2.0 + loss_feature_cross) / 2.0
        loss_ne = loss_ne / 2.0

        loss_kl = kl_loss_soft_centroids(
            cluster_q, hs, cluster_all, h_all, epoch, args.kl_max, args.kl_anneal_epochs
        )
        loss = loss_sample + loss_cluster + 0*loss_centroid + loss_feature + 0*loss_inst_cent + loss_ne + float(args.w_kl) * loss_kl

        if (epoch + 1) % int(args.print_interval) == 0 or epoch == 0:
            w_list = weights_h.cpu().tolist()
            print(
                f"epoch {epoch} loss {loss.item():.4f} "
                f"sample {loss_sample.item():.4f} cluster {loss_cluster.item():.4f} "
                f"feat {loss_feature.item():.4f} "
                f"centroid {loss_centroid.item():.4f} inst_cent {loss_inst_cent.item() if torch.is_tensor(loss_inst_cent) else loss_inst_cent:.4f} "
                f"ne {loss_ne.item():.4f} kl {loss_kl.item():.4f} "
                f"weights [{w_list[0]:.3f}, {w_list[1]:.3f}] "
                f"homo [{homo_rate[0]:.3f}, {homo_rate[1]:.3f}]"
            )

        optimizer.zero_grad()
        loss.backward()
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()

        model.eval()
        with torch.no_grad():
            do_eval = (epoch + 1) % int(args.eval_interval) == 0 or epoch == 0
            if do_eval:
                emb = h_all
                X = emb.detach().cpu().numpy()
                if not np.isfinite(X).all():
                    y_pred = np.argmax(cluster_all.detach().cpu().numpy(), axis=1)
                    eval_centers = None
                else:
                    km = KMeans(n_clusters=class_num, n_init=int(args.kmeans_n_init), random_state=args.seed)
                    y_pred = km.fit_predict(X)
                    eval_centers = km.cluster_centers_
                nmi, acc, ari, f1 = eva(labels.detach().cpu().numpy(), y_pred, epoch=epoch, visible=True)
                prev_best = float(best_acc)
                if float(acc) > best_acc:
                    best_acc = float(acc)
                    best_nmi = float(nmi)
                    best_state = {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_acc": best_acc,
                        "best_nmi": best_nmi,
                        "homo_rate": homo_rate,
                        "args": vars(args),
                        "eval_snapshot": {
                            "y_true": labels.detach().cpu(),
                            "y_pred": torch.from_numpy(y_pred).long(),
                            "metrics": {"acc": float(acc), "nmi": float(nmi), "ari": float(ari), "f1": float(f1)},
                        },
                    }
                    torch.save(best_state, os.path.join(save_dir, "best.pt"))
                if float(best_acc) > prev_best:
                    no_improve = 0
                else:
                    no_improve += 1
                if int(args.early_stop_patience) > 0 and no_improve >= int(args.early_stop_patience):
                    print(f"early stop at epoch {epoch}, best_acc {best_acc:.4f}")
                    break

            if int(args.update_weights) == 1:
                do_update = (epoch + 1) % int(args.update_weights_interval) == 0 or epoch == 0
                if do_update:
                    y_pred_t = torch.from_numpy(y_pred).to(device) if do_eval else torch.argmax(cluster_all, dim=1)
                    for v in range(2):
                        r, _ = cal_homo_ratio_fast(adjs[v], y_pred_t, self_loop=True)
                        homo_rate[v] = r
                    w = torch.tensor(homo_rate, device=device, dtype=torch.float32)
                    w = torch.clamp(w, min=float(args.weights_min))
                    w = w / (w.sum() + 1e-12)
                    mom = max(0.0, min(0.999, float(args.weights_momentum)))
                    weights_h = mom * weights_h + (1.0 - mom) * w
                    weights_h = weights_h / (weights_h.sum() + 1e-12)

    if best_state is not None:
        print(f"best_acc {best_state['best_acc']:.4f} best_nmi {best_state['best_nmi']:.4f} epoch {best_state['epoch']}")


if __name__ == "__main__":
    main()
