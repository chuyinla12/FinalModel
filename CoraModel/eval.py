import argparse
import os

import numpy as np
import torch
from sklearn.cluster import KMeans

from data import default_data_dir, load_cora
from models import DualViewGCN
from utils import eva, get_device, set_seed
from views import build_knn_view, make_message_passing_adj


def main():
    p = argparse.ArgumentParser(description="Evaluate best checkpoint")
    root = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--dataset", type=str, default="cora", help="数据集名称")
    p.add_argument("--ckpt", type=str, default=None, help="模型路径，不指定则根据 dataset 自动寻找")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--cuda", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kmeans_n_init", type=int, default=20)
    args = p.parse_args()

    if args.ckpt is None:
        args.ckpt = os.path.join(root, "runs", args.dataset, "best.pt")
    
    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint not found at {args.ckpt}")
        return

    print(f"Loading checkpoint from {args.ckpt}...")
    state = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = state.get("args", {}) or {}
    set_seed(int(ckpt_args.get("seed", args.seed)))

    data_dir = args.data_dir or default_data_dir()
    labels, _, features, adj_label = load_cora(data_dir)
    device = get_device(bool(args.cuda))
    labels = labels.to(device)
    features = features.to(device)
    adj_label = adj_label.to(device).to(torch.float32)

    class_num = int(labels.max().item()) + 1
    input_dim = int(features.size(1))
    adj_raw_mp = make_message_passing_adj(adj_label)
    adj_knn, adj_knn_mp = build_knn_view(
        features,
        k=int(ckpt_args.get("knn_k", 20)),
        p_low_deg=float(ckpt_args.get("p_low_deg", 0.1)),
        p_high_ebc=float(ckpt_args.get("p_high_ebc", 0.4)),
        ebc_approx_k=int(ckpt_args.get("ebc_approx_k", 256)),
        seed=int(ckpt_args.get("seed", args.seed)),
    )
    adj_knn = adj_knn.to(device)
    adj_knn_mp = adj_knn_mp.to(device)

    model = DualViewGCN(
        input_dim=input_dim,
        hidden_dim=int(ckpt_args.get("hidden_dim", 256)),
        output_dim=int(ckpt_args.get("output_dim", 64)),
        class_num=class_num,
        gcn_dropout=float(ckpt_args.get("gcn_dropout", 0.1)),
        gcn_impl=str(ckpt_args.get("gcn_impl", "pyg")),
        classifier_hidden=ckpt_args.get("classifier_hidden", [128, 64]),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    homo_rate = state.get("homo_rate", [0.5, 0.5])
    w = torch.tensor(homo_rate, device=device, dtype=torch.float32)
    w = w / (w.sum() + 1e-12)

    with torch.no_grad():
        out = model(
            xs=[features, features],
            adjs_mp=[adj_raw_mp, adj_knn_mp],
            adjs_labels=[adj_label, adj_knn],
            weights_h=w,
        )
        h_all = out["h_all"]
        cluster_all = out["cluster_all"]

    X = h_all.detach().cpu().numpy()
    if not np.isfinite(X).all():
        y_pred = np.argmax(cluster_all.detach().cpu().numpy(), axis=1)
    else:
        km = KMeans(
            n_clusters=class_num,
            n_init=int(ckpt_args.get("kmeans_n_init", args.kmeans_n_init)),
            random_state=int(ckpt_args.get("seed", args.seed)),
        )
        y_pred = km.fit_predict(X)

    print("checkpoint", args.ckpt)
    eva(labels.detach().cpu().numpy(), y_pred, epoch=state.get("epoch", -1), visible=True)


if __name__ == "__main__":
    main()
