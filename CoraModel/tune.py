"""快速网格调参：搜索能达到更高 ACC 的超参组合。"""
import itertools
import os
import re
import subprocess
import json
import argparse

PYTHON = r"C:\Users\Miku12\.conda\envs\ahgfc\python.exe"
MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")

# 基础参数，作为搜索的起点
BASE = {
    "seed": 1,
    "epochs": 400,
    "gcn_impl": "pyg",
    "kmeans_n_init": 50,
    "eval_interval": 20,
    "print_interval": 100,
    "update_weights": 0,  # 固定权重
}

# 调参空间
GRID = {
    "lr": [5e-4, 1e-3, 2e-3],
    "w_cluster": [2.0, 4.0, 8.0],
    "w_kl": [0.5, 1.0, 2.0],
    "w_ne": [2.0, 4.0, 8.0],
    "w_centroid": [0.0, 0.5, 1.0],
    "w_inst_cent": [0.0, 0.5, 1.0],
    "w_feature": [0.0, 0.5, 1.0],
    "tau_cluster": [0.2, 0.5, 0.7],
    "min_conf_ratio": [0.2, 0.4, 0.6],
    "v2_prop": ["gcn", "ppr"],
}


def run_config(overrides, dataset="cora"):
    cfg = {**BASE, **overrides, "dataset": dataset}
    cmd = [PYTHON, MAIN]
    for k, v in cfg.items():
        if k == "classifier_hidden":
            cmd.extend([f"--{k}", *[str(x) for x in v]])
        else:
            cmd.extend([f"--{k}", str(v)])
    
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    
    # 解析输出中的最好结果: best_acc 0.7452 best_nmi 0.5821 epoch 180
    m = re.findall(r"best_acc\s+([\d.]+)\s+best_nmi\s+([\d.]+)\s+epoch\s+(\d+)", out)
    if not m:
        # 尝试另一种匹配方式，有时输出格式略有不同
        m = re.findall(r"ACC: ([\d.]+), NMI: ([\d.]+)", out)
        if not m:
            print("FAILED", cfg)
            print(out[-1000:])
            return None
        acc, nmi = m[-1]
        epoch = -1
    else:
        acc, nmi, epoch = m[-1]
    
    return float(acc), float(nmi), int(epoch), cfg, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--max_trials", type=int, default=100)
    args = parser.parse_args()

    keys = list(GRID.keys())
    combos = []
    for vals in itertools.product(*[GRID[k] for k in keys]):
        combos.append(dict(zip(keys, vals)))

    # 随机打乱或者按照某种策略排序
    import random
    random.seed(BASE["seed"])
    random.shuffle(combos)

    # 限制搜索规模
    combos = combos[:args.max_trials]

    best_acc_val = -1.0
    best_config_details = None
    results = []

    for i, overrides in enumerate(combos):
        print(f"\n=== trial {i + 1}/{len(combos)} {overrides} ===")
        ret = run_config(overrides, dataset=args.dataset)
        if ret is None:
            continue
            
        acc, nmi, epoch, cfg, _ = ret
        results.append((acc, nmi, epoch, cfg))
        print(f"trial {i + 1}: acc={acc:.4f} nmi={nmi:.4f} epoch={epoch}")
        
        if acc > best_acc_val:
            best_acc_val = acc
            best_config_details = (nmi, epoch, cfg)
            # 实时保存当前最好的
            save_best(args.dataset, best_acc_val, best_config_details)

    results.sort(key=lambda x: -x[0])
    print("\n======== TOP 10 ========")
    for row in results[:10]:
        print(f"acc={row[0]:.4f} nmi={row[1]:.4f} epoch={row[2]} cfg={row[3]}")


def save_best(dataset, best_acc_val, best_config_details):
    nmi, epoch, cfg = best_config_details
    print(f"\nUpdating BEST acc={best_acc_val:.4f} nmi={nmi:.4f} epoch={epoch}")
    
    # 完整的默认配置，确保 json 文件全量
    full_config = {
        "dataset": dataset,
        "seed": 0,
        "epochs": 200,
        "lr": 5e-4,
        "weight_decay": 1e-3,
        "grad_clip": 5.0,
        "eval_interval": 10,
        "print_interval": 10,
        "kmeans_n_init": 20,
        "update_weights": 1,
        "update_weights_interval": 10,
        "weights_momentum": 0.9,
        "weights_min": 0.05,
        "knn_k": 20,
        "p_low_deg": 0.4,
        "p_low_deg_anneal_epochs": 150,
        "p_high_ebc": 0.4,
        "ebc_approx_k": 256,
        "v1_prop": "ppr",
        "v2_prop": "gcn",
        "ppr_alpha": 0.15,
        "ppr_threshold": 1e-4,
        "hidden_dim": 256,
        "output_dim": 64,
        "gcn_dropout": 0.1,
        "gcn_impl": "pyg",
        "classifier_hidden": [128, 64],
        "w_sample": 1.0,
        "w_cluster": 2.0,
        "w_centroid": 1.0,
        "w_inst_cent": 1.0,
        "w_feature": 1.0,
        "w_ne": 4.0,
        "w_kl": 1.0,
        "tau_sample": 0.5,
        "tau_cluster": 0.2,
        "kl_max": 0.5,
        "kl_anneal_epochs": 150,
        "early_stop_patience": 0,
        "conf_warmup_epochs": 100,
        "min_conf_ratio": 0.2,
        "cuda": 1
    }
    
    # 用 BASE 覆盖默认
    full_config.update(BASE)
    # 用最优搜索结果覆盖
    full_config.update(cfg)
    
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"{dataset}_best.json")
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=4)
    
    print(f"Full best config saved to: {config_path}")


if __name__ == "__main__":
    main()
