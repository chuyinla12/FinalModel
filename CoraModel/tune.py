"""快速网格调参：搜索能达到更高 ACC 的超参组合。"""
import itertools
import os
import re
import subprocess
import json
import argparse

PYTHON = r"C:\Users\Miku12\.conda\envs\ahgfc\python.exe"
MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")

BASE = {
    "seed": 1,
    "epochs": 300,
    "gcn_impl": "pyg",
    "kmeans_n_init": 50,
    "eval_interval": 10,
    "print_interval": 50,
}

GRID = {
    "lr": [5e-4, 1e-3],
    "w_cluster": [2.0, 3.0, 4.0],
    "w_kl": [1.0, 3.0, 5.0],
    "w_ne": [2.0, 4.0],
    "p_high_ebc": [0.3, 0.4],
    "tau_cluster": [0.15, 0.2],
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
    m = re.findall(r"best_acc\s+([\d.]+)\s+best_nmi\s+([\d.]+)\s+epoch\s+(\d+)", out)
    if not m:
        print("FAILED", cfg)
        print(out[-2000:])
        return None
    acc, nmi, epoch = m[-1]
    return float(acc), float(nmi), int(epoch), cfg, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--max_trials", type=int, default=24)
    args = parser.parse_args()

    keys = list(GRID.keys())
    combos = []
    for vals in itertools.product(*[GRID[k] for k in keys]):
        combos.append(dict(zip(keys, vals)))

    # 先跑一组默认基线
    combos = [{}] + combos[:args.max_trials]

    best = (-1.0, None)
    results = []
    for i, overrides in enumerate(combos):
        print(f"\n=== trial {i + 1}/{len(combos)} {overrides} ===")
        ret = run_config(overrides, dataset=args.dataset)
        if ret is None:
            continue
        acc, nmi, epoch, cfg, _ = ret
        results.append((acc, nmi, epoch, cfg))
        print(f"trial {i + 1}: acc={acc:.4f} nmi={nmi:.4f} epoch={epoch}")
        if acc > best[0]:
            best = (acc, (nmi, epoch, cfg))

    results.sort(key=lambda x: -x[0])
    print("\n======== TOP 5 ========")
    for row in results[:5]:
        print(f"acc={row[0]:.4f} nmi={row[1]:.4f} epoch={row[2]} cfg={row[3]}")

    if best[1] is not None:
        acc, (nmi, epoch, cfg) = best[0], best[1]
        print(f"\nBEST acc={acc:.4f} nmi={nmi:.4f} epoch={epoch}")
        
        # 写入配置文件
        config_to_save = {**BASE, **cfg}
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, f"{args.dataset}_best.json")
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_to_save, f, indent=4)
        
        print(f"Best config saved to: {config_path}")
        print("config:", config_to_save)


if __name__ == "__main__":
    main()
