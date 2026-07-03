"""Wave 3: extra seeds for the statistically underpowered arms.

Per seed: baseline, supcon_true at lam {0.1, 0.3, 1.0}, supcon_shuffled at lam 0.1
(the lambda where the shuffled arm made the most progress -> hardest test of the
never-groks claim). Grokfast (n=5, consistent) and norm_matched (13/13 fails) are
already adequately powered.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_grid import run_one, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=50000)
    ap.add_argument("--grid_dir", type=str, default=os.path.join(ROOT, "runs", "grid"))
    args = ap.parse_args()
    base = ["--epochs", str(args.epochs)]
    for seed in args.seeds:
        s = ["--seed", str(seed)]
        run_one(os.path.join(args.grid_dir, f"baseline_s{seed}"),
                ["--condition", "baseline"] + s + base)
        for lam in (0.1, 0.3, 1.0):
            run_one(os.path.join(args.grid_dir, f"supcon_true_lam{lam}_s{seed}"),
                    ["--condition", "supcon_true", "--lambda_con", str(lam)] + s + base)
        run_one(os.path.join(args.grid_dir, f"supcon_shuffled_lam0.1_s{seed}"),
                ["--condition", "supcon_shuffled", "--lambda_con", "0.1"] + s + base)


if __name__ == "__main__":
    main()
