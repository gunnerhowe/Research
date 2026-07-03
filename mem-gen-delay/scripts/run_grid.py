"""Sequential grid runner. Skips runs whose summary.json already exists.

Phase A: baseline, supcon_true (lambda sweep), supcon_shuffled (matched), grokfast.
Phase B: norm_matched — requires the paired supcon_true run's norms.npy (same seed,
         primary lambda), per the matched-counterfactual methodology of 2606.13753.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_one(out_dir, extra):
    if os.path.exists(os.path.join(out_dir, "summary.json")):
        print(f"skip (done): {out_dir}")
        return
    cmd = [sys.executable, os.path.join(ROOT, "src", "train.py"),
           "--out_dir", out_dir] + extra
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    ap.add_argument("--primary_lam", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=50000)
    ap.add_argument("--grid_dir", type=str, default=os.path.join(ROOT, "runs", "grid"))
    args = ap.parse_args()

    base = ["--epochs", str(args.epochs)]
    if args.phase == "A":
        for seed in args.seeds:
            s = ["--seed", str(seed)]
            run_one(os.path.join(args.grid_dir, f"baseline_s{seed}"),
                    ["--condition", "baseline"] + s + base)
            for lam in args.lams:
                run_one(os.path.join(args.grid_dir, f"supcon_true_lam{lam}_s{seed}"),
                        ["--condition", "supcon_true", "--lambda_con", str(lam)] + s + base)
                run_one(os.path.join(args.grid_dir, f"supcon_shuffled_lam{lam}_s{seed}"),
                        ["--condition", "supcon_shuffled", "--lambda_con", str(lam)] + s + base)
            run_one(os.path.join(args.grid_dir, f"grokfast_s{seed}"),
                    ["--condition", "grokfast"] + s + base)
    else:
        for seed in args.seeds:
            for lam in args.lams:
                traj = os.path.join(args.grid_dir, f"supcon_true_lam{lam}_s{seed}", "norms.npy")
                if not os.path.exists(traj):
                    print(f"missing norm trajectory: {traj}", file=sys.stderr)
                    continue
                run_one(os.path.join(args.grid_dir, f"norm_matched_lam{lam}_s{seed}"),
                        ["--condition", "norm_matched", "--seed", str(seed),
                         "--norm_traj", traj] + base)


if __name__ == "__main__":
    main()
