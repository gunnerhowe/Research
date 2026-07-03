"""KS secondary-system mini-grid: seeds 1-3 x {base, tpp, pois} (+shuf opt).

Same protocol as L96 grid (noisy obs, clean catalog refs, matched ft budget).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ["--data_dir", "data/ksr03", "--ref_dir", "data/ks"]
EVAL_DATA = ["--data_dir", "data/ks"]


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT)
    print(f"<<< exit {r.returncode} ({time.time()-t0:.0f}s)", flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"failed: {cmd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--conditions", nargs="+",
                    default=["base", "tpp", "pois", "marg"])
    ap.add_argument("--aux_weight", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ft_steps", type=int, default=800)
    args = ap.parse_args()

    for seed in args.seeds:
        base_ck = ROOT / f"runs/ks_grid_basephase_s{seed}/ckpt.pt"
        if not base_ck.exists():
            run([str(SRC / "train_surrogate.py"), "--phase", "base",
                 "--condition", "base", "--seed", str(seed),
                 "--out", str(base_ck.parent)] + DATA)
        for cond in args.conditions:
            out = ROOT / f"runs/ks_ft_{cond}_s{seed}"
            ck = out / "ckpt.pt"
            if not ck.exists():
                run([str(SRC / "train_surrogate.py"), "--phase", "ft",
                     "--condition", cond, "--seed", str(seed),
                     "--init", str(base_ck), "--out", str(out),
                     "--steps", str(args.ft_steps), "--lr", str(args.lr),
                     "--aux_weight", str(args.aux_weight),
                     "--stats", "ks"] + DATA)
            if not (out / "metrics.json").exists():
                run([str(SRC / "eval_surrogate.py"), "--ckpt", str(ck)]
                    + EVAL_DATA)
        if not (base_ck.parent / "metrics.json").exists():
            run([str(SRC / "eval_surrogate.py"), "--ckpt", str(base_ck)]
                + EVAL_DATA)


if __name__ == "__main__":
    main()
