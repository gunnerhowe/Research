"""Confirmatory experiment grid (single GPU, resumable via skip-if-exists).

Protocol (PLAN.md D5/D9/D11/D13): per seed — one shared FM base + one det
base (phase 1, noisy obs dt02r03); each condition fine-tunes the shared base
with identical budget (800 steps, lr 1e-4, lambda 10); eval vs clean dt02.

Usage: python scripts/run_grid.py            # seeds 1-5, core conditions
       python scripts/run_grid.py --seeds 0 --conditions shuf tpp_mle  # ablations
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
FT_CONDS = ["base", "tpp", "pois", "marg", "push", "det"]
DATA = ["--data_dir", "data/dt02r03", "--ref_dir", "data/dt02"]
EVAL_DATA = ["--data_dir", "data/dt02"]


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT)
    print(f"<<< exit {r.returncode} ({time.time()-t0:.0f}s)", flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"failed: {cmd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--conditions", nargs="+", default=FT_CONDS)
    ap.add_argument("--aux_weight", type=float, default=10.0)
    ap.add_argument("--marg_weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ft_steps", type=int, default=800)
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()

    for seed in args.seeds:
        base_ck = ROOT / f"runs/grid_basephase_s{seed}/ckpt.pt"
        if seed == 0:  # dev seed reuses the existing base
            base_ck = ROOT / "runs/dt02r03_base_s0/ckpt.pt"
        if not base_ck.exists():
            run([str(SRC / "train_surrogate.py"), "--phase", "base",
                 "--condition", "base", "--seed", str(seed),
                 "--out", str(base_ck.parent)] + DATA)
        det_ck = ROOT / f"runs/grid_detphase_s{seed}/ckpt.pt"
        if "det" in args.conditions and not det_ck.exists():
            run([str(SRC / "train_surrogate.py"), "--phase", "base",
                 "--condition", "det", "--seed", str(seed),
                 "--out", str(det_ck.parent)] + DATA)

        for cond in args.conditions:
            out = ROOT / f"runs/ft_{cond}_s{seed}"
            ck = out / "ckpt.pt"
            if not ck.exists():
                init = det_ck if cond == "det" else base_ck
                run([str(SRC / "train_surrogate.py"), "--phase", "ft",
                     "--condition", cond, "--seed", str(seed),
                     "--init", str(init), "--out", str(out),
                     "--steps", str(args.ft_steps),
                     "--lr", str(args.lr),
                     "--aux_weight", str(args.aux_weight),
                     "--marg_weight", str(args.marg_weight)] + DATA)
            if not args.skip_eval and not (out / "metrics.json").exists():
                run([str(SRC / "eval_surrogate.py"), "--ckpt", str(ck)]
                    + EVAL_DATA)
        # also eval the raw base (pre-ft anchor) once per seed
        if not args.skip_eval and not (base_ck.parent / "metrics.json").exists():
            run([str(SRC / "eval_surrogate.py"), "--ckpt", str(base_ck)]
                + EVAL_DATA)


if __name__ == "__main__":
    main()
