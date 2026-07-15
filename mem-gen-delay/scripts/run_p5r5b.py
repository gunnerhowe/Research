"""P5 R5b: confirmation runs for the label-free result (prereg in STATUS before launch).

9 MNIST runs with --log_spectra: seeds {10,11,12} x {baseline, supcon_aug, shufpair}.
Frozen thresholds under test: d.top1_frac >= 0.006197 (label-free) and cos_gap >= 0.1107
(task-aware), exactly as fit on seed 8 (out5/r5_scored.json). Idempotent.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, "runs", "grid5r5b")
COMMON = ["--loss", "mse", "--opt", "adamw", "--lr", "1e-3", "--wd", "0.01",
          "--init_scale", "8", "--batch_size", "200", "--steps", "100000",
          "--log_spectra"]


def run_one(name, extra):
    out = os.path.join(GRID, name)
    if os.path.exists(os.path.join(out, "summary.json")):
        print(f"skip (done): {name}", flush=True)
        return
    cmd = [sys.executable, os.path.join(ROOT, "src", "train_mnist.py"),
           "--out_dir", out] + COMMON + extra
    print(">>", name, flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    os.makedirs(GRID, exist_ok=True)
    for s in (10, 11, 12):
        sa = ["--seed", str(s)]
        run_one(f"baseline_s{s}", ["--condition", "baseline"] + sa)
        run_one(f"supcon_aug_s{s}",
                ["--condition", "supcon_aug", "--lambda_con", "0.3"] + sa)
        run_one(f"supcon_shufpair_s{s}",
                ["--condition", "supcon_shufpair", "--lambda_con", "0.3"] + sa)


if __name__ == "__main__":
    main()
