"""P5 R5: label-free probe battery runs (pre-registered in STATUS.md before launch).

10 MNIST runs with --log_spectra (label-free penultimate-spectrum stats: effective rank,
participation ratio, top-1 eigenfraction), fresh seeds {8, 9}, five arms spanning positives,
a structural negative, and a censoring-noise negative. Idempotent.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, "runs", "grid5r5")
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
    for s in (8, 9):
        sa = ["--seed", str(s)]
        run_one(f"baseline_s{s}", ["--condition", "baseline"] + sa)
        run_one(f"supcon_aug_s{s}",
                ["--condition", "supcon_aug", "--lambda_con", "0.3"] + sa)
        run_one(f"supcon_label_s{s}",
                ["--condition", "supcon_label", "--lambda_con", "0.3"] + sa)
        run_one(f"supcon_shufpair_s{s}",
                ["--condition", "supcon_shufpair", "--lambda_con", "0.3"] + sa)
        run_one(f"base_c92_s{s}",
                ["--condition", "baseline", "--norm_clamp", "92"] + sa)


if __name__ == "__main__":
    main()
