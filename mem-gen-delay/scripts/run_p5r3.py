"""P5 R3: prospective-validation runs (pre-registered in STATUS.md before launch).

20 new MNIST runs the frozen forecasters have never seen: fresh seeds {5,6,7}, one unseen
pin (clamp 60), one unseen dose (lam 0.05), structural negatives (shufpair), and two
censoring-noise negatives (base_c92). Frozen artifacts and predictions P-R3a..d are
committed before any of these runs start; idempotent (skip if summary.json exists).
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, "runs", "grid5r3")
COMMON = ["--loss", "mse", "--opt", "adamw", "--lr", "1e-3", "--wd", "0.01",
          "--init_scale", "8", "--batch_size", "200", "--steps", "100000"]


def run_one(name, extra):
    out = os.path.join(GRID, name)
    if os.path.exists(os.path.join(out, "summary.json")):
        print(f"skip (done): {name}", flush=True)
        return
    cmd = [sys.executable, os.path.join(ROOT, "src", "train_mnist.py"),
           "--out_dir", out] + COMMON + extra
    print(">>", name, flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def jobs():
    out = []
    for s in (5, 6, 7):
        sa = ["--seed", str(s)]
        out.append((f"baseline_s{s}", ["--condition", "baseline"] + sa))
        out.append((f"supcon_aug_s{s}",
                    ["--condition", "supcon_aug", "--lambda_con", "0.3"] + sa))
        out.append((f"supcon_label_s{s}",
                    ["--condition", "supcon_label", "--lambda_con", "0.3"] + sa))
        out.append((f"supcon_shufpair_s{s}",
                    ["--condition", "supcon_shufpair", "--lambda_con", "0.3"] + sa))
        out.append((f"base_clamp60_s{s}",
                    ["--condition", "baseline", "--norm_clamp", "60"] + sa))
        out.append((f"aug_lam0.05_s{s}",
                    ["--condition", "supcon_aug", "--lambda_con", "0.05"] + sa))
    for s in (5, 6):
        out.append((f"base_c92_s{s}",
                    ["--condition", "baseline", "--norm_clamp", "92", "--seed", str(s)]))
    return out


def main():
    os.makedirs(GRID, exist_ok=True)
    for name, extra in jobs():
        run_one(name, extra)


if __name__ == "__main__":
    main()
