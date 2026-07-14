"""Paper 4 PRE-REGISTERED AMENDMENT grid (E2/E3/E4) — see STATUS.md prereg block 2026-07-14.

Committed BEFORE any of these runs start. Rationale: the free-norm race in runs/grid4 confounds
the prior's structure channel with its norm side-effect (supcon_aug's weight norm is FROZEN at
~92 by the aux-term/weight-decay equilibrium, while baseline's norm decays to ~66 before it
generalizes). These arms de-confound:

  E2  matched-norm sweep: clamp {35,50,65,80,92} x {baseline, supcon_aug} x seeds {0,3}
      (c23 cells already exist in runs/grid4). Decisive cell: base_c92 — does baseline EVER
      generalize at the norm where supcon_aug did?
  E3  content test: supcon_nn — label-free CROSS-EXAMPLE couples (greedy nearest-neighbor
      pixel matching; purity measured & reported). Free norm, seeds {0,3}.
  E4  dose: supcon_aug lambda {0.03,0.1}, seeds {0,3}.

--extend adds seeds {1,4} for E2+E3 (run only if the 2-seed pattern is clean).
Job order = decisiveness, so early reads are the informative cells.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, "runs", "grid4b")
COMMON = ["--loss", "mse", "--opt", "adamw", "--lr", "1e-3", "--wd", "0.01",
          "--init_scale", "8", "--batch_size", "200", "--steps", "100000"]
LAM = ["--lambda_con", "0.3"]


def run_one(name, extra):
    out = os.path.join(GRID, name)
    if os.path.exists(os.path.join(out, "summary.json")):
        print(f"skip (done): {name}", flush=True)
        return
    cmd = [sys.executable, os.path.join(ROOT, "src", "train_mnist.py"),
           "--out_dir", out] + COMMON + extra
    print(">>", name, flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def jobs(seeds):
    out = []
    # --- decisive first: the matched-norm 92 cells + the cross-example prior ---
    for s in seeds:
        sa = ["--seed", str(s)]
        out.append((f"base_c92_s{s}", ["--condition", "baseline", "--norm_clamp", "92"] + sa))
        out.append((f"aug_c92_s{s}", ["--condition", "supcon_aug", "--norm_clamp", "92"] + LAM + sa))
    for s in seeds:
        out.append((f"nn_s{s}", ["--condition", "supcon_nn"] + LAM + ["--seed", str(s)]))
    # --- E2 sweep, high norm -> low (informative cells first) ---
    for c in (80, 65, 50, 35):
        for s in seeds:
            sa = ["--seed", str(s)]
            out.append((f"base_c{c}_s{s}",
                        ["--condition", "baseline", "--norm_clamp", str(c)] + sa))
            out.append((f"aug_c{c}_s{s}",
                        ["--condition", "supcon_aug", "--norm_clamp", str(c)] + LAM + sa))
    # --- E4 dose (base seeds only; not part of --extend) ---
    if seeds == (0, 3):
        for lam in ("0.03", "0.1"):
            for s in seeds:
                out.append((f"aug_lam{lam}_s{s}",
                            ["--condition", "supcon_aug", "--lambda_con", lam,
                             "--seed", str(s)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extend", action="store_true",
                    help="run seeds {1,4} for E2+E3 (only after 2-seed pattern is clean)")
    args = ap.parse_args()
    os.makedirs(GRID, exist_ok=True)
    seeds = (1, 4) if args.extend else (0, 3)
    for name, extra in jobs(seeds):
        run_one(name, extra)


if __name__ == "__main__":
    main()
