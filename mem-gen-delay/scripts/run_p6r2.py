"""P6 R2/R3 fleet: 30 positives + 10 data-ablation negatives + 5 architectural negatives.

Pre-registered in plan_p6.md before launch; seed 0 (the smoke) excluded. Idempotent.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, "runs", "grid6r2")
COMMON = ["--steps", "16000", "--lr", "1e-3"]


def run_one(name, extra):
    out = os.path.join(GRID, name)
    if os.path.exists(os.path.join(out, "summary.json")):
        print(f"skip (done): {name}", flush=True)
        return
    cmd = [sys.executable, os.path.join(ROOT, "src", "train_lm.py"),
           "--out_dir", out] + COMMON + extra
    print(">>", name, flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    os.makedirs(GRID, exist_ok=True)
    for s in range(1, 31):
        run_one(f"rep_s{s}", ["--condition", "rep", "--seed", str(s)])
    for s in range(1, 11):
        run_one(f"norep_s{s}", ["--condition", "norep", "--seed", str(s)])
    for s in range(1, 6):
        run_one(f"onelayer_s{s}", ["--condition", "onelayer", "--seed", str(s)])


if __name__ == "__main__":
    main()
