"""P6 R5: blind ship-gate runs — never-seen config (d_model 320, p_rep 0.6).

Pre-registered in plan_p6.md with the frozen forecaster BEFORE these runs exist.
Idempotent.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, "runs", "grid6r5")
COMMON = ["--steps", "20000", "--lr", "1e-3", "--d_model", "320", "--p_rep", "0.6"]


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
    for s in range(101, 111):
        run_one(f"rep_s{s}", ["--condition", "rep", "--seed", str(s)])
    for s in range(101, 104):
        run_one(f"norep_s{s}", ["--condition", "norep", "--seed", str(s)])


if __name__ == "__main__":
    main()
