"""P7 R2 fleet: dose sweep + placement control (prereg plan_p7.md commit 1affb4a).

DOSE dose{b}_s21..25 for b in {1,2,3,4,6}: hard pattern, layer 0 head 0.
PLACEMENT hardL1_s21..25: identical +8 prev-token bias on LAYER 1 head 0.
beta=8 endpoint is grid7c hard; beta=0 endpoint is grid6r2 rep. Idempotent.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, "runs", "grid7d")
COMMON = ["--steps", "16000", "--lr", "1e-3", "--condition", "rep", "--log_heads"]


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
    for b in (1, 2, 3, 4, 6):
        for s in range(21, 26):
            run_one(f"dose{b}_s{s}",
                    ["--seed", str(s), "--scaffold", "hard", "--scaffold_beta", str(b)])
    for s in range(21, 26):
        run_one(f"hardL1_s{s}",
                ["--seed", str(s), "--scaffold", "hard", "--scaffold_beta", "8",
                 "--scaffold_layer", "1"])
    print("P7 R2 fleet complete", flush=True)


if __name__ == "__main__":
    main()
