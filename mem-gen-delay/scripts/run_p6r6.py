"""P6 R6 trap-language fleet: trigram (TRI_MODES=2) rep seeds 1-10 + norep seeds 1-10.

Constants finalized after smoke v3 (seed 0 excluded); predictions P-T1..3 and K-T were
frozen at commit 4a5083f BEFORE any trigram fleet run. Idempotent.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, "runs", "grid6r6")
COMMON = ["--steps", "24000", "--lr", "1e-3", "--lang", "trigram"]


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
    for s in range(1, 11):
        run_one(f"rep_s{s}", ["--condition", "rep", "--seed", str(s)])
    for s in range(1, 11):
        run_one(f"norep_s{s}", ["--condition", "norep", "--seed", str(s)])


if __name__ == "__main__":
    main()
