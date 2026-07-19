"""P7 R1 fleet: scaffold-seeding causal test (prereg plan_p7.md commit b8eb219).

Arms: hard x10, seed x10, sink x5 (placebo), near x5 (exploratory), norephard x5
(data-necessity corner). Controls = grid6r2 rep (n=30), reused under bit-identity
guard K-C2b. Idempotent.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, "runs", "grid7c")
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
    for s in range(1, 11):
        run_one(f"hard_s{s}",
                ["--condition", "rep", "--seed", str(s), "--scaffold", "hard"])
    for s in range(1, 11):
        run_one(f"seed_s{s}",
                ["--condition", "rep", "--seed", str(s), "--scaffold", "seed"])
    for s in range(1, 6):
        run_one(f"sink_s{s}",
                ["--condition", "rep", "--seed", str(s), "--scaffold", "sink"])
    for s in range(1, 6):
        run_one(f"near_s{s}",
                ["--condition", "rep", "--seed", str(s), "--scaffold", "near"])
    for s in range(1, 6):
        run_one(f"norephard_s{s}",
                ["--condition", "norep", "--seed", str(s), "--scaffold", "hard"])
    print("P7 R1 fleet complete", flush=True)


if __name__ == "__main__":
    main()
