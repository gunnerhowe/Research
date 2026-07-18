"""R-ESNb fleet: trigram-onelayer fourth-corner negatives + fresh-config prospective cell.

Pre-registered in plan_p6.md at commit 4ca549e BEFORE this grid existed. Idempotent.
Launch only when the GPU is idle (the user's other project has priority).
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, "runs", "grid6esnb")

FRESH = [
    ("fresh1", ["--condition", "rep", "--seed", "401", "--lr", "1.5e-3",
                "--d_model", "192", "--steps", "20000"]),
    ("fresh2", ["--condition", "rep", "--seed", "402", "--lr", "8e-4",
                "--d_model", "288", "--steps", "32000"]),
    ("fresh3", ["--condition", "rep", "--seed", "403", "--lr", "1e-3",
                "--d_model", "224", "--p_rep", "0.45", "--steps", "20000"]),
    ("fresh4", ["--condition", "rep", "--seed", "404", "--lr", "6e-4",
                "--batch_size", "96", "--steps", "32000"]),
    ("fresh5", ["--condition", "rep", "--seed", "405", "--lr", "1.2e-3",
                "--d_model", "320", "--p_rep", "0.7", "--steps", "20000"]),
]


def run_one(name, extra):
    out = os.path.join(GRID, name)
    if os.path.exists(os.path.join(out, "summary.json")):
        print(f"skip (done): {name}", flush=True)
        return
    cmd = [sys.executable, os.path.join(ROOT, "src", "train_lm.py"),
           "--out_dir", out] + extra
    print(">>", name, flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    os.makedirs(GRID, exist_ok=True)
    # TRAPONE: grid6r6 rep config except architecture (the fourth corner)
    for s in range(1, 11):
        run_one(f"trapone_s{s}",
                ["--condition", "onelayer", "--seed", str(s), "--lang", "trigram",
                 "--steps", "24000", "--lr", "1e-3"])
    for name, extra in FRESH:
        run_one(name, extra)
    # FRESH negatives: same configs, norep, seeds 411..415
    for i, (name, extra) in enumerate(FRESH):
        neg = [("norep" if v == "rep" else v) for v in extra]
        neg[neg.index("--seed") + 1] = str(411 + i)
        run_one(f"freshn{i + 1}", neg)
    print("R-ESNb fleet complete", flush=True)


if __name__ == "__main__":
    main()
