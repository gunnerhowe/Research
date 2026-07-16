"""P6 R7 (third config axis: new language) + R8 (gap-origin probe). Pre-registered in
plan_p6.md before these grids exist. Idempotent; R7 first (gate runs), then R8."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_one(grid, name, extra):
    out = os.path.join(ROOT, "runs", grid, name)
    if os.path.exists(os.path.join(out, "summary.json")):
        print(f"skip (done): {grid}/{name}", flush=True)
        return
    cmd = [sys.executable, os.path.join(ROOT, "src", "train_lm.py"),
           "--out_dir", out] + extra
    print(">>", grid, name, flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    # R7: new language (lang_seed 888), original config, frozen-rule gate
    for s in range(201, 211):
        run_one("grid6r7", f"rep_s{s}",
                ["--condition", "rep", "--seed", str(s), "--steps", "20000",
                 "--lr", "1e-3", "--lang_seed", "888"])
    for s in range(201, 204):
        run_one("grid6r7", f"norep_s{s}",
                ["--condition", "norep", "--seed", str(s), "--steps", "20000",
                 "--lr", "1e-3", "--lang_seed", "888"])
    # R8: gap-origin — lr axis at batch 64, batch axis at lr 1e-3
    for lr in ("5e-4", "2e-3"):
        for s in range(301, 306):
            run_one("grid6r8", f"lr{lr}_s{s}",
                    ["--condition", "rep", "--seed", str(s), "--steps", "32000",
                     "--lr", lr])
    for bs in ("32", "128"):
        for s in range(311, 316):
            run_one("grid6r8", f"b{bs}_s{s}",
                    ["--condition", "rep", "--seed", str(s), "--steps", "32000",
                     "--lr", "1e-3", "--batch_size", bs])


if __name__ == "__main__":
    main()
