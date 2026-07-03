"""One-command reproduction of every experiment in the paper, in order.

    python -u scripts/run_all.py            # everything (long; GPU)
    python -u scripts/run_all.py --skip-charlm

Each stage is resumable (completed runs are skipped), so re-running after an
interruption continues where it left off.
"""
import argparse
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    print(f"\n$$$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"!!! stage failed with code {r.returncode}: {' '.join(cmd)}", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--skip-charlm", action="store_true")
    ap.add_argument("--synthetic-steps", type=int, default=6000)
    ap.add_argument("--charlm-steps", type=int, default=10000)
    ap.add_argument("--charlm-seeds", nargs="+", default=["0", "1"])
    args = ap.parse_args()
    py = sys.executable

    if not args.skip_synthetic:
        # baselines: 3 seeds; ablations: 2 seeds (handled by the driver)
        run([py, "-u", "-m", "experiments.run_synthetic", "--ablations",
             "--steps", str(args.synthetic_steps), "--seeds", "0", "1", "2"])

    if not args.skip_charlm:
        run([py, "-u", "-m", "experiments.run_charlm", "--corpus", "enwik8",
             "--steps", str(args.charlm_steps), "--seeds", *args.charlm_seeds])

    run([py, "-u", "-m", "scripts.analyze"])
    run([py, "-u", "-m", "scripts.make_figures"])
    if os.path.exists(os.path.join(ROOT, "results", "charlm", "enwik8", "semrf__seed0.ckpt.pt")):
        run([py, "-u", "-m", "scripts.anchor_analysis"])
    print("\nALL STAGES DONE", flush=True)


if __name__ == "__main__":
    main()
