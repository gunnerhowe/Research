"""Resume the final compute queue after session restart.

1. Retry ks_ft_margtpp seeds 2-3 once each (tolerate failure: the margtpp
   row can stand as a single-seed exploratory probe if CUDA instability
   persists).
2. Complete the L96 grid seeds 6-8 (resumable).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, tolerate=False):
    print(f">>> {' '.join(cmd)}", flush=True)
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT)
    print(f"<<< exit {r.returncode}", flush=True)
    if r.returncode != 0 and not tolerate:
        raise RuntimeError(cmd)
    return r.returncode == 0


for s in (2, 3):
    out = ROOT / f"runs/ks_ft_margtpp_s{s}"
    if (out / "metrics.json").exists():
        continue
    ok = run(["src/train_surrogate.py", "--phase", "ft", "--condition",
              "margtpp", "--seed", str(s),
              "--init", f"runs/ks_grid_basephase_s{s}/ckpt.pt",
              "--data_dir", "data/ksr03", "--ref_dir", "data/ks",
              "--steps", "800", "--lr", "1e-4", "--aux_weight", "3",
              "--self_tpp_steps", "25", "--marg_weight", "1",
              "--stats", "ks", "--b_roll", "8", "--sink_pts", "512",
              "--out", str(out)], tolerate=True)
    if ok:
        run(["src/eval_surrogate.py", "--ckpt", str(out / "ckpt.pt"),
             "--data_dir", "data/ks"], tolerate=True)
    else:
        print(f"!! margtpp seed {s} failed again — proceeding without it",
              flush=True)

run(["scripts/run_grid.py", "--seeds", "6", "7", "8"])
print("ALL DONE", flush=True)
