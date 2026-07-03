"""Create noisy-observation variants of a data dir (train/val only; eval refs
stay clean, following Jiang et al. 2023: emulate the true system from noisy
observations). Noise: iid Gaussian, r * std(X)."""
import argparse
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/dt02")
ap.add_argument("--r", type=float, default=0.3)
args = ap.parse_args()

src = ROOT / args.src
dst = ROOT / f"{args.src}r{str(args.r).replace('.', '')}"
dst.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)

for split in ["train", "val"]:
    d = np.load(src / f"l96_{split}.npz")
    X = d["X"]
    Xn = X + rng.normal(0, args.r * X.std(), X.shape).astype(np.float32)
    np.savez_compressed(dst / f"l96_{split}.npz", X=Xn,
                        **{k: d[k] for k in d.files if k != "X"})
    print(f"{split}: noisy copy written (r={args.r})")

for split in ["eval", "eval_long"]:
    shutil.copy(src / f"l96_{split}.npz", dst / f"l96_{split}.npz")
    print(f"{split}: clean copy")
print(f"-> {dst}  (now run train_tpp.py --data_dir {dst.relative_to(ROOT)})")
