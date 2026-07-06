"""Shared configuration for the experiment ladder. The fixed operating point is
the one pre-registered in PLAN.md (calibrated once, before the gated seeds)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- pre-registered fixed operating point (PLAN.md) ---------------------------
PRIMARY = "split_mnist"
SIGMAS = [0.0, 0.005, 0.01, 0.02, 0.035, 0.05, 0.08, 0.12, 0.2, 0.35]
HEADLINE_SEEDS = list(range(8))
ABLATION_SEEDS = list(range(5))

FIXED = dict(
    lr_task=0.1, lr_c=0.1, epochs=2, batch_size=128,
    barrier_scale=0.2, kappa=1.0, anchor_strength=1.0, decay=1.0,
    fisher_batches=8, hidden=100, n_layers=2,
)
TASKS_KW = dict(n_per_class_train=1000)

# EWC uses a stronger static anchor; lambda scanned in E3, default here:
EWC_ANCHOR = 4.0


def save(name, obj):
    path = RESULTS / name
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    print(f"[saved] {path}  ({path.stat().st_size/1024:.1f} KB)")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def stamp():
    """Environment stamp for reproducibility (no timestamps: keep results
    byte-stable on re-run)."""
    return {
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device": DEVICE,
        "numpy": np.__version__,
    }
