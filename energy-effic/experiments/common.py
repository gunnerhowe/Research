"""Shared experiment utilities (house convention, mirrors kac-rice)."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
CKPT = ROOT / "data" / "ckpt"
CKPT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_json(obj, path):
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.tolist()
        raise TypeError(type(o))

    Path(path).write_text(json.dumps(obj, indent=1, default=default))
    print(f"wrote {path}")


def load_json(path):
    return json.loads(Path(path).read_text())


def log(msg):
    print(msg, flush=True)
