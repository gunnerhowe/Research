"""Shared experiment utilities: result IO, seeding, timing."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def save_json(name: str, obj) -> Path:
    path = RESULTS / name

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serializable: {type(o)}")

    path.write_text(json.dumps(obj, indent=1, default=default))
    print(f"[saved] {path}")
    return path


def load_json(name: str):
    return json.loads((RESULTS / name).read_text())


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        print(f"[{self.label}] {time.perf_counter() - self.t0:.1f}s")
