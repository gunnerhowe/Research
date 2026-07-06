"""Continual-learning testbeds.

Two task streams, both single-head (domain-incremental) so that later tasks
overwrite the shared readout unless consolidation intervenes -- the regime where
a consolidation mechanism is visible.

  split_mnist_domain(): 5 binary tasks (0v1, 2v3, 4v5, 6v7, 8v9), each mapped to
      a shared {0,1} head. Standard Split-MNIST domain-IL.
  yin_yang(): the Yin-Yang dataset (Kriener et al. 2022), the BrainScaleS group's
      own small benchmark; continual variant = the pattern rotated per task, one
      shared 3-way head. Procedural (no download), so the BSS-2-native testbed.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch

# candidate locations for a pre-downloaded MNIST (avoid a network fetch if we can)
_MNIST_CANDIDATES = [
    Path.home() / "data",
    Path(__file__).resolve().parents[2] / "data",
    Path("E:/GitHub/Research/energy-effic/data"),
]


def _mnist_root():
    for c in _MNIST_CANDIDATES:
        if (c / "MNIST" / "raw").exists() or (c / "MNIST" / "processed").exists():
            return str(c)
    return str(Path(__file__).resolve().parents[2] / "data")


def load_mnist(train=True):
    """Return (X, y): X float32 (N,784) in [0,1], y int64 (N,). Uses a local copy
    if present, else downloads once."""
    from torchvision import datasets
    root = _mnist_root()
    try:
        ds = datasets.MNIST(root, train=train, download=False)
    except Exception:
        ds = datasets.MNIST(root, train=train, download=True)
    X = ds.data.reshape(-1, 784).float() / 255.0
    y = ds.targets.long()
    return X, y


_SPLIT_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def split_mnist_domain(n_per_class_train=None, seed=0):
    """5 domain-IL tasks. Each task t: digits _SPLIT_PAIRS[t], relabeled {0,1}
    (even index -> 0, odd -> 1), shared 2-way head. Returns a list of dicts with
    Xtr, ytr, Xte, yte (torch tensors on CPU)."""
    Xtr, ytr = load_mnist(train=True)
    Xte, yte = load_mnist(train=False)
    rng = np.random.default_rng(seed)
    tasks = []
    for (a, b) in _SPLIT_PAIRS:
        def _sel(X, y, subsample):
            idx = torch.nonzero((y == a) | (y == b), as_tuple=False).squeeze(1)
            lbl = (y[idx] == b).long()          # a->0, b->1
            Xs, ls = X[idx], lbl
            if subsample is not None:
                order = rng.permutation(Xs.shape[0])[: 2 * subsample]
                Xs, ls = Xs[order], ls[order]
            return Xs, ls
        xtr, ltr = _sel(Xtr, ytr, n_per_class_train)
        xte, lte = _sel(Xte, yte, None)
        tasks.append({"Xtr": xtr, "ytr": ltr, "Xte": xte, "yte": lte,
                      "name": f"{a}v{b}", "n_classes": 2})
    return tasks


# --------------------------------------------------------------------------- #
#  Yin-Yang (Kriener, Göltz, Petrovici 2022) + continual (rotated) variant     #
# --------------------------------------------------------------------------- #

def _yin_yang_class(x, y, r_big=0.5, r_small=0.1):
    """Class of point (x,y) in [0,1]^2 for a unit Yin-Yang centred at (0.5,0.5).
    0 = yin, 1 = yang, 2 = dot."""
    cx, cy = 0.5, 0.5
    dx, dy = x - cx, y - cy
    d = math.hypot(dx, dy)
    if d > r_big:
        return -1                                # outside the disk (rejected)
    # the two small dots
    d_small_top = math.hypot(x - cx, y - (cy + r_big / 2))
    d_small_bot = math.hypot(x - cx, y - (cy - r_big / 2))
    if d_small_top < r_small:
        return 1                                 # dot in the yin lobe -> yang
    if d_small_bot < r_small:
        return 0
    # the S-curve boundary: two half-disks of radius r_big/2
    is_yin = (d_small_bot < r_big / 2) or (d_small_top >= r_big / 2 and x < cx)
    # canonical Kriener construction:
    if d_small_top < r_big / 2:
        return 0                                 # upper half-disk -> yin
    if d_small_bot < r_big / 2:
        return 1                                 # lower half-disk -> yang
    return 0 if x < cx else 1


def _gen_yin_yang(n, rng, rot=0.0):
    """Sample n accepted points; features = (x, y, 1-x, 1-y) after rotating the
    sampling frame by `rot` radians about the centre (the per-task shift)."""
    pts, lbl = [], []
    c, s = math.cos(rot), math.sin(rot)
    while len(pts) < n:
        x, y = rng.random(), rng.random()
        # rotate about centre to define the (rotated) class geometry
        rx = 0.5 + c * (x - 0.5) - s * (y - 0.5)
        ry = 0.5 + s * (x - 0.5) + c * (y - 0.5)
        cls = _yin_yang_class(rx, ry)
        if cls < 0:
            continue
        pts.append([x, y, 1.0 - x, 1.0 - y])
        lbl.append(cls)
    return (torch.tensor(pts, dtype=torch.float32),
            torch.tensor(lbl, dtype=torch.long))


def yin_yang(n_tasks=5, n_train=2000, n_test=1000, seed=0, max_rot=math.pi / 2):
    """Continual Yin-Yang: n_tasks rotations of the pattern evenly spaced in
    [0, max_rot], shared 3-way head. Rotating the geometry moves the decision
    boundary, creating genuine cross-task interference."""
    rng = np.random.default_rng(seed)
    tasks = []
    rots = np.linspace(0.0, max_rot, n_tasks)
    for t, rot in enumerate(rots):
        xtr, ytr = _gen_yin_yang(n_train, rng, rot=float(rot))
        xte, yte = _gen_yin_yang(n_test, rng, rot=float(rot))
        tasks.append({"Xtr": xtr, "ytr": ytr, "Xte": xte, "yte": yte,
                      "name": f"rot{math.degrees(rot):.0f}", "n_classes": 3})
    return tasks


def get_tasks(name, **kw):
    if name == "split_mnist":
        return split_mnist_domain(**kw)
    if name == "yin_yang":
        return yin_yang(**kw)
    raise ValueError(f"unknown testbed {name}")


def input_dim(name):
    return {"split_mnist": 784, "yin_yang": 4}[name]


def n_head(name):
    return {"split_mnist": 2, "yin_yang": 3}[name]
