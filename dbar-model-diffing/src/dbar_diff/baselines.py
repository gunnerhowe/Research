"""Baseline model-comparison metrics: linear CKA (static geometry) and DSA
(deterministic-conjugacy dynamics, public dsa-metric package of Ostrow et al.,
arXiv:2306.10168).

Both are computed on hidden/residual state trajectories collected teacher-forced on a
shared task-sampled evaluation set, with the twin's noise semantics active — the same
condition the d̄ readout sees, so no metric is handed a cleaned-up strawman.
"""
from __future__ import annotations

import numpy as np

# One pre-registered DSA configuration for every pair (PLAN.md).
DSA_KW = dict(n_delays=8, delay_interval=1, rank=32, iters=1500, lr=5e-3,
              score_method="angular")


def linear_cka(X, Y):
    """Linear CKA between (N, d1) and (N, d2) activation matrices (column-centered)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xty = np.linalg.norm(X.T @ Y, "fro") ** 2
    xtx = np.linalg.norm(X.T @ X, "fro")
    yty = np.linalg.norm(Y.T @ Y, "fro")
    return float(xty / (xtx * yty))


def cka_states(states_a, states_b, skip=16):
    """CKA on (B, L, d) state tensors, transient dropped, flattened over (B, L)."""
    A = states_a[:, skip:].reshape(-1, states_a.shape[-1])
    B = states_b[:, skip:].reshape(-1, states_b.shape[-1])
    return linear_cka(A, B)


def dsa_distance(states_a, states_b, device="cuda", skip=16, verbose=False, **kw):
    """DSA angular Procrustes distance between two sets of state trajectories
    (B, L, d), via the public dsa-metric implementation."""
    from DSA import DSA
    cfg = {**DSA_KW, **kw}
    A = np.asarray(states_a[:, skip:], dtype=np.float64)
    B = np.asarray(states_b[:, skip:], dtype=np.float64)
    d = DSA(A, B, device=device, verbose=verbose, **cfg)
    return float(d.fit_score())
