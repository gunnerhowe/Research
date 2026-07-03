"""Partitions mapping continuous states to symbol streams.

The partition is a confound (spec risk #1): always build partition edges from the TRUTH
data and apply the identical partition to every surrogate.
"""
from __future__ import annotations

import numpy as np


def sign_symbols(x):
    """Binary partition: sign of a scalar series. Returns (symbols int8, alphabet size)."""
    return (np.asarray(x) > 0).astype(np.int8), 2


def quantile_edges(x_ref, k):
    """Interior quantile edges (k-1 of them) computed on reference (truth) data."""
    return np.quantile(np.asarray(x_ref), np.linspace(0, 1, k + 1)[1:-1])


def apply_edges(x, edges):
    """Symbolize a scalar series with precomputed edges. Returns (symbols int8, alphabet)."""
    return np.searchsorted(edges, np.asarray(x)).astype(np.int8), len(edges) + 1


def box_symbols_xz(X, x_edges, z_edges):
    """Coarse box partition on (x, z) coordinates of a (N,3) state array.

    Alphabet = (len(x_edges)+1) * (len(z_edges)+1). Edges come from truth data.
    """
    ix = np.searchsorted(x_edges, X[:, 0])
    iz = np.searchsorted(z_edges, X[:, 2])
    m = (len(x_edges) + 1) * (len(z_edges) + 1)
    return (ix * (len(z_edges) + 1) + iz).astype(np.int8), m
