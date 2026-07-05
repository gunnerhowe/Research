"""Naive-tiling null (E3-i): periodically tile the small-domain statistics.

Strict tiling of the FIELD u_tile(x) = u_small(x mod L_small) implies, exactly:
  - power only at k that are multiples of 2 pi / L_small, with the small-domain
    one-sided powers P_m carried over unchanged (density comb S(k) = S_small(k) *
    L_target / L_small at comb points, zero elsewhere);
  - C(r) = the L_small-periodic extension of C_small(r);
  - per-sector resonances defined only at comb wavenumbers (the small-domain ones).
Interpolated single-size nulls (interp-22, interp-88) live in scaling.interp_null.
"""
from __future__ import annotations

import numpy as np


def tile_power(p_small, L_small, L_target):
    """One-sided powers on the target grid implied by strict tiling.

    Returns (p_target, comb_mask) with p_target[m] = p_small[m'] when
    k_m = 2 pi m / L_target is a multiple of 2 pi / L_small (m' the small index),
    else 0."""
    ratio = L_target / L_small
    r = int(round(ratio))
    if abs(ratio - r) > 1e-9:
        raise ValueError("L_target must be an integer multiple of L_small")
    n_small = p_small.shape[0]
    n_target = (n_small - 1) * r + 1
    p = np.zeros(n_target)
    p[::r] = p_small
    comb = np.zeros(n_target, dtype=bool)
    comb[::r] = True
    return p, comb


def tile_corr(c_small, L_small, r_target, dx):
    """C(r) on r_target from the periodic extension of the small-domain C(r),
    c_small given on r = dx * (0..N_small-1)."""
    n_small = c_small.shape[0]
    idx = np.round((np.asarray(r_target, float) % L_small) / dx).astype(int) % n_small
    return c_small[idx]
