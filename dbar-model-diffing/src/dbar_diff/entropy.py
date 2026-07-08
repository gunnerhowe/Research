"""Entropy-rate estimators, the estimation wall, and the Fano-type d̄ lower bound.

Ported from ornstein-dist, adapted to multi-chain streams (block counts pooled across
chains; LZ78 averaged over chains).

- conditional block entropy h_n = H_n − H_{n−1} converges to h from above; plug-in
  undersampling bias pulls it down once observed support ~ sample count — the guard
  keeps us before the knee.
- LZ78 gives an independent, slowly-converging estimate; agreement is the sanity check.
- The wall: empirical d̄_n is consistent only for n ≲ log2(N_windows)/h
  (Marton–Shields). k_wall(sym) is used to restrict every claim (PLAN.md K2).
- Fano: |h(μ)−h(ν)| ≤ H_b(d̄) + d̄·log2(m−1) ⇒ d̄ ≥ g⁻¹(|Δh|): a certified lower
  bound on d̄ from an entropy-rate gap, independent of the OT machinery.
"""
from __future__ import annotations

import numpy as np

from .dbar import _as_chains, encode_blocks

_LN2 = np.log(2.0)


def _plugin_entropy_bits(counts, miller_madow=True):
    n_tot = counts.sum()
    p = counts / n_tot
    h = -np.sum(p * np.log2(p))
    if miller_madow:
        h += (len(counts) - 1) / (2.0 * n_tot * _LN2)
    return h


def block_entropy_curve(sym, m, n_max=24, miller_madow=True):
    """H_n and h_n = H_n − H_{n−1} (bits) for n = 1..n_max, pooled across chains."""
    max_n = int(np.floor(52 / np.log2(m)))
    n_max = min(n_max, max_n)
    ns = np.arange(1, n_max + 1)
    H = np.empty(len(ns))
    support = np.empty(len(ns), dtype=np.int64)
    n_windows = np.empty(len(ns), dtype=np.int64)
    for i, n in enumerate(ns):
        codes = encode_blocks(sym, int(n), m)
        _, counts = np.unique(codes, return_counts=True)
        H[i] = _plugin_entropy_bits(counts, miller_madow)
        support[i] = len(counts)
        n_windows[i] = len(codes)
    h_cond = np.diff(H, prepend=0.0)
    return {"n": ns, "H": H, "h_cond": h_cond, "support": support,
            "n_windows": n_windows}


def entropy_rate(sym, m, n_max=24, support_frac_guard=0.02):
    """h_n at the largest n whose observed support ≤ guard × #windows.

    Returns (estimate_bits, n_used, curve_dict).
    """
    curve = block_entropy_curve(sym, m, n_max)
    ok = curve["support"] <= support_frac_guard * curve["n_windows"]
    if not ok.any():
        raise ValueError("no block length passes the undersampling guard")
    idx = np.where(ok)[0][-1]
    return curve["h_cond"][idx], int(curve["n"][idx]), curve


def estimation_wall(sym, m, **kw):
    """k_wall = log2(N_windows)/ĥ: the block length beyond which the empirical
    d̄_n estimate is no longer consistent. Returns (k_wall, h_hat)."""
    h, _, curve = entropy_rate(sym, m, **kw)
    n_sym = int(curve["n_windows"][0])
    return float(np.log2(n_sym) / max(h, 1e-9)), float(h)


def lz78_entropy(sym, m):
    """LZ78 incremental-parsing entropy-rate estimate in bits/symbol (chain-averaged,
    weighted by chain length)."""
    chains = _as_chains(sym)
    total_bits, total_len = 0.0, 0
    for c in chains:
        children: dict[tuple[int, int], int] = {}
        node = 0
        next_id = 1
        phrases = 0
        for s in np.asarray(c).tolist():
            key = (node, s)
            child = children.get(key)
            if child is None:
                children[key] = next_id
                next_id += 1
                phrases += 1
                node = 0
            else:
                node = child
        if node != 0:
            phrases += 1
        total_bits += phrases * (np.log2(max(phrases, 2)) + np.log2(m))
        total_len += len(c)
    return total_bits / total_len


def binary_entropy(delta):
    delta = np.asarray(delta, dtype=np.float64)
    out = np.zeros_like(delta)
    mask = (delta > 0) & (delta < 1)
    d = delta[mask]
    out[mask] = -d * np.log2(d) - (1 - d) * np.log2(1 - d)
    return out if out.ndim else float(out)


def fano_g(delta, m):
    """g(δ) = H_b(δ) + δ log2(m−1), increasing on [0, 1 − 1/m]."""
    return binary_entropy(delta) + delta * (np.log2(m - 1) if m > 2 else 0.0)


def fano_lower_bound(dh, m):
    """d̄ ≥ g⁻¹(|Δh|): certified lower bound on d̄ from an entropy-rate gap (bits)."""
    dh = abs(float(dh))
    if dh <= 0:
        return 0.0
    hi = 1.0 - 1.0 / m
    if dh >= fano_g(hi, m):
        return hi
    lo_x, hi_x = 0.0, hi
    for _ in range(80):
        mid = 0.5 * (lo_x + hi_x)
        if fano_g(mid, m) < dh:
            lo_x = mid
        else:
            hi_x = mid
    return 0.5 * (lo_x + hi_x)
