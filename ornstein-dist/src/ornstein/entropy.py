"""Entropy-rate estimators and the Fano-type d̄ lower bound.

Two estimators bracket the answer in practice:
- conditional block entropy h_n = H_n - H_{n-1} (converges to h from above in n; plug-in
  undersampling bias pulls it DOWN once observed support ~ sample count, so read the curve
  before the bias knee),
- LZ78 phrase parsing, ĥ = C(log2 C + log2 m)/N (consistent, slowly converging, independent
  machinery — agreement is the sanity signal).

Fano bound (see docs/RESEARCH_NOTES.md §2): |h(μ)-h(ν)| ≤ H_b(d̄) + d̄·log2(m-1), hence
d̄ ≥ g⁻¹(|Δh|) with g(δ) = H_b(δ) + δ·log2(m-1). This corrects the spec's `d̄ ≥ |Δh|`.
"""
from __future__ import annotations

import numpy as np

from .dbar import encode_blocks

_LN2 = np.log(2.0)


def _plugin_entropy_bits(counts, miller_madow=True):
    n_tot = counts.sum()
    p = counts / n_tot
    h = -np.sum(p * np.log2(p))
    if miller_madow:
        h += (len(counts) - 1) / (2.0 * n_tot * _LN2)
    return h


def block_entropy_curve(sym, m, n_max=24, miller_madow=True):
    """H_n and h_n = H_n - H_{n-1} (bits) for n = 1..n_max, with support diagnostics.

    Returns dict of arrays: n, H, h_cond, support, n_windows.
    """
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
    """Point estimate: h_n at the largest n whose observed support ≤ guard × #windows.

    Returns (estimate_bits, n_used, curve_dict).
    """
    curve = block_entropy_curve(sym, m, n_max)
    ok = curve["support"] <= support_frac_guard * curve["n_windows"]
    if not ok.any():
        raise ValueError("no block length passes the undersampling guard")
    idx = np.where(ok)[0][-1]
    return curve["h_cond"][idx], int(curve["n"][idx]), curve


def lz78_entropy(sym, m):
    """LZ78 incremental-parsing entropy-rate estimate in bits/symbol."""
    sym = np.asarray(sym)
    children: dict[tuple[int, int], int] = {}
    node = 0
    next_id = 1
    phrases = 0
    for s in sym.tolist():
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
        phrases += 1  # trailing incomplete phrase
    c = phrases
    return c * (np.log2(max(c, 2)) + np.log2(m)) / len(sym)


def rigorous_gap_lb(curve_x, curve_y, m, support_frac_guard=0.02):
    """Best certified d̄ lower bound from block entropies at FINITE n.

    The Fano argument at block length n gives |H_n(X)-H_n(Y)|/n <= g(d̄_n) <= g(d̄),
    valid for every n (no entropy-rate limit needed). Returns (lb, n_at_max, gap_bits).
    """
    lb_best, n_best, gap_best = 0.0, 0, 0.0
    for i, n in enumerate(curve_x["n"]):
        ok_x = curve_x["support"][i] <= support_frac_guard * curve_x["n_windows"][i]
        ok_y = curve_y["support"][i] <= support_frac_guard * curve_y["n_windows"][i]
        if not (ok_x and ok_y):
            continue
        gap = abs(curve_x["H"][i] - curve_y["H"][i]) / n
        lb = fano_lower_bound(gap, m)
        if lb > lb_best:
            lb_best, n_best, gap_best = lb, int(n), float(gap)
    return lb_best, n_best, gap_best


def binary_entropy(delta):
    delta = np.asarray(delta, dtype=np.float64)
    out = np.zeros_like(delta)
    mask = (delta > 0) & (delta < 1)
    d = delta[mask]
    out[mask] = -d * np.log2(d) - (1 - d) * np.log2(1 - d)
    return out if out.ndim else float(out)


def fano_g(delta, m):
    """g(δ) = H_b(δ) + δ log2(m-1), increasing on [0, 1 - 1/m]."""
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
