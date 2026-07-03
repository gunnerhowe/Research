"""Ornstein d̄ estimation: exact OT between empirical n-block distributions under
normalized Hamming cost (POT network simplex).

Facts the estimator relies on (docs/RESEARCH_NOTES.md §1):
- d̄_n ≤ d̄ for all n, and d̄_n is non-decreasing along doubling n; d̄ = sup_n d̄_n.
  So the plateau of the curve is a conservative estimate of d̄.
- d̄_1 = TV between symbol marginals: a surrogate matching the partition marginal has
  d̄_1 ≈ 0 and all signal appears at n ≥ 2 — the static/dynamic split, per n.

Honesty device: empirical OT between two independent samples of the SAME process is > 0
at finite N. Every pair estimate is reported alongside truth-vs-truth and
surrogate-vs-surrogate floors computed with the identical procedure and budget.
"""
from __future__ import annotations

import numpy as np
import ot


def encode_blocks(sym, n, m):
    """Encode all overlapping n-blocks of a symbol stream as base-m integer codes.

    Exact for m**n < 2**53 (float64 convolution). Returns uint64 array, length N-n+1.
    """
    sym = np.asarray(sym)
    if n * np.log2(m) > 52:
        raise ValueError(f"n={n} too large for exact base-{m} encoding")
    if n == 1:
        return sym.astype(np.uint64)
    kernel = (float(m) ** np.arange(n)).astype(np.float64)
    codes = np.convolve(sym.astype(np.float64), kernel, mode="valid")
    return np.rint(codes).astype(np.uint64)


def _decode_digits(codes, n, m):
    digits = np.empty((len(codes), n), dtype=np.int8)
    c = codes.copy()
    for j in range(n):
        digits[:, j] = (c % m).astype(np.int8)
        c //= m
    return digits


def hamming_matrix(u1, u2, n, m):
    """Pairwise normalized Hamming distance between two arrays of block codes."""
    if m == 2:
        x = u1[:, None] ^ u2[None, :]
        return np.bitwise_count(x).astype(np.float64) / n
    d1 = _decode_digits(u1, n, m)
    d2 = _decode_digits(u2, n, m)
    return (d1[:, None, :] != d2[None, :, :]).mean(axis=2)


def _support(codes):
    uniq, counts = np.unique(codes, return_counts=True)
    return uniq, counts.astype(np.float64) / len(codes)


def dbar_n(sym_p, sym_q, n, m, n_blocks=2000, max_full_support=4096,
           seed=0, force_regime=None):
    """Estimate d̄_n between two symbol streams.

    Regime 'full': exact empirical block distributions (used when both supports are small).
    Regime 'sampled': n_blocks random windows per side (point-cloud OT) — used when the
    support would blow up the cost matrix. Floors must be computed with the same regime
    (pass force_regime) so signal and floor share the sampling budget.

    Returns (value, info dict).
    """
    codes_p = encode_blocks(sym_p, n, m)
    codes_q = encode_blocks(sym_q, n, m)
    up_full, wp_full = _support(codes_p)
    uq_full, wq_full = _support(codes_q)
    regime = force_regime
    if regime is None:
        regime = ("full" if len(up_full) <= max_full_support
                  and len(uq_full) <= max_full_support else "sampled")
    if regime == "full":
        up, wp, uq, wq = up_full, wp_full, uq_full, wq_full
    else:
        rng = np.random.default_rng(seed)
        sp = rng.choice(codes_p, size=min(n_blocks, len(codes_p)), replace=False)
        sq = rng.choice(codes_q, size=min(n_blocks, len(codes_q)), replace=False)
        up, wp = _support(sp)
        uq, wq = _support(sq)
    M = hamming_matrix(up, uq, n, m)
    val = ot.emd2(wp, wq, M, numItermax=2_000_000)
    info = {"regime": regime, "support_p": len(up), "support_q": len(uq),
            "full_support_p": len(up_full), "full_support_q": len(uq_full)}
    return float(val), info


def dbar_curve(sym_p, sym_q, m, ns=(1, 2, 4, 8, 16, 32), n_blocks=2000,
               repeats=4, seed=0):
    """d̄_n curve with per-n floors (truth-vs-truth and surrogate-vs-surrogate on
    disjoint halves) and repeat-based error bars in the sampled regime.

    Returns list of row dicts.
    """
    sym_p = np.asarray(sym_p)
    sym_q = np.asarray(sym_q)
    p1, p2 = np.array_split(sym_p, 2)
    q1, q2 = np.array_split(sym_q, 2)
    max_n = int(np.floor(52 / np.log2(m)))
    rows = []
    for n in ns:
        if n > max_n:
            continue
        vals = []
        v0, info = dbar_n(sym_p, sym_q, n, m, n_blocks=n_blocks, seed=seed)
        vals.append(v0)
        reg = info["regime"]
        n_rep = repeats if reg == "sampled" else 1
        for r in range(1, n_rep):
            v, _ = dbar_n(sym_p, sym_q, n, m, n_blocks=n_blocks,
                          seed=seed + 1000 * r, force_regime=reg)
            vals.append(v)
        floor_pp = [dbar_n(p1, p2, n, m, n_blocks=n_blocks,
                           seed=seed + 7 + 1000 * r, force_regime=reg)[0]
                    for r in range(n_rep)]
        floor_qq = [dbar_n(q1, q2, n, m, n_blocks=n_blocks,
                           seed=seed + 13 + 1000 * r, force_regime=reg)[0]
                    for r in range(n_rep)]
        rows.append({
            "n": int(n), "dbar": float(np.mean(vals)), "dbar_std": float(np.std(vals)),
            "floor_pp": float(np.mean(floor_pp)), "floor_qq": float(np.mean(floor_qq)),
            "floor": float(max(np.mean(floor_pp), np.mean(floor_qq))),
            "regime": reg, **{k: info[k] for k in ("support_p", "support_q",
                                                   "full_support_p", "full_support_q")},
        })
    return rows


def plateau_estimate(rows):
    """Conservative d̄ estimate from a curve: max over n of (d̄_n), with its floor.

    Since every true d̄_n ≤ d̄, the max of the estimates (noise floor subtracted by the
    reader — we report both) is the natural summary.
    """
    best = max(rows, key=lambda r: r["dbar"])
    return best["dbar"], best["floor"], best["n"]
