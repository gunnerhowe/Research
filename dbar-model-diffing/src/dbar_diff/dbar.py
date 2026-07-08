"""Ornstein d-bar estimation: exact OT between empirical n-block distributions under
normalized Hamming cost (POT network simplex).

Ported from the ornstein-dist project ("Beyond the Invariant Measure") and extended to
MULTI-CHAIN streams: a symbol stream here is a list of independent stationary chains
(free-running generation runs many chains in parallel); n-blocks are pooled across
chains and never cross a chain boundary.

Facts the estimator relies on:
- d̄_n ≤ d̄ for all n, and the empirical d̄_n is consistent only for
  n ≲ log2(N_windows)/h (Marton–Shields); beyond that wall it degrades silently.
  All claims are made at n below the wall (PLAN.md, K2).
- d̄_1 = total variation between symbol marginals: the static/dynamic split per n —
  a pair with d̄_1 ≈ 0 but d̄_n large at n ≥ 2 has the same "invariant measure" at the
  readout but is a different process.

Honesty device: empirical OT between two independent samples of the SAME process is > 0
at finite N. Every pair estimate is reported alongside same-process floors computed with
the identical procedure, regime, and budget.
"""
from __future__ import annotations

import numpy as np
import ot


def _as_chains(sym):
    """Normalize input to a list of 1-D int arrays (chains)."""
    if isinstance(sym, np.ndarray) and sym.ndim == 2:
        return [sym[i] for i in range(sym.shape[0])]
    if isinstance(sym, (list, tuple)):
        return [np.asarray(c) for c in sym]
    return [np.asarray(sym)]


def encode_blocks(chains, n, m):
    """Encode all overlapping n-blocks (per chain, pooled) as base-m integer codes.

    Exact for m**n < 2**53 (float64 convolution). Returns uint64 array.
    """
    chains = _as_chains(chains)
    if n * np.log2(m) > 52:
        raise ValueError(f"n={n} too large for exact base-{m} encoding")
    if n == 1:
        return np.concatenate([c.astype(np.uint64) for c in chains])
    kernel = (float(m) ** np.arange(n)).astype(np.float64)
    out = []
    for c in chains:
        if len(c) < n:
            continue
        codes = np.convolve(c.astype(np.float64), kernel, mode="valid")
        out.append(np.rint(codes).astype(np.uint64))
    return np.concatenate(out)


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


def dbar_n_codes(codes_p, codes_q, n, m, n_blocks=2000, max_full_support=4096,
                 seed=0, force_regime=None):
    """d̄_n from precomputed block-code arrays. Returns (value, info)."""
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


def dbar_n(sym_p, sym_q, n, m, **kw):
    """Estimate d̄_n between two (multi-chain) symbol streams."""
    return dbar_n_codes(encode_blocks(sym_p, n, m), encode_blocks(sym_q, n, m),
                        n, m, **kw)


def _mean_dbar(codes_a, codes_b, n, m, regime, n_blocks, repeats, seed0):
    vals = []
    n_rep = repeats if regime == "sampled" else 1
    for r in range(n_rep):
        v, _ = dbar_n_codes(codes_a, codes_b, n, m, n_blocks=n_blocks,
                            seed=seed0 + 1000 * r, force_regime=regime)
        vals.append(v)
    return float(np.mean(vals)), float(np.std(vals))


def dbar_pair_curve(sym_a, sym_b, sym_a2, sym_b2, m,
                    ns=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
                    n_blocks=2000, repeats=4, seed=0):
    """d̄_n curve for a model pair, with same-process floors from independent runs.

    sym_a/sym_b: one generation run of model A / model B (multi-chain).
    sym_a2/sym_b2: an INDEPENDENT generation run of each model (different sampling
    seed) — floor_n(A) = d̄_n(a, a2), floor_n(B) = d̄_n(b, b2), same regime and budget.

    Returns list of row dicts.
    """
    max_n = int(np.floor(52 / np.log2(m)))
    rows = []
    for n in ns:
        if n > max_n:
            continue
        ca = encode_blocks(sym_a, n, m)
        cb = encode_blocks(sym_b, n, m)
        ca2 = encode_blocks(sym_a2, n, m)
        cb2 = encode_blocks(sym_b2, n, m)
        v0, info = dbar_n_codes(ca, cb, n, m, n_blocks=n_blocks, seed=seed)
        reg = info["regime"]
        val, val_std = _mean_dbar(ca, cb, n, m, reg, n_blocks, repeats, seed)
        fa, _ = _mean_dbar(ca, ca2, n, m, reg, n_blocks, repeats, seed + 7)
        fb, _ = _mean_dbar(cb, cb2, n, m, reg, n_blocks, repeats, seed + 13)
        rows.append({
            "n": int(n), "dbar": val, "dbar_std": val_std,
            "floor_a": fa, "floor_b": fb, "floor": float(max(fa, fb)),
            "delta": float(val - max(fa, fb)),
            "regime": reg,
            "support_a": info["full_support_p"], "support_b": info["full_support_q"],
        })
    return rows


def plateau(rows, k_wall=None, n_min=2):
    """Winning block length n* = argmax over n of (d̄_n − floor_n), restricted to
    n_min ≤ n ≤ k_wall (the estimation wall; K2 guard). Returns the winning row."""
    ok = [r for r in rows if r["n"] >= n_min
          and (k_wall is None or r["n"] <= k_wall)]
    if not ok:
        return None
    return max(ok, key=lambda r: r["delta"])
