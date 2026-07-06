"""Seed statistics, paired tests, and the inverted-U adjudicator for GATE F.

Conventions (PLAN.md): >=5 seeds for GATE F, 8 for the headline retention curve;
report mean +- sd across seeds; paired Wilcoxon signed-rank for method vs method
at matched seed; the inverted-U test is the pre-registered GATE-F criterion.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def mean_sd(x, axis=0):
    x = np.asarray(x, float)
    return float(np.mean(x, axis=axis)) if x.ndim == 1 else np.mean(x, axis=axis), \
        (float(np.std(x, axis=axis, ddof=1)) if x.ndim == 1 else np.std(x, axis=axis, ddof=1))


def sem(x, axis=0):
    x = np.asarray(x, float)
    n = x.shape[axis]
    return np.std(x, axis=axis, ddof=1) / np.sqrt(max(n, 1))


def paired_wilcoxon(a, b):
    """Two-sided Wilcoxon signed-rank on paired (per-seed) samples a vs b.
    Returns (statistic, p, median_diff). Falls back gracefully for tiny n or all-
    zero differences (p=1)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0) or d.size < 3:
        return float("nan"), 1.0, float(np.median(d)) if d.size else float("nan")
    try:
        w = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(w.statistic), float(w.pvalue), float(np.median(d))
    except ValueError:
        return float("nan"), 1.0, float(np.median(d))


def wilcoxon_greater(a, b):
    """One-sided: is a > b (paired)? Returns p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0) or d.size < 3:
        return 1.0
    try:
        return float(sps.wilcoxon(a, b, alternative="greater").pvalue)
    except ValueError:
        return 1.0


def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(x)), float(lo), float(hi)


def inverted_u_test(sigmas, ret_by_seed, alpha=0.05):
    """Pre-registered GATE-F criterion.

    Input: sigmas (S,) increasing, ret_by_seed (n_seed, S) retention (mean past-
    task accuracy) per seed per noise level.

    An inverted-U (noise HELPS retention non-monotonically) is declared iff:
      1. the seed-mean curve peaks at an INTERIOR sigma* > 0 (not at either end);
      2. peak beats the zero-noise end: retention(sigma*) > retention(0),
         one-sided paired Wilcoxon p < alpha;
      3. peak beats the high-noise end: retention(sigma*) > retention(sigma_max),
         one-sided paired Wilcoxon p < alpha (the down-slope is real, not flat);
      4. the lift over sigma=0 exceeds the pooled seed sd at the peak.

    Returns a dict with the verdict and all supporting quantities.
    """
    sigmas = np.asarray(sigmas, float)
    R = np.asarray(ret_by_seed, float)          # (n_seed, S)
    mean = R.mean(axis=0)
    sd = R.std(axis=0, ddof=1)
    i_peak = int(np.argmax(mean))
    interior = 0 < i_peak < len(sigmas) - 1

    p_vs_zero = wilcoxon_greater(R[:, i_peak], R[:, 0])
    p_vs_hi = wilcoxon_greater(R[:, i_peak], R[:, -1])
    lift0 = float(mean[i_peak] - mean[0])
    lift_hi = float(mean[i_peak] - mean[-1])
    lift_exceeds_sd = lift0 > sd[i_peak]

    verdict = bool(interior and p_vs_zero < alpha and p_vs_hi < alpha
                   and lift_exceeds_sd)
    return {
        "sigmas": sigmas.tolist(),
        "mean": mean.tolist(),
        "sd": sd.tolist(),
        "sem": sem(R).tolist(),
        "i_peak": i_peak,
        "sigma_star": float(sigmas[i_peak]),
        "ret_at_zero": float(mean[0]),
        "ret_at_peak": float(mean[i_peak]),
        "ret_at_hi": float(mean[-1]),
        "lift_over_zero": lift0,
        "lift_over_hi": lift_hi,
        "interior_peak": bool(interior),
        "p_peak_gt_zero": float(p_vs_zero),
        "p_peak_gt_hi": float(p_vs_hi),
        "lift_exceeds_sd": bool(lift_exceeds_sd),
        "inverted_u": verdict,
    }


def monotone_decreasing_frac(sigmas, ret_by_seed):
    """Diagnostic for the CONTROLS: fraction of adjacent steps where seed-mean
    retention decreases. ~1.0 = monotone-decreasing (expected for plain OU/EWC/
    MESU: more noise only widens the stationary spread)."""
    mean = np.asarray(ret_by_seed, float).mean(axis=0)
    d = np.diff(mean)
    return float(np.mean(d < 0)) if d.size else float("nan")


def curvature_sign(sigmas, ret_by_seed):
    """Sign and magnitude of the quadratic term of retention vs sigma (concave
    < 0 supports an interior optimum). Fit on the seed-mean curve."""
    x = np.asarray(sigmas, float)
    y = np.asarray(ret_by_seed, float).mean(axis=0)
    if x.size < 3:
        return float("nan")
    c2 = np.polyfit(x, y, 2)[0]
    return float(c2)
