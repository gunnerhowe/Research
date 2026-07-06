"""Paired nonparametric statistics with effect sizes and bootstrap CIs.
House rule: >=3 seeds, paired Wilcoxon signed-rank at matched seed, effect
sizes + CIs, never a bare p-value."""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_wilcoxon(a, b, alternative="greater"):
    """One-sided paired Wilcoxon signed-rank on matched-seed pairs a,b.
    Returns (p, effect) with effect the matched-pairs rank-biserial
    correlation. NaN-safe: returns (nan, 0) if <3 usable non-tied pairs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    d = a - b
    nz = d[d != 0]
    if len(nz) < 3:
        return float("nan"), 0.0
    p = float(stats.wilcoxon(a, b, alternative=alternative,
                             zero_method="wilcox").pvalue)
    return p, rank_biserial(a, b)


def rank_biserial(a, b):
    """Matched-pairs rank-biserial: (sum R+ - sum R-)/sum R, in [-1,1]."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    rp = r[d > 0].sum()
    rm = r[d < 0].sum()
    return float((rp - rm) / r.sum())


def bootstrap_ci(x, n_boot=10000, alpha=0.05, stat=np.mean, seed=0):
    """Percentile bootstrap CI for a 1-sample statistic (e.g. mean paired
    difference)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)]
    lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), float("nan")
    r = stats.spearmanr(x[m], y[m])
    return float(r.statistic), float(r.pvalue)


def spearman_onesided_pos(x, y):
    """One-sided Spearman test for a POSITIVE monotone trend."""
    rho, p2 = spearman(x, y)
    if not np.isfinite(rho):
        return rho, float("nan")
    p1 = p2 / 2 if rho > 0 else 1 - p2 / 2
    return rho, float(p1)


def mean_sd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()), float(x.std())
