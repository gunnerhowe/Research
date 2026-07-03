"""Baseline metrics the spec requires d̄ to be compared against:
Wasserstein on the invariant measure (marginal and state-space), power-spectrum
distance, autocorrelation distance.
"""
from __future__ import annotations

import numpy as np
import ot
from scipy import signal, stats
from scipy.spatial import cKDTree


def w1_marginal(a, b):
    """1D Wasserstein-1 between value distributions of two scalar series."""
    return float(stats.wasserstein_distance(np.asarray(a), np.asarray(b)))


def w1_state(X, Y, n_sub=2000, repeats=4, seed=0, normalize_by=None):
    """W1 between empirical state-space measures (point clouds in R^d), with a
    same-vs-same floor computed on disjoint halves of X under the same budget.

    Returns (value, floor). Distances normalized by `normalize_by` (default: mean
    pairwise scale of X) so values are dimensionless and comparable.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    scale = normalize_by or float(np.std(X, axis=0).mean())

    def one(A, B, r):
        ia = rng.choice(len(A), min(n_sub, len(A)), replace=False)
        ib = rng.choice(len(B), min(n_sub, len(B)), replace=False)
        M = ot.dist(A[ia], B[ib], metric="euclidean") / scale
        wa = np.full(len(ia), 1.0 / len(ia))
        wb = np.full(len(ib), 1.0 / len(ib))
        return float(ot.emd2(wa, wb, M, numItermax=2_000_000))

    vals = [one(X, Y, r) for r in range(repeats)]
    X1, X2 = np.array_split(X, 2)
    floors = [one(X1, X2, r) for r in range(repeats)]
    return float(np.mean(vals)), float(np.mean(floors))


def delay_embed(x, dim=3, lag=1):
    x = np.asarray(x)
    n = len(x) - (dim - 1) * lag
    return np.stack([x[i * lag: i * lag + n] for i in range(dim)], axis=1)


def psd_log_distance(a, b, fs=1.0, nperseg=4096):
    """RMSE between log10 Welch PSDs (dB-like). 0 = identical spectra."""
    fa, Pa = signal.welch(np.asarray(a), fs=fs, nperseg=nperseg)
    fb, Pb = signal.welch(np.asarray(b), fs=fs, nperseg=nperseg)
    eps = 1e-20
    return float(np.sqrt(np.mean((10 * np.log10(Pa + eps) - 10 * np.log10(Pb + eps)) ** 2)))


def acf(x, max_lag):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    r = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
    return r / r[0]


def acf_distance(a, b, max_lag=200):
    """RMSE between normalized autocorrelation functions over lags 1..max_lag."""
    ra = acf(a, max_lag)[1:]
    rb = acf(b, max_lag)[1:]
    return float(np.sqrt(np.mean((ra - rb) ** 2)))


def rosenstein_lambda1(x, dt=1.0, dim=3, lag=2, n_max=100_000, n_ref=4000,
                       theiler=50, k_max=40, fit_range=(1, 15), seed=0):
    """Largest Lyapunov exponent from a scalar series (Rosenstein et al. 1993).

    Returns (lambda1 per unit time, r2 of the linear fit). For non-deterministic
    series (e.g. IAAFT surrogates) there is no exponential-divergence regime; the
    fit r2 exposes that — report both rather than the slope alone.
    """
    x = np.asarray(x, dtype=np.float64)[:n_max]
    emb = delay_embed(x, dim=dim, lag=lag)
    n = len(emb) - k_max
    tree = cKDTree(emb[:n])
    rng = np.random.default_rng(seed)
    refs = rng.choice(n, size=min(n_ref, n), replace=False)
    # nearest neighbor outside the Theiler window
    dists, idxs = tree.query(emb[refs], k=2 * theiler + 5)
    log_div = np.full((len(refs), k_max + 1), np.nan)
    for r, i in enumerate(refs):
        mask = np.abs(idxs[r] - i) > theiler
        if not mask.any():
            continue
        j = idxs[r][mask][0]
        if j >= n:
            continue
        d = np.linalg.norm(emb[i:i + k_max + 1] - emb[j:j + k_max + 1], axis=1)
        with np.errstate(divide="ignore"):
            log_div[r] = np.log(d)
    mean_log = np.nanmean(log_div, axis=0)
    k = np.arange(fit_range[0], fit_range[1] + 1)
    y = mean_log[k]
    A = np.vstack([k, np.ones_like(k)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - (res[0] / ss_tot if len(res) and ss_tot > 0 else np.nan)
    return float(coef[0] / dt), float(r2)
