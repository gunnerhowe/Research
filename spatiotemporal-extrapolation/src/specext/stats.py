"""Statistics conventions, spectrum-implied observables, and comparison metrics.

Density convention (PLAN.md): S(k_m) = P_m / (2 pi / L) with P_m the one-sided mode
power of the mean-zero field; S converges to the thermodynamic-limit spectral
density as L -> infinity and is the object the finite-size flow acts on.
"""
from __future__ import annotations

import numpy as np


def density_from_power(p_m, L):
    return p_m / (2 * np.pi / L)


def power_from_density(s_k, L):
    return s_k * (2 * np.pi / L)


def corr_from_power(p_full, L, r):
    """Equal-time two-point correlation C(r) from full one-sided power (m = 0..N//2):
    C(r) = sum_m P_m cos(k_m r). p_full 1-D, r any grid."""
    k = 2 * np.pi * np.arange(p_full.shape[0]) / L
    return np.cos(np.outer(np.asarray(r, float), k)) @ p_full


def corr_from_modes_power(p_m, k, r):
    """C(r) restricted to a retained mode set (one-sided powers p_m at wavenumbers k)."""
    return (p_m[None, :] * np.cos(np.outer(r, k))).sum(axis=1)


def spacetime_corr_leading(s_k, gamma, omega, k, L, r, t):
    """Leading-resonance space-time correlation:
    C(r, t) = sum_k P_k cos(k r) exp(-gamma |t|) cos(omega t)."""
    p = power_from_density(s_k, L)
    decay = np.exp(-np.abs(t)[:, None] * gamma[None, :]) * np.cos(np.abs(t)[:, None] * omega[None, :])
    cosr = np.cos(np.outer(r, k))
    return np.einsum("tk,rk,k->rt", decay, cosr, p)


def tau_e_leading(gamma, omega):
    """First |rho| <= 1/e time of rho(t) = exp(-gamma t) cos(omega t) (analytic grid)."""
    t = np.linspace(0, 50.0 / np.maximum(gamma, 1e-6), 4000, axis=-1)
    rho = np.exp(-gamma[..., None] * t) * np.abs(np.cos(omega[..., None] * t))
    below = rho <= np.exp(-1.0)
    below[..., 0] = False
    idx = below.argmax(axis=-1)
    return np.take_along_axis(t, idx[..., None], axis=-1)[..., 0]


# ---------------------------------------------------------------- metrics

def median_rel_err(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    ok = np.isfinite(pred) & np.isfinite(true) & (np.abs(true) > 0)
    if not ok.any():
        return np.nan
    return float(np.median(np.abs(pred[ok] - true[ok]) / np.abs(true[ok])))


def median_abs_log10_ratio(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    ok = np.isfinite(pred) & np.isfinite(true) & (pred > 0) & (true > 0)
    if not ok.any():
        return np.nan
    return float(np.median(np.abs(np.log10(pred[ok] / true[ok]))))


def rel_l2(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    ok = np.isfinite(pred) & np.isfinite(true)
    if not ok.any():
        return np.nan
    return float(np.linalg.norm(pred[ok] - true[ok]) / max(np.linalg.norm(true[ok]), 1e-300))


def slow_set_overlap(pred_gamma, true_gamma, k, n_top=16):
    """Fraction of the true n_top slowest sectors recovered by the predicted ranking.
    (Principal angles between the corresponding Fourier spans are 0/90 degrees by
    translation invariance, so the overlap fraction carries the same information;
    the paper states this.)"""
    ok = np.isfinite(pred_gamma) & np.isfinite(true_gamma)
    idx = np.nonzero(ok)[0]
    ps = idx[np.argsort(pred_gamma[idx])][:n_top]
    ts = idx[np.argsort(true_gamma[idx])][:n_top]
    return float(len(set(ps.tolist()) & set(ts.tolist())) / n_top)


def band_mask(k, lo, hi):
    k = np.asarray(k)
    return (k >= lo - 1e-12) & (k <= hi + 1e-12)
