"""Metrics for selection-corrected UQ (Paper A, A-E3).

All metrics take a Gaussian predictive (mu, var) per test point and true
outcomes y drawn from the POPULATION (a fresh uniform test set, no
selection). The headline metric is prediction-interval coverage split by
sampled-density region, where regions are defined by the oracle propensity
P(s=1 | x) of the generator:

    selected-against:  P(s=1 | x) <= 0.3
    mid:               0.3 < P(s=1 | x) < 0.7
    well-sampled:      P(s=1 | x) >= 0.7
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

REGIONS = {"against": (0.0, 0.3), "mid": (0.3, 0.7), "well": (0.7, 1.01)}
Z90 = stats.norm.ppf(0.95)  # central 90% interval half-width in sds


def region_masks(prop_x: np.ndarray) -> dict[str, np.ndarray]:
    return {name: (prop_x >= lo) & (prop_x < hi)
            for name, (lo, hi) in REGIONS.items()}


def picp(y, mu, var, level: float = 0.90) -> float:
    """Prediction-interval coverage probability (central interval)."""
    z = stats.norm.ppf(0.5 + level / 2.0)
    sd = np.sqrt(np.maximum(var, 1e-12))
    return float(np.mean(np.abs(y - mu) <= z * sd))


def gaussian_nll(y, mu, var) -> float:
    var = np.maximum(var, 1e-12)
    return float(np.mean(0.5 * ((y - mu)**2 / var + np.log(var)
                                + math.log(2 * math.pi))))


def quantile_ece(y, mu, var, n_levels: int = 19) -> float:
    """Regression calibration error (Kuleshov et al. 2018): mean |empirical
    CDF coverage - nominal| over evenly spaced quantile levels."""
    sd = np.sqrt(np.maximum(var, 1e-12))
    taus = np.linspace(0.05, 0.95, n_levels)
    zs = stats.norm.ppf(taus)
    emp = np.array([np.mean(y <= mu + z * sd) for z in zs])
    return float(np.mean(np.abs(emp - taus)))


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b))**2)))


def evaluate_predictive(y, mu, var, prop_x, f0x=None) -> dict:
    """Full metric block, overall and per region."""
    out: dict = {
        "picp90": picp(y, mu, var),
        "nll": gaussian_nll(y, mu, var),
        "ece": quantile_ece(y, mu, var),
    }
    if f0x is not None:
        out["rmse_f0"] = rmse(mu, f0x)
        out["bias_f0"] = float(np.mean(mu - f0x))
    for name, m in region_masks(prop_x).items():
        if m.sum() < 10:
            continue
        out[f"picp90_{name}"] = picp(y[m], mu[m], var[m])
        out[f"nll_{name}"] = gaussian_nll(y[m], mu[m], var[m])
        out[f"ece_{name}"] = quantile_ece(y[m], mu[m], var[m])
        if f0x is not None:
            out[f"rmse_f0_{name}"] = rmse(mu[m], f0x[m])
            out[f"bias_f0_{name}"] = float(np.mean(mu[m] - f0x[m]))
        out[f"n_{name}"] = int(m.sum())
    return out
