"""Selective-sampling harnesses with controlled selection mechanisms (Paper A).

Generator (the controlled object every claim in Paper A rests on):

    x ~ p(x) on [-3, 3]^d
    z ~ N(0, 1)                       instrument (affects selection only)
    (eps, u) ~ BVN(0, [[sigma^2, rho*sigma], [rho*sigma, 1]])
    y  = f0(x) + eps                  outcome
    s* = g0(x) + alpha * z + u        latent selection index
    s  = 1[s* > 0]

    Observed: (x_i, z_i, s_i) for all i; y_i only where s_i = 1.

Regimes:
- selection on observables (MAR control): rho = 0 (selection depends on x, z
  and independent noise only) -- importance weighting is consistent here.
- selection on unobservables (target regime): rho != 0 -- the sampling of
  training points is correlated with unobserved determinants of the outcome;
  importance weighting with ORACLE propensities is structurally helpless.
- mixtures: rho swept in [0, 0.9].
- instrument present (alpha > 0) vs absent (alpha = 0): Heckman
  identification is robust only with the exclusion restriction.

The oracle propensity P(s=1 | x, z) = Phi(g0(x) + alpha z) is returned for
every unit so importance-weighting baselines get exactly the quantity their
theory asks for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr


@dataclass
class SelectionData:
    x: np.ndarray          # (n, d) covariates
    z: np.ndarray          # (n,)  instrument
    y: np.ndarray          # (n,)  outcomes (NaN where unobserved)
    y_full: np.ndarray     # (n,)  counterfactual complete outcomes (oracle)
    s: np.ndarray          # (n,)  selection indicator
    f0x: np.ndarray        # (n,)  true outcome function values (oracle)
    propensity: np.ndarray  # (n,) oracle P(s=1 | x, z)
    prop_x: np.ndarray     # (n,)  oracle P(s=1 | x) (z integrated out)
    rho: float
    sigma: float
    alpha: float
    meta: dict


def f0_smooth(x: np.ndarray) -> np.ndarray:
    """Default 1-d outcome function: smooth, nonmonotone."""
    x1 = x[:, 0]
    return np.sin(1.5 * x1) + 0.5 * x1


def f0_linear(x: np.ndarray) -> np.ndarray:
    return x @ np.linspace(1.0, 0.5, x.shape[1])


def g0_region(x: np.ndarray, strength: float = 1.8) -> np.ndarray:
    """Selection index depending on x: units with large x1 are under-sampled.

    g0 decreasing in x1 => P(s=1|x) falls from ~Phi(strength) at x1=-3 to
    ~Phi(-strength) at x1=+3: a smooth 'selected-against region' at high x1.
    """
    return -strength * (x[:, 0] / 3.0)


def make_selection_data(
    n: int,
    rho: float,
    alpha: float = 1.0,
    sigma: float = 0.5,
    d: int = 1,
    f0=f0_smooth,
    g0=g0_region,
    g0_offset: float = 0.0,
    seed: int = 0,
    x_dist: str = "uniform",
) -> SelectionData:
    """Draw one dataset from the controlled generator."""
    rng = np.random.default_rng(seed)
    if x_dist == "uniform":
        x = rng.uniform(-3.0, 3.0, size=(n, d))
    elif x_dist == "normal":
        x = np.clip(rng.normal(0.0, 1.5, size=(n, d)), -3.0, 3.0)
    else:
        raise ValueError(x_dist)
    z = rng.normal(0.0, 1.0, size=n)

    cov = np.array([[sigma**2, rho * sigma], [rho * sigma, 1.0]])
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(2))
    eu = rng.normal(size=(n, 2)) @ L.T
    eps, u = eu[:, 0], eu[:, 1]

    fx = f0(x)
    y_full = fx + eps
    idx = g0(x) + g0_offset + alpha * z
    s = (idx + u > 0).astype(float)

    y = np.where(s > 0.5, y_full, np.nan)
    propensity = ndtr(idx)                                  # P(s=1 | x, z)
    prop_x = ndtr((g0(x) + g0_offset) / np.sqrt(1.0 + alpha**2))  # z out

    return SelectionData(
        x=x, z=z, y=y, y_full=y_full, s=s, f0x=fx,
        propensity=propensity, prop_x=prop_x,
        rho=rho, sigma=sigma, alpha=alpha,
        meta={"n": n, "d": d, "seed": seed, "x_dist": x_dist,
              "g0_offset": g0_offset,
              "selected_frac": float(s.mean())},
    )


def make_linear_selection_data(n: int, rho: float, alpha: float = 1.0,
                               sigma: float = 1.0, d: int = 2,
                               seed: int = 0) -> SelectionData:
    """Linear f0/g0 variant used by the classic-estimator tests (A-E0)."""
    return make_selection_data(n, rho, alpha=alpha, sigma=sigma, d=d,
                               f0=f0_linear,
                               g0=lambda x: 0.5 * x[:, 0] - 0.3 * x[:, 1]
                               if x.shape[1] > 1 else 0.5 * x[:, 0],
                               seed=seed, x_dist="normal")


def induce_mnar_selection(x: np.ndarray, y: np.ndarray, rho: float,
                          alpha: float = 1.0, k_x: float = 1.0,
                          target_frac: float = 0.5, seed: int = 0):
    """Induce MNAR selection on a REAL tabular dataset (A-E2).

    Documented rule (fully reproducible given the seed):
        m_i   = standardized linear prediction of y from x (the observable
                part of outcome quality);
        r_i   = standardized residual of y ~ linear(x) (the UNOBSERVABLE:
                exactly the part of y that x cannot explain);
        z_i   ~ N(0,1) independent instrument (data-collection randomness);
        s*_i  = -k_x * m_i + alpha * z_i
                + rho * r_i + sqrt(1 - rho^2) * v_i - c,   v_i ~ N(0,1)
        s_i   = 1[s*_i > 0].

    -k_x * m_i makes high-predicted-outcome regions selected AGAINST
    (loan-style selective observation), so the propensity varies over
    x-space; rho controls selection on unobservables; c is set so that
    ~target_frac of units are selected.

    Returns (s, z, propensity, prop_x):
        propensity = P(s=1 | x, z) = Phi(-k_x m + alpha z - c) -- the exact
                     quantity oracle importance weighting asks for (the
                     combined noise rho r + sqrt(1-rho^2) v has unit
                     variance and is independent of (x, z) by construction);
        prop_x     = P(s=1 | x) with z integrated out (region definition).
    """
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    X1 = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    m = X1 @ coef
    m = (m - m.mean()) / (m.std() + 1e-12)
    r = y - X1 @ coef
    r = (r - r.mean()) / (r.std() + 1e-12)
    z = rng.normal(size=n)
    v = rng.normal(size=n)
    noise = rho * r + np.sqrt(max(1 - rho**2, 0.0)) * v
    idx = -k_x * m + alpha * z + noise
    c = np.quantile(idx, 1.0 - target_frac)
    s = (idx - c > 0).astype(float)
    propensity = ndtr(-k_x * m + alpha * z - c)
    prop_x = ndtr((-k_x * m - c) / np.sqrt(1.0 + alpha**2))
    return s, z, propensity, prop_x
