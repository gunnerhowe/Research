r"""Rice-formula predictions of crossing and delta-event rates from trace stats.

Prediction side (no gradients needed): numpy/scipy. Three predictors used in E1:

(a) RICE (Gaussian, temporal structure): from fitted spectral moments
        lambda0 = Var a,  lambda2 = Var da/dt (finite differences),
        mu(u) = (1/pi) sqrt(lambda2/lambda0) exp(-(u-m)^2 / (2 lambda0)).
    Appendix refinement: the exact discrete-time stationary-Gaussian crossing
    probability per step, via Owen's T:
        P(cross u) = 4 T(h, sqrt((1-rho1)/(1+rho1))),  h=(u-m)/sqrt(lambda0),
    which fixes the continuous formula's underprediction when lag-1
    correlation is weak (rho1 -> 0 gives the iid value 2 Phi(h)(1-Phi(h));
    u=0 recovers the classical arccos(rho1)/pi sign-change rate).

(b) EMPIRICAL: measured crossing profile c(u) on a calibration split,
    interpolated to arbitrary levels (no Gaussianity; this profile is also
    what the differentiable training objective controls).

(c) IID baseline (no temporal structure): rho1 = 0 case of (a-discrete).

Delta-EVENT rate (send-on-delta with anchor memory, Neil et al. semantics):
the ladder/total-variation bound R(theta) <= TV_rate / theta, with Gaussian
TV_rate = E|da/dt| = sqrt(2 lambda2 / pi). Discrete sampling caps events at
1/dt per unit; hysteresis and multi-level jumps push true events below the
bound. E0 measures the correction factor gamma(theta / sigma_delta) once on
synthetic processes; predictions multiply the bound by the interpolated gamma.
"""

import math

import numpy as np
from scipy.special import owens_t
from scipy.stats import norm


# --------------------------------------------------------------------------
# Trace statistics
# --------------------------------------------------------------------------


def spectral_moments(traces, dt=1.0, axis=-1):
    """Fitted moments of sampled traces along `axis` (time).

    Returns dict of arrays (other axes preserved): mean, lam0 (variance),
    lam2 (variance of the FD derivative), rho1 (lag-1 autocorrelation of the
    mean-removed trace), tv_rate (mean |da|/dt), sigma_delta (std of one-step
    increments).
    """
    a = np.asarray(traces, dtype=np.float64)
    a = np.moveaxis(a, axis, -1)
    m = a.mean(axis=-1)
    lam0 = a.var(axis=-1)
    d = np.diff(a, axis=-1)
    lam2 = (d / dt).var(axis=-1) + (d / dt).mean(axis=-1) ** 2  # E[(a')^2] about 0-mean drift
    ac = a - m[..., None]
    denom = (ac[..., :-1] ** 2).mean(axis=-1) * (ac[..., 1:] ** 2).mean(axis=-1)
    rho1 = (ac[..., :-1] * ac[..., 1:]).mean(axis=-1) / np.sqrt(np.maximum(denom, 1e-30))
    tv_rate = np.abs(d).mean(axis=-1) / dt
    sigma_delta = d.std(axis=-1)
    return dict(mean=m, lam0=lam0, lam2=lam2, rho1=np.clip(rho1, -1.0, 1.0),
                tv_rate=tv_rate, sigma_delta=sigma_delta)


# --------------------------------------------------------------------------
# Crossing-rate predictors
# --------------------------------------------------------------------------


def rice_rate(levels, mean, lam0, lam2):
    """Continuous Rice formula, up+down crossings per unit time.

    levels: (L,); mean/lam0/lam2 scalars or arrays broadcastable against (L,).
    """
    u = np.asarray(levels, dtype=np.float64)
    lam0 = np.maximum(np.asarray(lam0, dtype=np.float64), 1e-30)
    lam2 = np.maximum(np.asarray(lam2, dtype=np.float64), 0.0)
    return (1.0 / math.pi) * np.sqrt(lam2 / lam0) * np.exp(-((u - mean) ** 2) / (2.0 * lam0))


def gaussian_discrete_rate(levels, mean, lam0, rho1, dt=1.0):
    """Exact crossing probability per step for a stationary Gaussian sequence.

    P((a_t-u)(a_{t+1}-u) < 0) = 4 * OwensT(h, sqrt((1-rho1)/(1+rho1))).
    Returned as a rate per unit time (P / dt).
    """
    lam0 = np.maximum(np.asarray(lam0, dtype=np.float64), 1e-30)
    h = (np.asarray(levels, dtype=np.float64) - mean) / np.sqrt(lam0)
    rho = np.clip(np.asarray(rho1, dtype=np.float64), -0.999999, 0.999999)
    aa = np.sqrt((1.0 - rho) / (1.0 + rho))
    return 4.0 * owens_t(h, np.broadcast_to(aa, h.shape)) / dt


def iid_rate(levels, mean, lam0, dt=1.0):
    """Gaussian-iid baseline (no temporal structure): 2 Phi(h)(1-Phi(h)) / dt."""
    lam0 = np.maximum(np.asarray(lam0, dtype=np.float64), 1e-30)
    h = (np.asarray(levels, dtype=np.float64) - mean) / np.sqrt(lam0)
    p = norm.cdf(h)
    return 2.0 * p * (1.0 - p) / dt


class EmpiricalProfile:
    """Predictor (b): measured crossing profile on a calibration split.

    Built from hard crossing counts on a dense level grid; predicts the rate
    at arbitrary levels by linear interpolation (0 outside the grid).
    """

    def __init__(self, grid_levels, grid_rates):
        self.levels = np.asarray(grid_levels, dtype=np.float64)
        self.rates = np.asarray(grid_rates, dtype=np.float64)

    def __call__(self, levels):
        return np.interp(np.asarray(levels, dtype=np.float64),
                         self.levels, self.rates, left=0.0, right=0.0)

    def integral(self):
        """int c(u) du = calibration TV rate (trapezoid)."""
        return float(np.trapezoid(self.rates, self.levels))


# --------------------------------------------------------------------------
# Delta-event rate predictions (send-on-delta with anchor memory)
# --------------------------------------------------------------------------


def sod_rate_bound(theta, tv_rate, dt=1.0):
    """Ladder / total-variation upper bound on the send-on-delta event rate,
    capped at the sampling rate: min(tv_rate/theta, 1/dt)."""
    theta = np.asarray(theta, dtype=np.float64)
    return np.minimum(np.asarray(tv_rate, dtype=np.float64) / np.maximum(theta, 1e-30),
                      1.0 / dt)


def gaussian_tv_rate(lam2):
    """Gaussian mean |da/dt| = sqrt(2 lambda2 / pi)."""
    return np.sqrt(2.0 * np.asarray(lam2, dtype=np.float64) / math.pi)


class SodCorrection:
    """Empirical correction gamma = events_measured / bound as a function of
    x = theta / sigma_delta (std of one-step increments), measured once in E0
    on synthetic processes and reused for prediction on real traces.
    gamma -> 1 as x -> 0 is NOT expected (discrete-time hysteresis bites at
    all scales); we simply interpolate the measured curve in log-x.
    """

    def __init__(self, x_grid, gamma):
        self.logx = np.log(np.asarray(x_grid, dtype=np.float64))
        self.gamma = np.asarray(gamma, dtype=np.float64)

    def __call__(self, x):
        lx = np.log(np.maximum(np.asarray(x, dtype=np.float64), 1e-12))
        return np.interp(lx, self.logx, self.gamma,
                         left=self.gamma[0], right=self.gamma[-1])

    def to_json(self):
        return dict(x_grid=np.exp(self.logx).tolist(), gamma=self.gamma.tolist())

    @classmethod
    def from_json(cls, d):
        return cls(np.asarray(d["x_grid"]), np.asarray(d["gamma"]))


def sod_rate_predicted(theta, tv_rate, sigma_delta, correction=None, dt=1.0):
    """Bound x empirical correction (if given)."""
    r = sod_rate_bound(theta, tv_rate, dt)
    if correction is not None:
        r = r * correction(np.asarray(theta) / np.maximum(sigma_delta, 1e-30))
    return np.minimum(r, 1.0 / dt)
