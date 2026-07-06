"""Testbed-A: 2D synthetic populations with a known MNAR selection mechanism.

A *population* p(x) is fixed and known (closed-form density where available).
A curated *corpus* D_obs is what survives a missing-not-at-random selector
    s_beta(x) = sigmoid(-beta * phi_std(x)),
where phi is a fixed feature that also carries label information (default:
the first coordinate x_1). At beta=0 the selector is a uniform 50% thinning
(MCAR: s == 0.5 everywhere, I(O;X)=0); as beta grows it censors the high-phi
tail ever harder (MNAR). This is the knob the whole paper sweeps.

D_ref is a *separate, uncensored, unlabeled* same-domain pool ~ p(x) (the
"full-support reference cover"): a controlled affordance that is NOT available
on real corpora (see limitations / E5 / K5). It is the reference class for the
density-ratio selection estimator in selection.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# ------------------------------------------------------------------ testbeds


def eight_gaussians(n, seed, radius=4.0, std=0.5):
    """8 isotropic Gaussians on a ring. Closed-form density available."""
    rng = np.random.default_rng(seed)
    ang = np.arange(8) * (2 * np.pi / 8)
    centers = radius * np.stack([np.cos(ang), np.sin(ang)], 1)
    comp = rng.integers(0, 8, n)
    X = centers[comp] + std * rng.standard_normal((n, 2))
    # binary label = arm parity (alternating around the ring); not used by E0
    y = (comp % 2).astype(int)
    return X.astype(np.float64), y


def eight_gaussians_logpdf(X, radius=4.0, std=0.5):
    ang = np.arange(8) * (2 * np.pi / 8)
    centers = radius * np.stack([np.cos(ang), np.sin(ang)], 1)
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)     # (n,8)
    logk = -0.5 * d2 / std**2 - np.log(2 * np.pi * std**2)
    return _logsumexp(logk, axis=1) - np.log(8.0)


def two_moons(n, seed, noise=0.12, scale=2.0):
    """Two interleaving half-circles. The connected-manifold task used for the
    full ladder: the censored tail of each moon is reachable by SHORT
    extrapolation ALONG the moon (the recoverable collar), not by jumping to a
    foreign mode."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0
    t0 = rng.uniform(0, np.pi, n0)
    x0 = np.stack([np.cos(t0), np.sin(t0)], 1)
    t1 = rng.uniform(0, np.pi, n1)
    x1 = np.stack([1.0 - np.cos(t1), 0.5 - np.sin(t1)], 1)
    X = np.concatenate([x0, x1], 0)
    y = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(int)
    X = X + noise * rng.standard_normal(X.shape)
    X = X * scale
    perm = rng.permutation(n)
    return X[perm].astype(np.float64), y[perm]


FOREIGN_CENTER = np.array([4.6, -1.1])   # off the moon manifold, high-phi tail
FOREIGN_STD = 0.20
FOREIGN_FRAC = 0.10


def two_moons_foreign(n, seed, noise=0.12, scale=2.0):
    """Two-moons PLUS a DISCONNECTED class-1 satellite blob in the censored
    (high-phi) region and OFF the moon manifold. The satellite is genuinely
    non-recombinable from the observed moon structure: the K4 positive control
    the method MUST fail to recover (it can only synthesize censored
    COMBINATIONS of present structure, not foreign structure)."""
    rng = np.random.default_rng(seed)
    n_f = int(FOREIGN_FRAC * n)
    Xm, ym = two_moons(n - n_f, seed=int(rng.integers(1 << 30)), noise=noise,
                       scale=scale)
    Xf = FOREIGN_CENTER + FOREIGN_STD * rng.standard_normal((n_f, 2))
    yf = np.ones(n_f, dtype=int)                      # foreign mode is class 1
    X = np.concatenate([Xm, Xf])
    y = np.concatenate([ym, yf])
    perm = rng.permutation(len(X))
    return X[perm].astype(np.float64), y[perm]


def foreign_mask(X):
    """Test points belonging to the foreign satellite (by proximity to its
    center), used to score K4 recovery in isolation."""
    return np.linalg.norm(X - FOREIGN_CENTER, axis=1) < 3 * FOREIGN_STD


def pinwheel(n, seed, n_arms=5, rate=0.25, std=0.3, radial=1.0):
    """Warped-spiral arms (Johnson et al.). E0 robustness testbed."""
    rng = np.random.default_rng(seed)
    arm = rng.integers(0, n_arms, n)
    r = radial * rng.standard_normal(n) * 0.3 + 1.5
    theta = arm * (2 * np.pi / n_arms) + rate * r + std * rng.standard_normal(n)
    X = np.stack([r * np.cos(theta), r * np.sin(theta)], 1)
    y = (arm % 2).astype(int)
    return X.astype(np.float64), y


TESTBEDS = {
    "eight_gaussians": eight_gaussians,
    "two_moons": two_moons,
    "two_moons_foreign": two_moons_foreign,
    "pinwheel": pinwheel,
}


# --------------------------------------------------------------- selector


@lru_cache(maxsize=None)
def _standardizer(testbed, coord=0):
    """phi mean/std from a large fixed population sample (deterministic, so the
    selector definition is pre-registered and reproducible)."""
    X, _ = TESTBEDS[testbed](40000, seed=0)
    phi = X[:, coord]
    return float(phi.mean()), float(phi.std())


def phi_std(X, testbed, coord=0):
    m, s = _standardizer(testbed, coord)
    return (X[:, coord] - m) / s


def selection_prob(X, beta, testbed, coord=0):
    """s_beta(x) = sigmoid(-beta * phi_std(x)). beta=0 -> 0.5 everywhere."""
    z = -beta * phi_std(X, testbed, coord)
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------- corpora

# The high-phi tail that gets progressively censored. A FIXED spatial region
# (beta-independent) so the same test points define "the censored slice" for
# every beta; at beta=0 they are not actually censored (s=0.5) so gain -> 0.
SLICE_PHI_THRESHOLD = 0.5   # phi_std > 0.5  (~top 30% by phi); pre-registered


@dataclass
class Corpora:
    testbed: str
    beta: float
    X_obs: np.ndarray          # curated corpus (labeled)
    y_obs: np.ndarray
    X_ref: np.ndarray          # uncensored reference pool (unlabeled)
    X_test: np.ndarray         # frozen full-population test set (labeled)
    y_test: np.ndarray
    s_obs_true: np.ndarray     # true s_beta on the surviving corpus
    obs_frac: float            # P(observed) = E_p[s_beta]  (known)

    def slice_mask(self, X):
        return phi_std(X, self.testbed) > SLICE_PHI_THRESHOLD


def make_corpora(testbed, beta, seed, n_pop=8000, n_ref=8000, n_test=8000,
                 coord=0):
    """Draw a population, apply the MNAR selector to get the curated corpus,
    and draw independent reference and test pools."""
    rng = np.random.default_rng(1000 + seed)
    X_pop, y_pop = TESTBEDS[testbed](n_pop, seed=rng.integers(1 << 30))
    s = selection_prob(X_pop, beta, testbed, coord)
    keep = rng.random(len(X_pop)) < s
    X_obs, y_obs = X_pop[keep], y_pop[keep]
    s_obs_true = s[keep]

    X_ref, _ = TESTBEDS[testbed](n_ref, seed=rng.integers(1 << 30))
    X_test, y_test = TESTBEDS[testbed](n_test, seed=rng.integers(1 << 30))

    # obs_frac from an independent large sample (the known marginal selection
    # rate; used to scale s_hat to a proper probability in selection.py).
    X_big, _ = TESTBEDS[testbed](40000, seed=7)
    obs_frac = float(selection_prob(X_big, beta, testbed, coord).mean())

    return Corpora(testbed, beta, X_obs, y_obs, X_ref, X_test, y_test,
                   s_obs_true, obs_frac)


def _logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m.squeeze(axis) + np.log(np.exp(a - m).sum(axis=axis)))
