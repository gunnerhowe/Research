import numpy as np
import torch

from selamp import data
from selamp.selection import SelectionEstimator
from selamp.stats import spearman


def _fit_small(beta=2.0, testbed="eight_gaussians", seed=0):
    c = data.make_corpora(testbed, beta=beta, seed=seed, n_pop=6000,
                          n_ref=6000, n_test=4000)
    est = SelectionEstimator(n_members=3, epochs=200).fit(
        c.X_obs, c.X_ref, c.obs_frac, seed=seed)
    return c, est


def test_shat_recovers_known_selector():
    c, est = _fit_small()
    s_true = data.selection_prob(c.X_test, c.beta, c.testbed)
    s_hat = est.s_hat(c.X_test)
    rho, _ = spearman(s_hat, s_true)
    assert rho > 0.7                       # E0-style recovery (unit scale)


def test_shat_is_differentiable():
    c, est = _fit_small()
    X = torch.tensor(c.X_test[:64], dtype=torch.float32,
                     device=est.device, requires_grad=True)
    s = est.s_hat_torch(X)
    g = torch.autograd.grad(s.sum(), X)[0]
    assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_uncertainty_higher_in_deep_complement():
    c, est = _fit_small(beta=4.0)
    phi = data.phi_std(c.X_test, c.testbed)
    u = est.uncertainty(c.X_test)
    # log-odds epistemic uncertainty GROWS into the deep, unobserved complement
    # (off-support extrapolation) vs the well-observed core -- the signal the
    # collar gate relies on to refuse steering where s_hat is untrustworthy.
    deep = u[phi > 1.5].mean()
    obs = u[phi < 0].mean()
    assert deep > 2 * obs


def test_calibration_ece_small():
    _, est = _fit_small()
    assert est.ece() < 0.15
