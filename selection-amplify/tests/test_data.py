import numpy as np

from selamp import data
from selamp.entropy import mutual_info_OX


def test_eight_gaussian_density_normalizes():
    # integrate closed-form pdf on a grid; should be ~1
    xs = np.linspace(-7, 7, 400)
    gx, gy = np.meshgrid(xs, xs)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    p = np.exp(data.eight_gaussians_logpdf(pts))
    dA = (xs[1] - xs[0]) ** 2
    assert abs(p.sum() * dA - 1.0) < 0.02


def test_selector_beta0_is_mcar():
    X, _ = data.eight_gaussians(2000, seed=1)
    s = data.selection_prob(X, beta=0.0, testbed="eight_gaussians")
    assert np.allclose(s, 0.5, atol=1e-9)


def test_selector_censors_high_phi():
    X, _ = data.two_moons(4000, seed=1)
    s = data.selection_prob(X, beta=4.0, testbed="two_moons")
    phi = data.phi_std(X, "two_moons")
    # high-phi points are censored (low s); low-phi retained
    assert s[phi > 1].mean() < 0.2
    assert s[phi < -1].mean() > 0.8


def test_IOX_zero_at_beta0_and_rises():
    X, _ = data.two_moons(20000, seed=2)
    betas = [0.0, 0.5, 1, 2, 4, 8]
    iox = [mutual_info_OX(data.selection_prob(X, b, "two_moons")) for b in betas]
    assert iox[0] < 1e-9
    assert all(np.diff(iox) > 0)          # strictly increasing in beta


def test_obs_frac_symmetric():
    # symmetric phi => E[s]~0.5 for all beta (corpus size ~ constant, shape shifts)
    c = data.make_corpora("two_moons", beta=4.0, seed=0, n_pop=8000)
    assert abs(c.obs_frac - 0.5) < 0.03
