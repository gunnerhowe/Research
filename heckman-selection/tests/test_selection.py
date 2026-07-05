"""A-E0 faithfulness gate (pre-registered): our Heckman implementations must
reproduce classic econometric reference results before any experiment runs.

External references:
- Cameron & Trivedi (2005) sec. 16.6 RandHIE selection model. Their Stata
  output (seven digits, data/mma16p3selection.txt, downloaded from
  cameron.econ.ucdavis.edu/mmabook/mma16p3selection.txt) provides reference
  values for BOTH the two-step and the ML estimator. The JSS paper for R's
  sampleSelection (Toomet & Henningsen 2008, sec. 5.2) replicates the same
  numbers, so this is a three-way cross-validated reference.
- statsmodels Probit: the selection equation must agree to machine precision.
- A statsmodels-composed classic two-step on Mroz87 (Greene 2002 ex. 22.8
  spec): our integrated two-step must match the composition exactly.
- Large-n synthetic recovery with known truth (rho, sigma, beta).
"""

import numpy as np
import pytest
import statsmodels.api as sm

from heckesel.datasets import load_mroz_greene, load_randhie_ct
from heckesel.selection import (heckman_loglik, heckman_mle, heckman_two_step,
                                inverse_mills, probit_fit)
from heckesel.synth import make_linear_selection_data

# ------------------------------------------------------- Stata references
# from data/mma16p3selection.txt (heckman LNMED $XLIST, select(DMED=$XLIST))

STATA_MLE_OUT = np.array([
    -.0760236, -.1497199, .01493, -.023522, .3548628, .0286474, .1559173,
    .4451223, .9986065, .1214009, -.1583018, .0175951, .0057376, .5503441,
    -.1976875, -.5653227, -.5358684, 2.107745])
STATA_MLE_SEL = np.array([
    -.1068027, -.108769, .0294804, .0007403, .2848256, .0210805, .0576901,
    .2237238, .7984291, .0553122, -.031201, .031499, -.0006072, .4093059,
    .0530643, -.3953421, -.5831049, -.2141574])
STATA_MLE_RHO = .7355982
STATA_MLE_SIGMA = 1.570053
STATA_MLE_LL = -10170.11

STATA_2S_OUT = np.array([
    -.0279209, -.0922898, .0052225, -.0295212, .2814948, .021617, .1474026,
    .3821683, .833294, .0990973, -.1441358, .0033639, .0055556, .3846323,
    -.2565136, -.392146, -.2633649, 2.882514])
STATA_2S_SEL = np.array([
    -.118708, -.1279483, .0283091, .0075319, .2732013, .0224861, .0387516,
    .1920062, .6397294, .0518413, -.0335599, .036307, .0002631, .4451035,
    .111489, -.4512845, -.6057367, -.271605])
STATA_2S_LAMBDA = .2358048
STATA_2S_RHO = 0.16833
STATA_2S_SIGMA = 1.4008246


@pytest.fixture(scope="module")
def randhie():
    return load_randhie_ct()


@pytest.fixture(scope="module")
def mroz():
    return load_mroz_greene()


# ---------------------------------------------------------------- probit


def test_randhie_dimensions(randhie):
    y, X, s, W, *_ = randhie
    assert len(s) == 5574
    assert int(s.sum()) == 4281


def test_probit_matches_statsmodels_randhie(randhie):
    y, X, s, W, *_ = randhie
    ours = probit_fit(W, s)
    ref = sm.Probit(s, W).fit(disp=0, method="newton", tol=1e-10)
    assert ours.converged
    np.testing.assert_allclose(ours.params, ref.params, atol=1e-8)
    assert abs(ours.loglik - ref.llf) < 1e-6
    np.testing.assert_allclose(np.sqrt(np.diag(ours.cov)), ref.bse, rtol=1e-5)


def test_probit_matches_statsmodels_mroz(mroz):
    y, X, s, W, *_ = mroz
    Wn = W / np.max(np.abs(W), axis=0)  # scale (age^2, faminc are huge)
    ours = probit_fit(Wn, s)
    ref = sm.Probit(s, Wn).fit(disp=0, method="newton", tol=1e-10)
    np.testing.assert_allclose(ours.params, ref.params, atol=1e-8)


def test_probit_matches_stata_twostep_selection(randhie):
    y, X, s, W, *_ = randhie
    ours = probit_fit(W, s)
    np.testing.assert_allclose(ours.params, STATA_2S_SEL, atol=2e-6)


# --------------------------------------------------------------- two-step


def test_two_step_matches_stata_randhie(randhie):
    y, X, s, W, *_ = randhie
    r = heckman_two_step(y, X, s, W)
    np.testing.assert_allclose(r.beta, STATA_2S_OUT, atol=5e-6)
    np.testing.assert_allclose(r.beta_lambda, STATA_2S_LAMBDA, atol=5e-6)
    np.testing.assert_allclose(r.sigma, STATA_2S_SIGMA, atol=5e-6)
    np.testing.assert_allclose(r.rho, STATA_2S_RHO, atol=5e-5)


def test_two_step_matches_composed_statsmodels_mroz(mroz):
    """Our integrated two-step == statsmodels Probit + OLS-with-IMR, exactly."""
    y, X, s, W, *_ = mroz
    sel = s > 0.5
    pr = sm.Probit(s, W).fit(disp=0, method="newton", tol=1e-10)
    eta = W[sel] @ pr.params
    lam = inverse_mills(eta)
    Z = np.column_stack([X[sel], lam])
    ols = sm.OLS(y[sel], Z).fit()

    r = heckman_two_step(y, X, s, W)
    np.testing.assert_allclose(r.beta, ols.params[:-1], atol=1e-8)
    np.testing.assert_allclose(r.beta_lambda, ols.params[-1], atol=1e-8)


# --------------------------------------------------------------------- MLE


def test_mle_matches_stata_randhie(randhie):
    y, X, s, W, *_ = randhie
    r = heckman_mle(y, X, s, W)
    assert r.converged
    np.testing.assert_allclose(r.loglik, STATA_MLE_LL, atol=0.02)
    np.testing.assert_allclose(r.beta, STATA_MLE_OUT, atol=2e-4)
    np.testing.assert_allclose(r.gamma, STATA_MLE_SEL, atol=2e-4)
    np.testing.assert_allclose(r.rho, STATA_MLE_RHO, atol=1e-4)
    np.testing.assert_allclose(r.sigma, STATA_MLE_SIGMA, atol=1e-4)


def test_mle_improves_loglik_over_twostep_mroz(mroz):
    y, X, s, W, *_ = mroz
    Wn = W / np.max(np.abs(W), axis=0)
    ts = heckman_two_step(y, X, s, Wn)
    ll_ts = heckman_loglik(y, X, s, Wn, ts.beta, ts.gamma, ts.sigma, ts.rho)
    ml = heckman_mle(y, X, s, Wn)
    assert ml.loglik > ll_ts - 1e-6


# ------------------------------------------------------ synthetic recovery


def test_synthetic_recovery_large_n():
    """With an instrument and rho=0.6, both estimators recover the truth at
    n=100k; naive OLS on the selected sample does not."""
    data = make_linear_selection_data(n=100_000, rho=0.6, alpha=1.0,
                                      sigma=1.0, d=2, seed=7)
    n = data.meta["n"]
    X = np.column_stack([np.ones(n), data.x])
    W = np.column_stack([np.ones(n), data.x, data.z])
    beta_true = np.array([0.0, 1.0, 0.5])

    ts = heckman_two_step(data.y, X, data.s, W)
    ml = heckman_mle(data.y, X, data.s, W)
    np.testing.assert_allclose(ts.beta, beta_true, atol=0.05)
    np.testing.assert_allclose(ml.beta, beta_true, atol=0.03)
    np.testing.assert_allclose(ml.rho, 0.6, atol=0.05)
    np.testing.assert_allclose(ml.sigma, 1.0, atol=0.02)

    # naive OLS on selected sample: biased intercept (selection on
    # unobservables shifts E[eps | s=1] != 0)
    sel = data.s > 0.5
    ols, *_ = np.linalg.lstsq(X[sel], data.y[sel], rcond=None)
    assert abs(ols[0] - beta_true[0]) > 0.10


def test_mar_control_rho_zero():
    """rho=0 (selection on observables only): naive OLS is fine and the MLE
    correctly estimates rho ~ 0."""
    data = make_linear_selection_data(n=50_000, rho=0.0, alpha=1.0,
                                      sigma=1.0, d=2, seed=11)
    n = data.meta["n"]
    X = np.column_stack([np.ones(n), data.x])
    W = np.column_stack([np.ones(n), data.x, data.z])
    ml = heckman_mle(data.y, X, data.s, W)
    assert abs(ml.rho) < 0.05
    sel = data.s > 0.5
    ols, *_ = np.linalg.lstsq(X[sel], data.y[sel], rcond=None)
    np.testing.assert_allclose(ols, [0.0, 1.0, 0.5], atol=0.05)


# ----------------------------------------------------------------- numerics


def test_inverse_mills_stability():
    z = np.array([-50.0, -40.0, -10.0, 0.0, 5.0, 40.0])
    lam = inverse_mills(z)
    assert np.all(np.isfinite(lam))
    # asymptote: lambda(z) ~ -z as z -> -inf
    np.testing.assert_allclose(lam[0], 50.0, rtol=1e-3)
    np.testing.assert_allclose(lam[3], np.sqrt(2 / np.pi), rtol=1e-12)
    assert lam[5] < 1e-8
