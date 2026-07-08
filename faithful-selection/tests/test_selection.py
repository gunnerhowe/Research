"""Gate the estimator core: synthetic recovery, statsmodels agreement,
rho tests, estimand math, heckprob, and the naive-bias demonstration that
motivates the whole paper."""

import numpy as np
import pytest
import statsmodels.api as sm

from faithsel import selection as sel


def simulate_heckman(n, beta, gamma, sigma, rho, seed=0):
    rng = np.random.default_rng(seed)
    k_o = len(beta) - 1
    Xc = rng.normal(size=(n, k_o))
    X = np.column_stack([np.ones(n), Xc])
    zi = rng.integers(0, 2, size=n).astype(float)      # instrument
    W = np.column_stack([X, zi])
    assert len(gamma) == W.shape[1]
    cov = np.array([[1.0, rho * sigma], [rho * sigma, sigma**2]])
    err = rng.multivariate_normal([0, 0], cov, size=n)
    s = (W @ gamma + err[:, 0] > 0).astype(float)
    y_star = X @ beta + err[:, 1]
    y = np.where(s > 0.5, y_star, np.nan)
    return y, y_star, X, s, W


BETA = np.array([1.0, 0.8, -0.5])
GAMMA = np.array([0.2, 0.6, -0.4, 0.9])   # last coord: strong instrument
SIGMA, RHO = 1.3, 0.6


class TestProbit:
    def test_matches_statsmodels(self):
        rng = np.random.default_rng(1)
        n = 4000
        W = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
        s = (W @ np.array([0.3, 0.8, -0.5])
             + rng.normal(size=n) > 0).astype(float)
        ours = sel.probit_fit(W, s)
        smf = sm.Probit(s, W).fit(disp=0)
        np.testing.assert_allclose(ours.params, smf.params, atol=1e-6)
        np.testing.assert_allclose(ours.loglik, smf.llf, rtol=1e-8)
        np.testing.assert_allclose(np.sqrt(np.diag(ours.cov)),
                                   smf.bse, rtol=1e-3)


class TestRecovery:
    def test_two_step_recovers(self):
        y, _, X, s, W = simulate_heckman(60000, BETA, GAMMA, SIGMA, RHO,
                                         seed=2)
        fit = sel.heckman_two_step(y, X, s, W)
        np.testing.assert_allclose(fit.beta, BETA, atol=0.06)
        assert abs(fit.rho - RHO) < 0.1
        assert abs(fit.sigma - SIGMA) < 0.06

    def test_mle_recovers_tighter(self):
        y, _, X, s, W = simulate_heckman(60000, BETA, GAMMA, SIGMA, RHO,
                                         seed=3)
        fit = sel.heckman_mle(y, X, s, W)
        assert fit.converged
        np.testing.assert_allclose(fit.beta, BETA, atol=0.04)
        assert abs(fit.rho - RHO) < 0.05
        assert abs(fit.sigma - SIGMA) < 0.04

    def test_fixed_rho_at_zero_equals_probit_plus_ols(self):
        y, _, X, s, W = simulate_heckman(5000, BETA, GAMMA, SIGMA, RHO,
                                         seed=4)
        fit0 = sel.heckman_mle_fixed_rho(y, X, s, W, rho=0.0)
        selm = s > 0.5
        ols = sm.OLS(y[selm], X[selm]).fit()
        pr = sel.probit_fit(W, s)
        np.testing.assert_allclose(fit0.beta, ols.params, atol=1e-4)
        np.testing.assert_allclose(fit0.gamma, pr.params, atol=1e-4)
        sig_mle = np.sqrt(np.mean(ols.resid**2))
        assert abs(fit0.sigma - sig_mle) < 1e-3
        ll_sep = (pr.loglik
                  + float(np.sum(-0.5 * (ols.resid / fit0.sigma)**2
                                 - np.log(fit0.sigma)
                                 - 0.5 * np.log(2 * np.pi))))
        np.testing.assert_allclose(fit0.loglik, ll_sep, rtol=1e-6)


class TestRhoTests:
    def test_rejects_when_confounded(self):
        y, _, X, s, W = simulate_heckman(8000, BETA, GAMMA, SIGMA, 0.6,
                                         seed=5)
        mle = sel.heckman_mle(y, X, s, W)
        assert sel.rho_wald_test(mle)["p"] < 0.01
        assert sel.rho_lr_test(y, X, s, W, fit_mle=mle)["p"] < 0.01

    def test_no_rejection_when_rho_zero(self):
        y, _, X, s, W = simulate_heckman(8000, BETA, GAMMA, SIGMA, 0.0,
                                         seed=6)
        mle = sel.heckman_mle(y, X, s, W)
        assert sel.rho_wald_test(mle)["p"] > 0.05
        assert sel.rho_lr_test(y, X, s, W, fit_mle=mle)["p"] > 0.05


class TestEstimands:
    def test_naive_biased_corrected_recovers(self):
        """The paper's core claim on synthetic data: with rho > 0, the naive
        selected-sample mean over-states the population mean; the corrected
        estimand recovers it; hidden-mean correction beats assuming zero."""
        y, y_star, X, s, W = simulate_heckman(40000, BETA, GAMMA, SIGMA, 0.7,
                                              seed=7)
        fit = sel.heckman_two_step(y, X, s, W)
        est = sel.estimands(fit, y_star, s, X, W)
        tgt = sel.ground_truth_targets(y_star, s)
        assert est["naive_selected"] - tgt["true_pop"] > 0.15   # biased up
        assert (abs(est["corrected_pop"] - tgt["true_pop"])
                < abs(est["naive_selected"] - tgt["true_pop"]) / 3)
        assert (abs(est["corrected_hidden"] - tgt["true_hidden"])
                < abs(0.0 - tgt["true_hidden"]) / 3)

    def test_conditional_mean_formulas(self):
        """E[y|x,V=1] and E[y|x,V=0] closed forms vs empirical means."""
        beta = np.array([0.5])
        gamma = np.array([0.3, 0.8])
        n = 400000
        rng = np.random.default_rng(8)
        X = np.ones((n, 1))
        W = np.column_stack([X, rng.integers(0, 2, n).astype(float)])
        cov = np.array([[1.0, 0.6 * 1.2], [0.6 * 1.2, 1.44]])
        err = rng.multivariate_normal([0, 0], cov, size=n)
        s = (W @ gamma + err[:, 0] > 0).astype(float)
        y_star = X @ beta + err[:, 1]
        fit = sel.HeckmanResult(beta=beta, gamma=gamma, sigma=1.2, rho=0.6,
                                beta_lambda=0.72, loglik=None, method="oracle",
                                converged=True)
        selm = s > 0.5
        pred_obs = fit.predict_observed(X, W)
        pred_hid = fit.predict_hidden(X, W)
        assert abs(np.mean(pred_obs[selm]) - np.mean(y_star[selm])) < 0.01
        assert abs(np.mean(pred_hid[~selm]) - np.mean(y_star[~selm])) < 0.01

    def test_rho_sensitivity_brackets_truth(self):
        y, y_star, X, s, W = simulate_heckman(20000, BETA, GAMMA, SIGMA, 0.6,
                                              seed=9)
        rows = sel.rho_sensitivity(y, X, s, W,
                                   rho_grid=(0.0, 0.3, 0.6, 0.9))
        tgt = sel.ground_truth_targets(y_star, s)
        best = min(rows, key=lambda r: abs(r["corrected_pop"]
                                           - tgt["true_pop"]))
        assert best["rho_fixed"] == pytest.approx(0.6)


class TestPhi2:
    def test_matches_scipy(self):
        from scipy.stats import multivariate_normal
        import torch
        rng = np.random.default_rng(10)
        a = rng.normal(size=50)
        b = rng.normal(size=50)
        for rho in (-0.85, -0.3, 0.0, 0.45, 0.9):
            ours = sel._phi2_torch(torch.tensor(a), torch.tensor(b),
                                   torch.tensor(float(rho))).numpy()
            ref = np.array([
                multivariate_normal.cdf([ai, bi], mean=[0, 0],
                                        cov=[[1, rho], [rho, 1]])
                for ai, bi in zip(a, b)])
            np.testing.assert_allclose(ours, ref, atol=1e-8)


class TestHeckprob:
    def simulate(self, n, rho, seed=11):
        rng = np.random.default_rng(seed)
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        W = np.column_stack([X, rng.integers(0, 2, n).astype(float)])
        beta = np.array([0.2, 0.7])
        gamma = np.array([-0.1, 0.5, 0.8])
        cov = np.array([[1.0, rho], [rho, 1.0]])
        err = rng.multivariate_normal([0, 0], cov, size=n)
        s = (W @ gamma + err[:, 0] > 0).astype(float)
        y_star = (X @ beta + err[:, 1] > 0).astype(float)
        y = np.where(s > 0.5, y_star, np.nan)
        return y, y_star, X, s, W, beta, gamma

    def test_recovery(self):
        y, _, X, s, W, beta, gamma = self.simulate(60000, 0.6)
        fit = sel.heckprob_mle(y, X, s, W)
        assert fit.converged
        np.testing.assert_allclose(fit.beta, beta, atol=0.06)
        np.testing.assert_allclose(fit.gamma, gamma, atol=0.05)
        assert abs(fit.rho - 0.6) < 0.08

    def test_rho_test_power_and_level(self):
        y, _, X, s, W, *_ = self.simulate(20000, 0.6, seed=12)
        fit = sel.heckprob_mle(y, X, s, W)
        t = sel.heckprob_rho_tests(y, X, s, W, fit)
        assert t["lr_p"] < 0.01 and t["wald_p"] < 0.01
        y, _, X, s, W, *_ = self.simulate(20000, 0.0, seed=13)
        fit = sel.heckprob_mle(y, X, s, W)
        t = sel.heckprob_rho_tests(y, X, s, W, fit)
        assert t["lr_p"] > 0.05

    def test_hidden_prediction(self):
        y, y_star, X, s, W, *_ = self.simulate(200000, 0.7, seed=14)
        fit = sel.heckprob_mle(y, X, s, W)
        selm = s > 0.5
        pred_hidden = fit.predict_hidden(X, W)
        true_hidden = y_star[~selm].mean()
        assert abs(pred_hidden[~selm].mean() - true_hidden) < 0.02
        naive_err = abs(y_star[selm].mean() - true_hidden)
        assert abs(pred_hidden[~selm].mean() - true_hidden) < naive_err / 3


class TestBootstrap:
    def test_ci_covers(self):
        y, y_star, X, s, W = simulate_heckman(4000, BETA, GAMMA, SIGMA, 0.6,
                                              seed=15)
        out = sel.bootstrap_fit(y, X, s, W, y_full=y_star, n_boot=200, seed=0)
        assert out["rho"]["lo95"] < 0.6 < out["rho"]["hi95"]
        assert out["n_fail"] < 20
