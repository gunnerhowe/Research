"""Paper B machinery tests: SH censoring harness, pow3 fitting, Tobit
correction direction, and multi-bracket Heckman population recovery."""

import numpy as np
import pytest

from heckesel.lc import (POP_MU, default_pop_cov, curve_values, run_sh,
                         fit_pow3_ls, fit_pow3_tobit, fit_pow3_with_cov,
                         phi_final, EBModel)

T = 52
T_FULL = np.arange(1, T + 1, dtype=float)


def _population(n, seed, sigma):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(default_pop_cov())
    phi = POP_MU + rng.standard_normal((n, 3)) @ L.T
    y_true = curve_values(phi, T_FULL)
    y_obs = y_true + sigma * rng.standard_normal(y_true.shape)
    return phi, y_true, y_obs


def test_run_sh_counts_and_thresholds():
    _, _, y_obs = _population(999, 0, 0.01)
    sh = run_sh(y_obs, [4, 12, 36], 3.0)
    assert sh.alive[:, 1].sum() == 333
    assert sh.alive[:, 2].sum() == 111
    assert sh.alive[:, 3].sum() == 37
    for k, t in enumerate(sh.rungs):
        surv = sh.alive[:, k + 1]
        assert np.all(y_obs[surv, t - 1] >= sh.thresholds[k] - 1e-12)
    # killed at rung k => alive before, below threshold at k
    for i in np.where(sh.kill_rung == 1)[0][:5]:
        assert sh.alive[i, 1] and not sh.alive[i, 2]


def test_fit_pow3_recovers_noiseless():
    phi = np.array([0.85, np.log(0.3), np.log(0.7)])
    y = curve_values(phi[None], T_FULL)[0]
    est = fit_pow3_ls(T_FULL, y)
    assert abs(phi_final(est, T)[0] - y[-1]) < 1e-4


def test_tobit_reduces_survivor_bias():
    """At the first rung, the truncation-corrected per-curve fit must cut
    the naive fit's positive survivor bias."""
    phi, y_true, y_obs = _population(600, 1, 0.02)
    sh = run_sh(y_obs, [4, 12, 36], 3.0)
    surv = np.where(sh.alive[:, 1])[0][:120]
    t = np.arange(1, 5, dtype=float)
    re = np.array([4.0])
    rt = np.array([sh.thresholds[0]])
    naive, tobit = [], []
    for i in surv:
        naive.append(phi_final(fit_pow3_ls(t, y_obs[i, :4]), T)[0])
        tobit.append(phi_final(
            fit_pow3_tobit(t, y_obs[i, :4], re, rt, 0.02), T)[0])
    b_naive = np.mean(np.array(naive) - y_true[surv, -1])
    b_tobit = np.mean(np.array(tobit) - y_true[surv, -1])
    assert b_naive > 0.005                 # the bias exists
    assert b_tobit < b_naive - 0.002       # the correction shrinks it


@pytest.mark.slow
def test_multibracket_heckman_recovers_population():
    """Survivor-only population fit: naive is biased toward the survivor
    population; the Heckman-corrected fit with multi-bracket survival terms
    (the exclusion restriction) recovers the unselected population mean."""
    sigma = 0.01
    rng = np.random.default_rng(0)
    L = np.linalg.cholesky(default_pop_cov())
    groups, ph_all, V_all = [], [], []
    for rungs in ([4, 12, 36], [12, 36], [36]):
        phi = POP_MU + rng.standard_normal((999, 3)) @ L.T
        y_true = curve_values(phi, T_FULL)
        y_obs = y_true + sigma * rng.standard_normal(y_true.shape)
        sh = run_sh(y_obs, rungs, 3.0)
        surv = np.where(sh.alive[:, -1])[0]
        fits = [fit_pow3_with_cov(T_FULL, y_obs[i], sigma) for i in surv]
        ph = np.array([f[0] for f in fits])
        V = np.array([f[1] for f in fits])
        groups.append((np.array(rungs, float), np.array(sh.thresholds),
                       len(ph)))
        ph_all.append(ph)
        V_all.append(V)
    ph_all = np.vstack(ph_all)
    V_all = np.vstack(V_all)

    naive_mu_a = ph_all[:, 0].mean()
    m = EBModel.fit(ph_all, V_all, mode="survivor_heckman",
                    survival_groups=groups, sigma_obs=sigma, seed=0)
    true_a = POP_MU[0]
    assert naive_mu_a - true_a > 0.05          # naive is badly biased
    # corrected fit cuts the bias by >2x (slight overshoot is expected and
    # reported honestly in the paper; magnitude bounded here)
    assert abs(m.mu[0] - true_a) < 0.5 * (naive_mu_a - true_a)
    assert abs(np.sqrt(m.cov[0, 0]) - 0.07) < 0.02  # Sigma recovered
