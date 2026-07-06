import numpy as np

from selamp.stats import (bootstrap_ci, paired_wilcoxon, rank_biserial,
                          spearman_onesided_pos)


def test_paired_wilcoxon_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 12)
    b = rng.normal(0.0, 0.1, 12)
    p, eff = paired_wilcoxon(a, b, alternative="greater")
    assert p < 0.05 and eff > 0.5


def test_paired_wilcoxon_nan_on_degenerate():
    a = np.array([0.3, 0.3, 0.3])
    p, eff = paired_wilcoxon(a, a)
    assert np.isnan(p)


def test_rank_biserial_sign():
    a = np.array([1.0, 2, 3, 4, 5])
    b = np.array([0.0, 1, 2, 3, 4])
    assert rank_biserial(a, b) > 0.9


def test_spearman_onesided_positive():
    x = np.arange(10.0)
    y = 2 * x + 1
    rho, p = spearman_onesided_pos(x, y)
    assert rho > 0.99 and p < 0.05


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 0.2, 200)
    lo, hi = bootstrap_ci(x, n_boot=2000)
    assert lo < 1.0 < hi
