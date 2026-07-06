"""The inverted-U adjudicator and paired tests."""
import numpy as np

from doobsyn.stats import (inverted_u_test, monotone_decreasing_frac,
                           paired_wilcoxon, curvature_sign)


def test_inverted_u_detected():
    sig = np.array([0.0, 0.05, 0.1, 0.2, 0.4])
    rng = np.random.default_rng(0)
    # planted inverted-U peaking at sigma=0.1
    base = np.array([0.60, 0.66, 0.70, 0.63, 0.55])
    R = base[None, :] + rng.normal(0, 0.005, (8, 5))
    out = inverted_u_test(sig, R)
    assert out["inverted_u"] is True
    assert out["interior_peak"] is True
    assert 0 < out["i_peak"] < 4
    assert out["p_peak_gt_zero"] < 0.05
    assert out["p_peak_gt_hi"] < 0.05


def test_monotone_curve_is_not_inverted_u():
    sig = np.array([0.0, 0.05, 0.1, 0.2, 0.4])
    rng = np.random.default_rng(1)
    base = np.array([0.70, 0.66, 0.61, 0.55, 0.48])   # monotone decreasing
    R = base[None, :] + rng.normal(0, 0.005, (8, 5))
    out = inverted_u_test(sig, R)
    assert out["inverted_u"] is False
    assert monotone_decreasing_frac(sig, R) > 0.9


def test_curvature_sign_concave_for_inverted_u():
    sig = np.array([0.0, 0.05, 0.1, 0.2, 0.4])
    R = np.array([[0.60, 0.66, 0.70, 0.63, 0.55]])
    assert curvature_sign(sig, R) < 0


def test_paired_wilcoxon_directional():
    rng = np.random.default_rng(2)
    a = rng.normal(0.7, 0.02, 10)
    b = a - 0.05                                   # a strictly above b
    _, p, med = paired_wilcoxon(a, b)
    assert med > 0 and p < 0.05
