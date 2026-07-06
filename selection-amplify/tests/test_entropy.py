import numpy as np

from selamp.entropy import binary_entropy_bits, d_comp, mutual_info_OX


def test_binary_entropy_peaks_at_half():
    assert abs(binary_entropy_bits(0.5) - 1.0) < 1e-9
    assert binary_entropy_bits(0.01) < 0.1


def test_IOX_zero_for_constant_s():
    s = np.full(5000, 0.3)
    assert mutual_info_OX(s) < 1e-9
    assert d_comp(s) < 1e-9


def test_IOX_matches_analytic_two_point():
    # half the mass at s=0, half at s=1: H(O)=1, H(O|X)=0 => I=1 bit
    s = np.concatenate([np.zeros(5000), np.ones(5000)])
    assert abs(mutual_info_OX(s) - 1.0) < 1e-6


def test_dcomp_positive_when_varying():
    rng = np.random.default_rng(0)
    s = rng.random(10000)
    assert d_comp(s) > 0
