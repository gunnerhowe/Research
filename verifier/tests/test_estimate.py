"""Estimator recovery gates — no model calls. Ground truth is known by
construction, so a good estimator must recover it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from novjudge.estimate import make_synthetic, fit_mixed, hackability_index  # noqa: E402


def test_recovers_known_effects():
    df = make_synthetic(n_stems=200, beta_S=1.5, beta_G=0.8, beta_SG=0.0, seed=1)
    fit = fit_mixed(df, judge="synthetic", B=500, seed=1)
    assert abs(fit.beta_S - 1.5) < 0.2, fit.beta_S
    assert abs(fit.beta_G - 0.8) < 0.2, fit.beta_G
    # 95% CIs should cover the truth.
    assert fit.ci_S[0] <= 1.5 <= fit.ci_S[1]
    assert fit.ci_G[0] <= 0.8 <= fit.ci_G[1]


def test_null_signal_effect_ci_covers_zero():
    df = make_synthetic(n_stems=200, beta_S=1.5, beta_G=0.0, seed=2)
    fit = fit_mixed(df, judge="synthetic", B=500, seed=2)
    assert fit.ci_G[0] <= 0.0 <= fit.ci_G[1], fit.ci_G


def test_hackability_index_directions():
    # Signal dominates (beta_G large, beta_S ~0): LH should beat HL → index high.
    hi = make_synthetic(n_stems=150, beta_S=0.1, beta_G=2.0, noise_sd=0.2, seed=3)
    assert hackability_index(hi, judge="synthetic", B=300, seed=3)["hackability_index"] > 0.8
    # Substance dominates: HL beats LH → index low.
    lo = make_synthetic(n_stems=150, beta_S=2.0, beta_G=0.1, noise_sd=0.2, seed=4)
    assert hackability_index(lo, judge="synthetic", B=300, seed=4)["hackability_index"] < 0.2
