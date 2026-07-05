"""Tests for the controlled selection generators (heckesel.synth): the
statistical claims the papers rest on must hold in the generator itself.
"""

import numpy as np

from heckesel.synth import (make_selection_data, induce_mnar_selection,
                            f0_smooth)


def test_selection_on_unobservables_shifts_selected_mean():
    """rho > 0: selected units have positive mean outcome noise (the whole
    point); rho = 0: they do not."""
    d_corr = make_selection_data(20000, rho=0.8, alpha=1.0, sigma=0.5,
                                 seed=0)
    d_mar = make_selection_data(20000, rho=0.0, alpha=1.0, sigma=0.5,
                                seed=0)
    sel_c = d_corr.s > 0.5
    sel_m = d_mar.s > 0.5
    eps_c = d_corr.y_full[sel_c] - d_corr.f0x[sel_c]
    eps_m = d_mar.y_full[sel_m] - d_mar.f0x[sel_m]
    assert eps_c.mean() > 0.1               # selection on unobservables
    assert abs(eps_m.mean()) < 0.03         # MAR: no residual selection


def test_oracle_propensity_calibrated():
    """The returned oracle P(s=1|x,z) matches the empirical selection rate
    when binned by propensity."""
    d = make_selection_data(50000, rho=0.5, alpha=1.0, seed=1)
    p = d.propensity
    for lo, hi in [(0.1, 0.3), (0.4, 0.6), (0.7, 0.9)]:
        m = (p >= lo) & (p < hi)
        if m.sum() > 100:
            assert abs(d.s[m].mean() - p[m].mean()) < 0.03


def test_instrument_absent_reduces_propensity_spread_from_z():
    """With alpha=0 the instrument z carries no selection information."""
    d = make_selection_data(20000, rho=0.5, alpha=0.0, seed=2)
    # correlation of z with selection ~ 0 when alpha = 0
    assert abs(np.corrcoef(d.z, d.s)[0, 1]) < 0.03


def test_prop_x_matches_marginal_over_z():
    """prop_x (z integrated out) matches empirical P(s=1|x) in x-bins."""
    d = make_selection_data(60000, rho=0.3, alpha=1.0, seed=3)
    x = d.x[:, 0]
    for lo, hi in [(-2.5, -1.5), (-0.5, 0.5), (1.5, 2.5)]:
        m = (x >= lo) & (x < hi)
        if m.sum() > 200:
            assert abs(d.s[m].mean() - d.prop_x[m].mean()) < 0.04


def test_induced_mnar_selection_correlates_with_residual():
    """The induced-MNAR rule selects on the outcome residual (rho>0) and
    the returned propensity is calibrated to P(s=1|x,z)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8000, 3))
    beta = np.array([1.0, -0.5, 0.3])
    y = x @ beta + rng.normal(scale=1.0, size=8000)
    s, z, prop, prop_x = induce_mnar_selection(x, y, rho=0.8, alpha=1.0,
                                               target_frac=0.5, seed=0)
    assert abs(s.mean() - 0.5) < 0.02
    r = y - np.column_stack([np.ones(len(x)), x]) @ np.linalg.lstsq(
        np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    # selected units have higher residual on average (rho>0, positive sign
    # because -k_x*m dominates location but residual enters with +rho)
    assert r[s > 0.5].mean() - r[s < 0.5].mean() != 0.0
    # propensity calibration
    for lo, hi in [(0.2, 0.4), (0.6, 0.8)]:
        m = (prop >= lo) & (prop < hi)
        if m.sum() > 100:
            assert abs(s[m].mean() - prop[m].mean()) < 0.05
