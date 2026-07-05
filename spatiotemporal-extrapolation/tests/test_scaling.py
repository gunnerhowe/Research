"""Scaling-fit round trips (house test: plant a flow, recover it)."""
import numpy as np

from specext.scaling import (pointwise_fit, pointwise_predict, SmoothFlow,
                             interp_null, L_BASE)
from specext.tiling import tile_power, tile_corr


def test_pointwise_roundtrip():
    L = np.array([22.0, 44.0, 66.0, 88.0])
    rng = np.random.default_rng(0)
    y = 1.7 + 3.1 * (L_BASE / L) + rng.normal(0, 1e-4, L.shape)
    fit = pointwise_fit(L, y, se=np.full(4, 1e-4), power=1)
    assert abs(fit["y_inf"] - 1.7) < 1e-3
    assert abs(fit["c"] - 3.1) < 5e-3
    assert fit["r2"] > 0.999
    pred = pointwise_predict(fit, 1408.0)
    assert abs(pred - (1.7 + 3.1 * L_BASE / 1408)) < 1e-3


def test_pointwise_form_selection():
    L = np.array([22.0, 44.0, 66.0, 88.0])
    y2 = 0.5 + 2.0 * (L_BASE / L) ** 2
    f1 = pointwise_fit(L, y2, power=1)
    f2 = pointwise_fit(L, y2, power=2)
    assert f2["aicc"] < f1["aicc"]


def test_smooth_flow_roundtrip():
    rng = np.random.default_rng(1)
    rows_k, rows_L, rows_y = [], [], []
    f0 = lambda k: 0.2 + 0.5 * np.sin(1.5 * k)
    f1 = lambda k: -0.3 + 0.2 * k
    for L in [22.0, 44.0, 66.0, 88.0]:
        k = np.arange(1, int(3.0 * L / (2 * np.pi)) + 1) * 2 * np.pi / L
        y = f0(k) + (L_BASE / L) * f1(k) + rng.normal(0, 1e-3, k.shape)
        rows_k.append(k), rows_L.append(np.full_like(k, L)), rows_y.append(y)
    flow = SmoothFlow(k_lo=0.05, k_hi=3.05, knot_spacing=0.25).fit(
        np.concatenate(rows_k), np.concatenate(rows_L), np.concatenate(rows_y))
    k_t = np.arange(2, 660) * 2 * np.pi / 1408.0
    k_t = k_t[k_t >= 0.29]  # in-range band only (low-k extrapolation is E4)
    pred = flow.predict(k_t, 1408.0)
    true = f0(k_t) + (L_BASE / 1408.0) * f1(k_t)
    assert np.abs(pred - true).max() < 0.02


def test_interp_null_and_tiling():
    k22 = np.arange(1, 11) * 2 * np.pi / 22.0
    y22 = np.exp(-k22)
    kt = np.arange(1, 640) * 2 * np.pi / 1408.0
    out = interp_null(k22, y22, kt, log_y=True)
    inside = (kt >= k22[0]) & (kt <= k22[-1])
    assert np.abs(np.log(out[inside]) - (-kt[inside])).max() < 0.02
    # strict tiling conventions
    p_small = np.array([0.0, 1.0, 0.5, 0.25])
    p_t, comb = tile_power(p_small, 22.0, 88.0)
    assert p_t.sum() == p_small.sum() and comb.sum() == 4
    assert p_t[4] == 1.0 and p_t[8] == 0.5
    c_small = np.cos(np.linspace(0, 2 * np.pi, 64, endpoint=False))
    dx = 22.0 / 64
    r_t = np.array([0.0, 22.0, 44.0 + dx])
    c_t = tile_corr(c_small, 22.0, r_t, dx)
    assert np.allclose(c_t, [c_small[0], c_small[0], c_small[1]])
