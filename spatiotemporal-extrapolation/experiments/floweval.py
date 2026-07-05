"""Shared: fit FSS flows from measured curves, build predictions on a target grid,
and score any predictor against measured large-L ground truth (E2/E3/E4 all use
this one metric suite; PLAN.md bands)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from specext.scaling import PointwiseFlow, interp_null                           # noqa: E402
from specext.stats import (median_rel_err, median_abs_log10_ratio, rel_l2,      # noqa: E402
                           corr_from_power, power_from_density, tau_e_leading,
                           slow_set_overlap, band_mask)
from specext.tiling import tile_power, tile_corr                                 # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import K_INRANGE_LO, K_FULL_LO, K_GATE_MAX, DX                       # noqa: E402

R_MAX = 44.0


def fit_edmd_flows(curves, seed=None):
    """Pointwise 1/L flows (the GATE-S recipe) for gamma, omega, S. seed=None uses
    the seed-MEAN training curves (denoised, as GATE S did); an int uses that
    seed's realization (for K2 uncertainty). curves: {L: analyze_measurement}."""
    flows = {}
    for q, log_y in (("gamma", True), ("omega", False), ("s_density", True)):
        size_curves = {}
        for L, c in curves.items():
            arr = c[q]
            nseed = arr.shape[0]
            y = arr.mean(axis=0) if seed is None else arr[seed]
            se = arr.std(axis=0, ddof=1) / np.sqrt(nseed)  # seed SE per mode
            size_curves[L] = (c["k"], y, se)
        flows[q] = PointwiseFlow(log_y=log_y).fit(size_curves)
    return flows


def predict_edmd_flow(flows, k_target, L_target):
    return {"gamma": np.maximum(flows["gamma"].predict(k_target, L_target), 0.0),
            "omega": np.maximum(flows["omega"].predict(k_target, L_target), 0.0),
            "s_density": flows["s_density"].predict(k_target, L_target)}


def predict_interp_null(curves_src, seed, k_target):
    """Single-size no-flow null: PCHIP in k of the seed's measured curves."""
    c = curves_src
    return {"gamma": interp_null(c["k"], c["gamma"][seed], k_target, log_y=True),
            "omega": np.maximum(interp_null(c["k"], c["omega"][seed], k_target), 0.0),
            "s_density": interp_null(c["k"], c["s_density"][seed], k_target,
                                     log_y=True)}


def truth_from_measurement(meas):
    """Seed-mean ground-truth curves at the target size."""
    return {"gamma": meas["gamma"].mean(axis=0),
            "omega": meas["omega"].mean(axis=0),
            "s_density": meas["s_density"].mean(axis=0),
            "gamma_se": meas["gamma"].std(axis=0, ddof=1) / np.sqrt(meas["gamma"].shape[0]),
            "s_se": meas["s_density"].std(axis=0, ddof=1) / np.sqrt(meas["s_density"].shape[0]),
            "k": meas["k"], "p_mean": meas["p_mean"].mean(axis=0),
            "N": meas["N"], "L": meas["L"]}


def score(pred, truth, band=(K_FULL_LO, K_GATE_MAX)):
    """Metric suite for one predictor on the target grid, over the headline
    full-support band [2pi/22, 2.2]. pred/truth carry gamma/omega/s_density on
    truth['k']; NaNs in pred are scored where finite. C(r) is BAND-LIMITED to the
    headline band on both sides (a fair bulk-correlation comparison: the flow
    predicts the bulk modes; genuinely new long-wave modes are scored separately)."""
    k = truth["k"]
    L = truth["L"]
    mask = band_mask(k, *band)
    out = {}
    out["gamma_med_rel"] = median_rel_err(pred["gamma"][mask], truth["gamma"][mask])
    out["omega_med_abs"] = (float(np.nanmedian(np.abs(pred["omega"][mask] -
                                                      truth["omega"][mask])))
                            if np.isfinite(pred["omega"][mask]).any() else np.nan)
    out["omega_typical"] = float(np.nanmedian(np.abs(truth["omega"][mask])))
    out["s_med_log10"] = median_abs_log10_ratio(pred["s_density"][mask],
                                                truth["s_density"][mask])
    # band-limited bulk correlation: reconstruct both sides from the SAME headline
    # band so the comparison isolates the predicted modes (not unmodelled low-k)
    r = np.arange(0.0, R_MAX + DX / 2, DX)
    ok = np.isfinite(pred["s_density"]) & mask
    p_pred = power_from_density(pred["s_density"][ok], L)
    c_pred = np.cos(np.outer(r, k[ok])) @ p_pred
    p_true = power_from_density(truth["s_density"][mask], L)
    c_true = np.cos(np.outer(r, k[mask])) @ p_true
    out["c_rel_l2"] = rel_l2(c_pred, c_true)
    ok = np.isfinite(pred["gamma"]) & mask
    tp = tau_e_leading(np.maximum(pred["gamma"][ok], 1e-6), pred["omega"][ok])
    tt = tau_e_leading(np.maximum(truth["gamma"][ok], 1e-6), truth["omega"][ok])
    out["tau_med_rel"] = median_rel_err(tp, tt)
    # slow-mode subspace: restrict ranking to the headline band (the new-mode band
    # is genuinely slower and is scored separately)
    g_pred = np.where(np.isfinite(pred["gamma"]) & mask, pred["gamma"], np.inf)
    g_true = np.where(mask, truth["gamma"], np.inf)
    out["slow_overlap"] = slow_set_overlap(g_pred, g_true, k)
    return out


def _band_metrics(pred, truth, mask):
    if not mask.any():
        return None
    return {"n_modes": int(mask.sum()),
            "gamma_med_rel": median_rel_err(pred["gamma"][mask], truth["gamma"][mask]),
            "s_med_log10": median_abs_log10_ratio(pred["s_density"][mask],
                                                  truth["s_density"][mask])}


def score_new_band(pred, truth):
    """Metrics for the two out-of-full-support regions, reported separately from
    the headline band: partial-support [2pi/88, 2pi/22) (only larger training
    boxes reach here) and new-mode (k < 2pi/88, no training support -> pure
    extrapolation in k)."""
    k = truth["k"]
    partial = _band_metrics(pred, truth, (k >= K_INRANGE_LO) & (k < K_FULL_LO))
    newmode = _band_metrics(pred, truth, k < K_INRANGE_LO)
    if partial is None and newmode is None:
        return None
    return {"partial_support": partial, "new_mode": newmode}


def strict_tiling_scores(curves_22, seed, truth):
    """Strict-comb tiling null: only C(r) and band-integrated spectrum are
    defined; per-k dynamic metrics are N/A by construction."""
    L_t, N_t = truth["L"], truth["N"]
    p22 = curves_22["p_mean"][seed]
    p_t, comb = tile_power(p22, 22.0, L_t)
    r = np.arange(0.0, R_MAX + DX / 2, DX)
    n22 = curves_22["N"]
    c22 = corr_from_power(p22, 22.0, DX * np.arange(n22))
    c_pred = tile_corr(c22, 22.0, r, DX)
    c_true = corr_from_power(truth["p_mean"], L_t, r)
    # band-integrated spectra (variance in k-bands), comb vs truth
    edges = np.array([K_FULL_LO, 0.6, 0.9, 1.2, 1.5, 2.2])
    k_full_t = 2 * np.pi * np.arange(len(truth["p_mean"])) / L_t
    k_full_tile = 2 * np.pi * np.arange(len(p_t)) / L_t
    bi_true, bi_tile = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bi_true.append(truth["p_mean"][(k_full_t >= lo) & (k_full_t < hi)].sum())
        bi_tile.append(p_t[(k_full_tile >= lo) & (k_full_tile < hi)].sum())
    return {"c_rel_l2": rel_l2(c_pred, c_true),
            "band_power_med_rel": median_rel_err(np.array(bi_tile), np.array(bi_true)),
            "band_edges": edges.tolist(),
            "band_power_tile": [float(x) for x in bi_tile],
            "band_power_true": [float(x) for x in bi_true]}
