"""E4 — honest boundary: push the flow until it breaks.

 (1) extrapolation distance: error vs L over {176, 352, 704, 1408, 2816};
 (2) boundary-condition sensitivity: odd-parity (Dirichlet-type) KS at L=176 —
     bulk two-point correlation, variance healing profile, and bulk-point
     temporal autocorrelation vs the periodic-trained flow's predictions;
 (3) band breakdown at 1408: energy-containing vs microscale (non-universal) k;
 (4) new-mode band (k below the smallest trained wavenumber) — where the route
     must extrapolate in k as well as in L.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (RUNS, SEEDS, SIZES_TRAIN, L_HOLDOUT, L_TARGET, DT_S, DX,    # noqa: E402
                    K_INRANGE_LO, analyze_measurement, measure_size, n_of,
                    save_json, WELCH_BLOCK)
from floweval import (fit_edmd_flows, predict_edmd_flow, predict_interp_null,   # noqa: E402
                      truth_from_measurement, score, score_new_band, R_MAX)
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from specext.ks import ks_stream_batch                                           # noqa: E402
from specext.edmd import WelchAccum                                              # noqa: E402
from specext.stats import rel_l2, power_from_density                             # noqa: E402

LADDER = [176.0, 352.0, 704.0, 1408.0, 2816.0]


def ladder_measurements(skip_sim):
    paths = {}
    for L in LADDER:
        tag = f"L{L:g}"
        p = RUNS / f"measure_{tag}.npz"
        if not p.exists() or not skip_sim:
            if p.exists():
                paths[L] = p
                continue
            seeds = SEEDS if L < 2816 else [0]
            p = measure_size(L, seeds=seeds, tag=tag)
        paths[L] = p
    return paths


def error_vs_L(paths, curves):
    rows = []
    for L in LADDER:
        meas = analyze_measurement(paths[L])
        truth = truth_from_measurement(meas)
        k_t = truth["k"]
        for name in ("fitted_flow", "interp88"):
            # seed-mean point estimate (consistent with E2/E3 headline estimator)
            if name == "fitted_flow":
                pred = predict_edmd_flow(fit_edmd_flows(curves, None), k_t, L)
            else:
                pred = predict_interp_null(curves[88.0], None, k_t)
            s = score(pred, truth)
            rows.append({"L": L, "method": name,
                         "gamma_med_rel": s["gamma_med_rel"],
                         "s_med_log10": s["s_med_log10"], "c_rel_l2": s["c_rel_l2"]})
            print(f"ladder L={L:g} {name}: gamma {rows[-1]['gamma_med_rel']:.4f} "
                  f"S {rows[-1]['s_med_log10']:.4f} C {rows[-1]['c_rel_l2']:.4f}")
    return rows


def odd_parity_study(curves, skip_sim):
    """Dirichlet-type KS at L=176: bulk statistics vs periodic + flow predictions."""
    L, N = L_HOLDOUT, n_of(L_HOLDOUT)
    T = 100_000.0
    n_samples = int(T / DT_S)
    cache = RUNS / "odd176_stats.npz"
    if not (skip_sim and cache.exists()):
        S = len(SEEDS)
        n_lag = 129
        c_r = np.zeros((S, n_lag))
        var_x = np.zeros((S, N))
        wel = WelchAccum(WELCH_BLOCK, S, 1)
        n_t = 0
        bulk = slice(N // 4, 3 * N // 4)
        for field in ks_stream_batch(L, N, n_samples, SEEDS, dt_sample=DT_S,
                                     transient=500.0, bc="odd",
                                     chunk_samples=4096):
            f = field.transpose(1, 0, 2)  # (S, n_t, N)
            var_x += (f ** 2).sum(axis=1)
            for rr in range(n_lag):
                c_r[:, rr] += (f[:, :, bulk] *
                               f[:, :, N // 4 + rr:3 * N // 4 + rr]).mean(axis=2).sum(axis=1)
            wel.add(field[:, :, N // 2][:, :, None].astype(np.complex128))
            n_t += field.shape[0]
        acf_pt = wel.autocorr(max_lag=100)[:, :, 0].real  # (101, S)
        np.savez_compressed(cache, c_r=c_r / n_t, var_x=var_x / n_t,
                            acf_pt=acf_pt, n_t=n_t)
    z = np.load(cache)
    c_r_odd = z["c_r"].mean(axis=0)
    var_prof = z["var_x"].mean(axis=0)
    acf_pt_odd = z["acf_pt"].mean(axis=1)
    r = DX * np.arange(129)
    t = np.arange(101) * DT_S

    # periodic ground truth at 176 (from E0 measurement)
    per = analyze_measurement(RUNS / "measure_L176.npz")
    truth = truth_from_measurement(per)
    k_t = truth["k"]
    c_per = np.cos(np.outer(r, 2 * np.pi * np.arange(len(truth["p_mean"])) / L)) \
        @ truth["p_mean"]
    # periodic point autocorrelation from mode autocorrs (retained band)
    from common import WelchAccumFromSums
    zper = np.load(RUNS / "measure_L176.npz")
    acf_modes = WelchAccumFromSums(zper["welch_psum"], int(zper["welch_nblocks"]),
                                   WELCH_BLOCK).autocorr(max_lag=100)
    acf_pt_per = 2.0 * acf_modes.real.sum(axis=2).mean(axis=1)

    # flow prediction (fitted flow, per-seed median) of the same objects
    preds = [predict_edmd_flow(fit_edmd_flows(curves, s), k_t, L) for s in SEEDS]
    g = np.nanmedian([p["gamma"] for p in preds], axis=0)
    om = np.nanmedian([p["omega"] for p in preds], axis=0)
    sd = np.nanmedian([p["s_density"] for p in preds], axis=0)
    p_pred = power_from_density(sd, L)
    c_flow = np.cos(np.outer(r, k_t)) @ p_pred
    acf_pt_flow = (p_pred[None, :] * np.exp(-np.outer(t, g)) *
                   np.cos(np.outer(t, om))).sum(axis=1)

    rho = lambda c: c / c[0]
    # variance healing length: distance from wall to within 10% of bulk variance
    bulk_var = var_prof[N // 4:3 * N // 4].mean()
    close = np.abs(var_prof - bulk_var) <= 0.1 * bulk_var
    heal_idx = np.argmax(close)  # first x index within 10%
    out = {
        "c_odd_vs_periodic_rel_l2": rel_l2(rho(c_r_odd), rho(c_per)),
        "c_odd_vs_flow_rel_l2": rel_l2(rho(c_r_odd), rho(c_flow)),
        "c_flow_vs_periodic_rel_l2": rel_l2(rho(c_flow), rho(c_per)),
        "acf_pt_odd_vs_periodic_rel_l2": rel_l2(rho(acf_pt_odd), rho(acf_pt_per)),
        "acf_pt_odd_vs_flow_rel_l2": rel_l2(rho(acf_pt_odd), rho(acf_pt_flow)),
        "healing_length": float(heal_idx * DX),
        "bulk_var_odd": float(bulk_var),
        "var_periodic": float(truth["p_mean"].sum()),
        "curves": {"r": r.tolist(), "c_odd": rho(c_r_odd).tolist(),
                   "c_per": rho(c_per).tolist(), "c_flow": rho(c_flow).tolist(),
                   "t": t.tolist(), "acf_pt_odd": rho(acf_pt_odd).tolist(),
                   "acf_pt_per": rho(acf_pt_per).tolist(),
                   "acf_pt_flow": rho(acf_pt_flow).tolist(),
                   "var_profile": var_prof.tolist()},
    }
    print(f"odd-parity: C(r) odd-vs-flow relL2 {out['c_odd_vs_flow_rel_l2']:.4f}, "
          f"point-acf odd-vs-flow relL2 {out['acf_pt_odd_vs_flow_rel_l2']:.4f}, "
          f"healing length {out['healing_length']:.1f}")
    return out


def band_breakdown(curves):
    meas = analyze_measurement(RUNS / "measure_L1408.npz")
    truth = truth_from_measurement(meas)
    k_t = truth["k"]
    bands = {"energy": (K_INRANGE_LO, 1.5), "microscale": (1.5, 3.0)}
    out = {"fitted_flow": {}}
    pred = predict_edmd_flow(fit_edmd_flows(curves, None), k_t, L_TARGET)  # seed-mean
    for bn, band in bands.items():
        out["fitted_flow"][bn] = score(pred, truth, band=band)
        print(f"band fitted_flow/{bn}: gamma {out['fitted_flow'][bn]['gamma_med_rel']:.4f} "
              f"S {out['fitted_flow'][bn]['s_med_log10']:.4f}")
    return out


def convergence_to_limit():
    """Why K2 fires: median rel distance of gamma(k;L) and S(k;L) to the L=1408
    limit, on the common grid, vs L. Quantifies how fast KS spectra converge (the
    mechanism behind interp-88 matching the target)."""
    from common import common_grid_indices
    limit = analyze_measurement(RUNS / "measure_L1408.npz")
    idxL, kc = common_grid_indices(limit["k"])
    g_lim = limit["gamma"].mean(axis=0)[idxL]
    s_lim = limit["s_density"].mean(axis=0)[idxL]
    rows = []
    for L in [22.0, 44.0, 66.0, 88.0, 176.0, 352.0, 704.0]:
        c = analyze_measurement(RUNS / f"measure_L{L:g}.npz")
        idx, _ = common_grid_indices(c["k"])
        g = c["gamma"].mean(axis=0)[idx]
        s = c["s_density"].mean(axis=0)[idx]
        rows.append({"L": L,
                     "gamma_rel_to_limit": float(np.nanmedian(np.abs(g - g_lim) / np.abs(g_lim))),
                     "s_rel_to_limit": float(np.nanmedian(np.abs(s - s_lim) / np.abs(s_lim)))})
        print(f"convergence L={L:g}: gamma {rows[-1]['gamma_rel_to_limit']:.1%} "
              f"S {rows[-1]['s_rel_to_limit']:.1%} from L=1408 limit")
    return rows


def corr_length(curve, L):
    """Integral (envelope) correlation length xi = int_0^{L/2} |C(r)|/C(0) dr,
    over the first half-domain to avoid the periodic revival C(r)=C(L-r). Robust to
    the fast pattern oscillation. Same definition as E5, so the two systems'
    xi are directly comparable (both turn out short, ~5 units -- the regime is set
    by the spectral finite-size drift, not by real-space correlation length)."""
    from specext.stats import corr_from_power
    p_full = curve["p_mean"].mean(axis=0)
    r = DX * np.arange(max(2, int((L / 2) / DX)))
    env = np.abs(corr_from_power(p_full, L, r))
    env = env / env[0]
    return float(np.sum(0.5 * (env[:-1] + env[1:])) * DX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sim", action="store_true")
    args = ap.parse_args()
    curves = {L: analyze_measurement(RUNS / f"measure_L{L:g}.npz")
              for L in SIZES_TRAIN}
    paths = ladder_measurements(skip_sim=True)
    xi = {f"{L:g}": corr_length(curves[L], L) for L in SIZES_TRAIN}
    print("KS corr length xi:", {k: round(v, 1) for k, v in xi.items()})
    results = {"convergence": convergence_to_limit(),
               "corr_length": xi,
               "ladder": error_vs_L(paths, curves),
               "odd_parity": odd_parity_study(curves, args.skip_sim),
               "bands_1408": band_breakdown(curves)}
    save_json("exp4_boundary.json", results)


if __name__ == "__main__":
    main()
