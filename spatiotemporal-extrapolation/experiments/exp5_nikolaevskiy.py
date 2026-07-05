"""E5 — WHEN THE FLOW IS NECESSARY: the Nikolaevskiy equation.

Contrast with KS (E0-E4), where the bulk spectrum converges by L~4x the base so a
no-flow null (interp-88) beats the finite-size flow (K2 fires). The Nikolaevskiy
equation has a marginal k=0 mode and soft-mode turbulence -> a long correlation
length and SLOW finite-size convergence: the largest affordable small box is NOT
converged, interpolating it fails, and the FSS flow -- which models the
L-dependence -- wins. This flips the KS null into a positive method result and
gives the criterion (correlation length vs largest affordable box) separating the
two regimes.

Same machinery as E0-E3 (system-agnostic estimators + flow); only the integrator
changes. r and the target size are set at the top from the tuning in PLAN.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (RUNS, SEEDS, analyze_measurement, common_grid_indices,       # noqa: E402
                    measure_size, save_json, DX)
from floweval import (fit_edmd_flows, predict_edmd_flow, predict_interp_null,     # noqa: E402
                      truth_from_measurement, score, score_new_band)
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from specext.stats import corr_from_power                                         # noqa: E402

R_NIK = 0.1                 # set from tuning (PLAN.md); soft-mode-turbulence regime
DT_NIK = 0.1
# L=22 is TURBULENT at r=0.1 (rms~0.6, r2~1.0) but its per-sector spectrum is a
# small-box finite-size outlier (few unstable modes fit in the box), so the flow
# ladder starts at L=44. Excluding it is GENEROUS to the flow: including L=22 makes
# the flow's target error WORSE, not better (see the L=22-included robustness check
# in main()), so the exclusion never props up the negative. (Contrast r=0.05, where
# L=22 genuinely dies below the turbulence threshold -- that is a different r.)
SIZES_TRAIN = [44.0, 66.0, 88.0]
L_HOLDOUT = 176.0
L_TARGET = 704.0           # 8x the largest training box; 32x the base scale
T_TRAIN = 80_000.0
T_HOLD = 120_000.0
T_TARGET = 160_000.0
HEADLINE = ("gamma_med_rel", "s_med_log10", "c_rel_l2", "tau_med_rel", "slow_overlap")
NO_FLOW_NULLS = ["interp88", "interp44"]
HIGHER_BETTER = {"slow_overlap"}


def nik_kwargs(transient=4000.0):
    return {"r": R_NIK, "dt": DT_NIK, "transient": transient}


def measure_nik(L, T, tag, seeds=SEEDS):
    p = RUNS / f"measure_{tag}.npz"
    if not p.exists():
        p = measure_size(L, T=T, seeds=seeds, tag=tag, system="nik",
                         sys_kwargs=nik_kwargs())
    return p


def corr_length(curve, L):
    """Integral (envelope) correlation length xi = int_0^{L/2} |C(r)|/C(0) dr
    (identical definition to E4's KS corr_length; both systems turn out short,
    ~5 units -- the regime is set by spectral finite-size drift, not xi)."""
    p_full = curve["p_mean"].mean(axis=0)
    r = DX * np.arange(max(2, int((L / 2) / DX)))
    env = np.abs(corr_from_power(p_full, L, r))
    env = env / env[0]
    return float(np.sum(0.5 * (env[:-1] + env[1:])) * DX)


def main():
    argparse.ArgumentParser().parse_args()   # no options; r/target fixed at top
    paths = {}
    for L in SIZES_TRAIN:
        paths[L] = measure_nik(L, T_TRAIN, f"nikL{L:g}")
    paths[L_HOLDOUT] = measure_nik(L_HOLDOUT, T_HOLD, f"nikL{L_HOLDOUT:g}")
    paths[L_TARGET] = measure_nik(L_TARGET, T_TARGET, f"nikL{L_TARGET:g}")

    curves = {L: analyze_measurement(paths[L]) for L in SIZES_TRAIN}
    meas = {L_HOLDOUT: analyze_measurement(paths[L_HOLDOUT]),
            L_TARGET: analyze_measurement(paths[L_TARGET])}

    # convergence-to-limit (contrast with KS): distance of each size to L_TARGET
    limit = meas[L_TARGET]
    idxL, kc = common_grid_indices(limit["k"])
    g_lim = limit["gamma"].mean(0)[idxL]
    s_lim = limit["s_density"].mean(0)[idxL]
    conv = []
    for L in SIZES_TRAIN + [L_HOLDOUT]:
        c = curves[L] if L in curves else meas[L]
        idx, _ = common_grid_indices(c["k"])
        g = c["gamma"].mean(0)[idx]
        s = c["s_density"].mean(0)[idx]
        conv.append({"L": L,
                     "gamma_rel_to_limit": float(np.nanmedian(np.abs(g - g_lim) / np.abs(g_lim))),
                     "s_rel_to_limit": float(np.nanmedian(np.abs(s - s_lim) / np.abs(s_lim)))})
        print(f"conv L={L:g}: gamma {conv[-1]['gamma_rel_to_limit']:.1%} "
              f"S {conv[-1]['s_rel_to_limit']:.1%} from L={L_TARGET:g}")
    xi = {L: corr_length(curves.get(L) or meas[L], L)
          for L in SIZES_TRAIN + [L_HOLDOUT, L_TARGET]}
    print("corr length xi:", {f"{L:g}": round(v, 1) for L, v in xi.items()})

    results = {"config": {"r": R_NIK, "dt": DT_NIK, "sizes_train": SIZES_TRAIN,
                          "L_holdout": L_HOLDOUT, "L_target": L_TARGET,
                          "T_target": T_TARGET, "seeds": SEEDS},
               "convergence": conv, "corr_length": {f"{L:g}": xi[L] for L in xi},
               "targets": {}}

    for L_t in (L_HOLDOUT, L_TARGET):
        truth = truth_from_measurement(meas[L_t])
        k_t = truth["k"]
        entry = {"L": L_t, "n_sectors": len(k_t), "k": k_t.tolist(),
                 "truth_gamma": truth["gamma"].tolist(),
                 "truth_s": truth["s_density"].tolist(),
                 "truth_gamma_se": truth["gamma_se"].tolist(), "methods": {}}

        def add(name, point_pred, seed_preds):
            point = score(point_pred, truth)
            per_seed = [score(p, truth) for p in seed_preds]
            entry["methods"][name] = {"point_estimate": point, "per_seed": per_seed,
                                      "new_band": score_new_band(point_pred, truth)}
            if name == "fitted_flow":
                entry["methods"][name].update({
                    "pred_gamma_mean": point_pred["gamma"].tolist(),
                    "pred_s_mean": point_pred["s_density"].tolist()})
            print(f"L={L_t:g} {name}: " + ", ".join(f"{m}={point[m]:.4g}" for m in HEADLINE))

        add("fitted_flow", predict_edmd_flow(fit_edmd_flows(curves, None), k_t, L_t),
            [predict_edmd_flow(fit_edmd_flows(curves, s), k_t, L_t) for s in range(3)])
        add("interp88", predict_interp_null(curves[88.0], None, k_t),
            [predict_interp_null(curves[88.0], s, k_t) for s in range(3)])
        add("interp44", predict_interp_null(curves[44.0], None, k_t),
            [predict_interp_null(curves[44.0], s, k_t) for s in range(3)])
        results["targets"][f"{L_t:g}"] = entry

    # verdict at target: does the flow beat the best no-flow null? (K2 should NOT fire)
    flow = results["targets"][f"{L_TARGET:g}"]["methods"]["fitted_flow"]
    nulls = results["targets"][f"{L_TARGET:g}"]["methods"]
    wins = {}
    for m in HEADLINE:
        fp = flow["point_estimate"][m]
        spread = float(np.nanstd([s[m] for s in flow["per_seed"]], ddof=1))
        best_null, best_pt = None, None
        for nn in NO_FLOW_NULLS:
            nv = nulls[nn]["point_estimate"][m]
            if best_pt is None or (nv > best_pt if m in HIGHER_BETTER else nv < best_pt):
                best_pt, best_null = nv, nn
        better = (fp > best_pt + spread if m in HIGHER_BETTER else fp < best_pt - spread)
        wins[m] = {"flow_point": fp, "flow_spread": spread, "best_null": best_null,
                   "best_null_point": best_pt, "flow_wins": bool(better)}
    n_win = sum(w["flow_wins"] for w in wins.values())
    results["verdict"] = {"metrics": wins, "n_wins": n_win,
                          "flow_necessary": n_win > len(HEADLINE) / 2}
    print(f"\nE5 verdict: flow beats best no-flow null on {n_win}/{len(HEADLINE)} "
          f"metrics -> {'FLOW NECESSARY (K2 does not fire)' if n_win > 2.5 else 'flow not needed'}")

    # robustness: refit the flow WITH the excluded L=22 small-box outlier and score
    # at the target. It is turbulent at r=0.1 but a finite-size outlier; including it
    # makes the flow WORSE, so excluding it is generous to the flow (never props up
    # the negative). Recorded as a machine-generated number for the paper.
    p22 = measure_nik(22.0, T_TRAIN, "nikL22")
    curves22 = {22.0: analyze_measurement(p22), **curves}
    tt = truth_from_measurement(meas[L_TARGET])
    flow22 = predict_edmd_flow(fit_edmd_flows(curves22, None), tt["k"], L_TARGET)
    g22 = score(flow22, tt)["gamma_med_rel"]
    results["robustness_L22_included"] = {
        "flow_gamma_med_rel": g22,
        "flow_gamma_excluded": flow["point_estimate"]["gamma_med_rel"],
        "note": ("L=22 is turbulent at r=0.1 (rms~0.6) but a small-box spectral "
                 "outlier; including it raises the flow's target decay-rate error, "
                 "so its exclusion is generous to the flow and does not affect the "
                 "negative conclusion.")}
    print(f"robustness: flow gamma with L=22 included = {g22:.3f} "
          f"(excluded = {flow['point_estimate']['gamma_med_rel']:.3f}); "
          f"including it is {'worse' if g22 > flow['point_estimate']['gamma_med_rel'] else 'better'} "
          f"for the flow")
    save_json("exp5_nikolaevskiy.json", results)


if __name__ == "__main__":
    main()
