"""E3 — the credibility section: nulls and oracles (K2 verdict).

All methods here operate on the data-driven Koopman (EDMD) spectrum, the headline
instrument chosen by the E1/K3 decision. Groups:

 NO large-L data (the flow's peers):
   - interp-22 / interp-44 / interp-88: the measured spectrum at ONE small size,
     smoothly interpolated in k to the target grid, NO size flow. interp-88 is the
     strongest null (use your biggest small box directly). This is also the
     spectral analog of evaluating a small-domain operator on a larger domain
     (the locality/zero-shot route; the neural 2606.14597 version is cite-compared
     in the paper, and E1 shows a deep operator's resonances do not zero-shot).
   - strict tiling: periodic tiling of the L=22 field statistics (translation
     invariance makes this strong for C(r) and band-integrated power).
   - fitted_flow_smallbase: the flow fit on an aggressive base {22,33,44} only.
 SOME large-L data (data efficiency):
   - EDMD at the target from limited data (T=2000, T=10000 time units).
 FULL large-L data (oracle upper bound + compute multiple):
   - EDMD at the target from the full trajectory (= the validation ground truth);
     we report its COMPUTE MULTIPLE over our small-L-only route.

K2 fires iff the flow fails to beat the best no-flow null on a majority of the
headline metrics by more than the seed spread.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (RESULTS, RUNS, SEEDS, SIZES_TRAIN, L_HOLDOUT, L_TARGET,     # noqa: E402
                    analyze_measurement, measure_size, save_json)
from floweval import (fit_edmd_flows, predict_edmd_flow, predict_interp_null,   # noqa: E402
                      strict_tiling_scores, truth_from_measurement, score,
                      score_new_band)

HEADLINE = ("gamma_med_rel", "s_med_log10", "c_rel_l2", "tau_med_rel",
            "slow_overlap")
HIGHER_BETTER = {"slow_overlap"}
NO_FLOW_NULLS = ["interp88", "interp44", "interp22"]


def edmd_limited(L_t, T, tag):
    p = RUNS / f"measure_{tag}.npz"
    if not p.exists():
        p = measure_size(L_t, T=T, tag=tag)
    a = analyze_measurement(p)
    point = {"gamma": a["gamma"].mean(axis=0), "omega": a["omega"].mean(axis=0),
             "s_density": a["s_density"].mean(axis=0)}
    seed_preds = [{"gamma": a["gamma"][s], "omega": a["omega"][s],
                   "s_density": a["s_density"][s]} for s in range(len(SEEDS))]
    return point, seed_preds, float(a["wall_s"])


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    curves = {L: analyze_measurement(RUNS / f"measure_L{L:g}.npz")
              for L in SIZES_TRAIN}
    curves22, curves44, curves88 = curves[22.0], curves[44.0], curves[88.0]
    p33 = RUNS / "measure_L33.npz"
    if not p33.exists():
        p33 = measure_size(33.0, tag="L33")
    curves_small = {22.0: curves22, 33.0: analyze_measurement(p33), 44.0: curves44}
    meas = {L_HOLDOUT: analyze_measurement(RUNS / "measure_L176.npz"),
            L_TARGET: analyze_measurement(RUNS / "measure_L1408.npz")}

    results = {"targets": {}, "compute": {}}
    for L_t in (L_HOLDOUT, L_TARGET):
        truth = truth_from_measurement(meas[L_t])
        k_t = truth["k"]
        entry = {"methods": {}}

        def add(name, point_pred, seed_preds, extra=None):
            # point_estimate = the reported headline (seed-mean prediction, a single
            # scored curve); per_seed = per-seed scores, used only for the spread.
            point = score(point_pred, truth)
            per_seed = [score(p, truth) for p in seed_preds]
            entry["methods"][name] = {
                "point_estimate": point, "per_seed": per_seed,
                "new_band": score_new_band(point_pred, truth)}
            if extra:
                entry["methods"][name].update(extra)
            print(f"L={L_t:g} {name}: " + ", ".join(
                f"{m}={point[m]:.4g}" for m in HEADLINE))

        add("fitted_flow",
            predict_edmd_flow(fit_edmd_flows(curves, None), k_t, L_t),
            [predict_edmd_flow(fit_edmd_flows(curves, s), k_t, L_t) for s in range(3)])
        add("fitted_flow_smallbase",
            predict_edmd_flow(fit_edmd_flows(curves_small, None), k_t, L_t),
            [predict_edmd_flow(fit_edmd_flows(curves_small, s), k_t, L_t) for s in range(3)])
        add("interp22", predict_interp_null(curves22, None, k_t),
            [predict_interp_null(curves22, s, k_t) for s in range(3)])
        add("interp44", predict_interp_null(curves44, None, k_t),
            [predict_interp_null(curves44, s, k_t) for s in range(3)])
        add("interp88", predict_interp_null(curves88, None, k_t),
            [predict_interp_null(curves88, s, k_t) for s in range(3)])
        tile_point = strict_tiling_scores(curves22, None, truth)
        tile_seed = [strict_tiling_scores(curves22, s, truth) for s in range(3)]
        entry["methods"]["strict_tiling"] = {
            "point_estimate": {m: tile_point[m] for m in ("c_rel_l2", "band_power_med_rel")},
            "per_seed": tile_seed}
        print(f"L={L_t:g} strict_tiling: c_rel_l2={tile_point['c_rel_l2']:.4g}, "
              f"band_power={tile_point['band_power_med_rel']:.4g}")
        Ts = (2000.0,) if L_t == L_HOLDOUT else (2000.0, 10000.0)
        for T in Ts:
            point_p, seed_p, wall = edmd_limited(L_t, T, f"L{L_t:g}_short{T:g}")
            add(f"edmd_limited_T{T:g}", point_p, seed_p, {"sim_wall_s": wall})
        results["targets"][f"{L_t:g}"] = entry

    # ---- K2 verdict at L_TARGET: compare point estimates (seed-mean); the flow's
    #      per-seed spread sets the margin it must beat by.
    flow = results["targets"][f"{L_TARGET:g}"]["methods"]["fitted_flow"]
    nulls = results["targets"][f"{L_TARGET:g}"]["methods"]
    wins = {}
    for m in HEADLINE:
        fpoint = flow["point_estimate"][m]
        spread = float(np.nanstd([s[m] for s in flow["per_seed"]], ddof=1))
        best_null, best_pt = None, None
        for nn in NO_FLOW_NULLS:
            nv = nulls[nn]["point_estimate"][m]
            if best_pt is None or (nv > best_pt if m in HIGHER_BETTER else nv < best_pt):
                best_pt, best_null = nv, nn
        better = (fpoint > best_pt + spread if m in HIGHER_BETTER
                  else fpoint < best_pt - spread)
        wins[m] = {"flow_point": fpoint, "flow_spread": spread, "best_null": best_null,
                   "best_null_point": best_pt, "flow_wins": bool(better)}
    n_win = sum(w["flow_wins"] for w in wins.values())
    verdict = {"metrics": wins, "n_wins": n_win, "majority": n_win > len(HEADLINE) / 2}

    # aggressive-base flow vs interp-44 (same win rule)
    sb = results["targets"][f"{L_TARGET:g}"]["methods"]["fitted_flow_smallbase"]
    i44 = results["targets"][f"{L_TARGET:g}"]["methods"]["interp44"]
    sb_wins = {}
    for m in HEADLINE:
        fpoint = sb["point_estimate"][m]
        nv = i44["point_estimate"][m]
        spread = float(np.nanstd([s[m] for s in sb["per_seed"]], ddof=1))
        better = (fpoint > nv + spread if m in HIGHER_BETTER else fpoint < nv - spread)
        sb_wins[m] = {"flow_point": fpoint, "interp44_point": nv, "flow_wins": bool(better)}
    verdict["smallbase_vs_interp44"] = {
        "metrics": sb_wins, "n_wins": sum(w["flow_wins"] for w in sb_wins.values()),
        "majority": sum(w["flow_wins"] for w in sb_wins.values()) > len(HEADLINE) / 2}
    verdict["k2_fires"] = not verdict["majority"]
    results["k2"] = verdict

    # ---- compute accounting (oracle = full large-L EDMD; ours = small-L + fit)
    small_sim = sum(float(np.load(RUNS / f"measure_L{L:g}.npz")["wall_s"])
                    for L in SIZES_TRAIN)
    oracle176 = float(np.load(RUNS / "measure_L176.npz")["wall_s"])
    oracle1408 = float(np.load(RUNS / "measure_L1408.npz")["wall_s"])
    lim1408 = float(np.load(RUNS / "measure_L1408_short10000.npz")["wall_s"])
    results["compute"] = {
        "ours_smallL_sim_s": small_sim,
        "oracle176_full_s": oracle176, "oracle176_multiple": oracle176 / small_sim,
        "oracle1408_full_s": oracle1408, "oracle1408_multiple": oracle1408 / small_sim,
        "limited1408_T10000_s": lim1408,
        "limited1408_multiple": lim1408 / small_sim}
    print(f"\nK2: flow beats best no-flow null on {n_win}/5 headline metrics -> "
          f"{'flow adds value' if verdict['majority'] else 'K2 FIRES'}")
    print(f"    oracle (full L=1408 EDMD) compute multiple over our small-L route: "
          f"{results['compute']['oracle1408_multiple']:.1f}x")
    save_json("exp3_baselines.json", results)


if __name__ == "__main__":
    main()
