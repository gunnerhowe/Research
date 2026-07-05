"""E2 — THE CLAIM: extrapolate learned (data-driven Koopman/EDMD) operator spectra
in domain size via a finite-size-scaling flow.

Fits the finite-size-scaling flow on L in {22,44,66,88} and predicts the
per-sector leading Ruelle-Pollicott resonances and spectral density at L = 176
(holdout) and L = 1408 (64x the base), with ZERO large-domain data. Per the K3
decision recorded in E1/PLAN.md, the operator whose spectrum is scaled is the
data-driven Koopman (EDMD) operator (a learned transfer operator); the deep
conv-Koopman autoencoder reproduces the invariant measure but not per-mode
resonances, so it is not the spectral instrument (see exp1).

Validation: direct ETDRK4 simulation + EDMD at the target sizes (the ground truth
the flow never touches for fitting).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (RUNS, SEEDS, SIZES_TRAIN, L_HOLDOUT, L_TARGET,             # noqa: E402
                    analyze_measurement, measure_size, save_json)
from floweval import (fit_edmd_flows, predict_edmd_flow,                        # noqa: E402
                      truth_from_measurement, score, score_new_band)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sim", action="store_true")
    args = ap.parse_args()

    p1408 = RUNS / "measure_L1408.npz"
    if not (args.skip_sim and p1408.exists()):
        p1408 = measure_size(L_TARGET, tag="L1408")
    meas = {L_HOLDOUT: analyze_measurement(RUNS / "measure_L176.npz"),
            L_TARGET: analyze_measurement(p1408)}
    curves = {L: analyze_measurement(RUNS / f"measure_L{L:g}.npz")
              for L in SIZES_TRAIN}

    results = {"targets": {}, "config": {"sizes_train": SIZES_TRAIN, "seeds": SEEDS,
                                         "method": "edmd_fitted_flow"}}
    for L_t in (L_HOLDOUT, L_TARGET):
        truth = truth_from_measurement(meas[L_t])
        k_t = truth["k"]
        # headline = seed-mean flow (GATE-S recipe); per-seed flows give the spread
        pred_mean = predict_edmd_flow(fit_edmd_flows(curves, None), k_t, L_t)
        preds = [predict_edmd_flow(fit_edmd_flows(curves, s), k_t, L_t)
                 for s in SEEDS]
        # point_estimate = the reported headline (seed-mean/GATE-S flow, a single
        # prediction, NOT a median over seeds); per_seed gives the spread only.
        point = score(pred_mean, truth)
        per_seed = [score(p, truth) for p in preds]
        newb = score_new_band(pred_mean, truth)
        entry = {"L": L_t, "n_sectors": len(k_t), "k": k_t.tolist(),
                 "truth_gamma": truth["gamma"].tolist(),
                 "truth_omega": truth["omega"].tolist(),
                 "truth_s": truth["s_density"].tolist(),
                 "truth_gamma_se": truth["gamma_se"].tolist(),
                 "estimator": "seed_mean_flow_point_estimate",
                 "methods": {"fitted_flow": {
                     "point_estimate": point,
                     "per_seed": per_seed,
                     "new_band": newb,
                     "pred_gamma_mean": pred_mean["gamma"].tolist(),
                     "pred_omega_mean": pred_mean["omega"].tolist(),
                     "pred_s_mean": pred_mean["s_density"].tolist()}}}
        m = point
        print(f"L={L_t:g} fitted_flow: " + ", ".join(
            f"{q}={m[q]:.4g}" for q in ("gamma_med_rel", "s_med_log10", "c_rel_l2",
                                        "tau_med_rel", "slow_overlap")))
        if newb and newb["new_mode"]:
            nb = newb["new_mode"]
            print(f"        new-mode band (k<2pi/88): gamma {nb['gamma_med_rel']:.3g}, "
                  f"S {nb['s_med_log10']:.3g} over {nb['n_modes']} modes")
        results["targets"][f"{L_t:g}"] = entry
    save_json("exp2_flow.json", results)


if __name__ == "__main__":
    main()
