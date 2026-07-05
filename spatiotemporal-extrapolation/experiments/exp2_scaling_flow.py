"""E2 — THE CLAIM: extrapolate learned/fitted operator spectra in domain size.

Fits the finite-size-scaling flow on L in {22,44,66,88} and predicts the
per-sector leading resonances and spectral density at L = 176 (holdout) and
L = 1408 (64x the base size), with ZERO large-L data. Two flow fitters:
  (a) fitted FSS flow: SmoothFlow splines in k, linear in ell = 22/L, on EDMD
      curves (transparent numerics);
  (b) learned operator FSS flow: size-conditioned conv Koopman propagator
      W(ell) = W0 + ell W1 trained jointly on the four small sizes; evaluated
      analytically on the refined kappa-grid at target ell.
Validation: direct ETDRK4 simulation + EDMD at the target sizes (the ground
truth our route never touches for fitting).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (DATA, RUNS, SEEDS, SIZES_TRAIN, L_HOLDOUT, L_TARGET,        # noqa: E402
                    analyze_measurement, measure_size, n_of, save_json)
from floweval import (fit_edmd_flows, predict_edmd_flow,                        # noqa: E402
                      truth_from_measurement, score, score_new_band)
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from specext.edmd import ModeGrid                                               # noqa: E402
from specext.koopman import ConvKoopman, train_model, model_spectrum, L_BASE    # noqa: E402

MODELS = RUNS / "models"
MODELS.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def train_flow_model(seed, steps=24000, force=False):
    path = MODELS / f"flow_s{seed}.pt"
    if path.exists() and not force:
        return path
    datasets = [{"L": L, "data": np.load(DATA / f"L{L:g}_s{seed}.npy", mmap_mode="r")}
                for L in SIZES_TRAIN]
    model, wall = train_model(datasets, seed=seed, flow=True, steps=steps,
                              device=DEV, log_fn=lambda s: print(f"[flow s{seed}]{s}"))
    torch.save({"state": model.state_dict(), "flow": True, "seed": seed,
                "wall_s": wall, "steps": steps, "sizes": SIZES_TRAIN}, path)
    print(f"trained flow model seed={seed} in {wall:.0f}s")
    return path


def neural_prediction(model_path, L_t):
    ckpt = torch.load(model_path, map_location=DEV, weights_only=False)
    model = ConvKoopman(flow=ckpt["flow"]).to(DEV)
    model.load_state_dict(ckpt["state"])
    model.eval()
    N_t = n_of(L_t)
    grid = ModeGrid(L_t, N_t)
    spec = model_spectrum(model, ell=L_BASE / L_t, m_idx=grid.m_idx, N=N_t,
                          device=DEV)
    lam = spec["lam"]
    return {"gamma": -lam.real, "omega": np.abs(lam.imag), "k": grid.k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--steps", type=int, default=24000)
    args = ap.parse_args()

    # ground truth at the target sizes (validation only)
    p1408 = RUNS / "measure_L1408.npz"
    if not (args.skip_sim and p1408.exists()):
        p1408 = measure_size(L_TARGET, tag="L1408")
    meas = {L_HOLDOUT: analyze_measurement(RUNS / "measure_L176.npz"),
            L_TARGET: analyze_measurement(p1408)}

    curves = {L: analyze_measurement(RUNS / f"measure_L{L:g}.npz")
              for L in SIZES_TRAIN}

    for seed in SEEDS:
        train_flow_model(seed, steps=args.steps)

    results = {"targets": {}, "config": {"sizes_train": SIZES_TRAIN,
                                         "steps": args.steps, "seeds": SEEDS}}
    for L_t in (L_HOLDOUT, L_TARGET):
        truth = truth_from_measurement(meas[L_t])
        k_t = truth["k"]
        entry = {"L": L_t, "n_sectors": len(k_t),
                 "truth_gamma": truth["gamma"].tolist(),
                 "truth_omega": truth["omega"].tolist(),
                 "truth_s": truth["s_density"].tolist(),
                 "truth_gamma_se": truth["gamma_se"].tolist(),
                 "k": k_t.tolist(), "methods": {}}
        # (a) fitted FSS flow, per seed
        preds_a = []
        for seed in SEEDS:
            flows = fit_edmd_flows(curves, seed)
            preds_a.append(predict_edmd_flow(flows, k_t, L_t))
        # (b) learned flow: neural dispersion + fitted density (statics shared)
        preds_b = []
        for seed in SEEDS:
            nb = neural_prediction(MODELS / f"flow_s{seed}.pt", L_t)
            pb = {"gamma": nb["gamma"], "omega": nb["omega"],
                  "s_density": preds_a[seed]["s_density"]}
            preds_b.append(pb)
        for name, preds in (("fitted_flow", preds_a), ("learned_flow", preds_b)):
            scores = [score(p, truth) for p in preds]
            newb = [score_new_band(p, truth) for p in preds]
            entry["methods"][name] = {
                "per_seed": scores,
                "median": {m: float(np.nanmedian([s[m] for s in scores]))
                           for m in scores[0]},
                "new_band": (None if newb[0] is None else
                             {m: float(np.nanmedian([s[m] for s in newb]))
                              for m in newb[0]}),
                "pred_gamma_mean": np.nanmean([p["gamma"] for p in preds],
                                              axis=0).tolist(),
                "pred_omega_mean": np.nanmean([p["omega"] for p in preds],
                                              axis=0).tolist(),
                "pred_s_mean": np.nanmean([p["s_density"] for p in preds],
                                          axis=0).tolist(),
            }
            print(f"L={L_t:g} {name}: " + ", ".join(
                f"{m}={entry['methods'][name]['median'][m]:.4g}"
                for m in ("gamma_med_rel", "s_med_log10", "c_rel_l2",
                          "tau_med_rel", "slow_overlap_top16")))
        results["targets"][f"{L_t:g}"] = entry
    save_json("exp2_flow.json", results)


if __name__ == "__main__":
    main()
