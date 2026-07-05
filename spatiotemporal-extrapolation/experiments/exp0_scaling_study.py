"""E0 — ground-truth scaling study (pure numerics, no learning): KS at
L = 22/44/66/88 (train) and 176 (holdout). Estimates per-sector leading
Ruelle-Pollicott/Koopman resonances and spectral densities, fits the finite-size
dependence on the common k-grid, and decides GATE S (PLAN.md, pre-registered).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (RUNS, SEEDS, SIZES_TRAIN, L_HOLDOUT, analyze_measurement,   # noqa: E402
                    common_grid_indices, measure_size, save_json)
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from specext.scaling import pointwise_fit, pointwise_predict                    # noqa: E402


def run_measurements(skip_sim):
    paths = {}
    for L in SIZES_TRAIN + [L_HOLDOUT]:
        tag = f"L{L:g}"
        p = RUNS / f"measure_{tag}.npz"
        if not (skip_sim and p.exists()):
            p = measure_size(L, store_fields=True, tag=tag)
        paths[L] = p
    return paths


def gate_s(curves):
    """Pre-registered GATE S: S1 (smooth-or-converged per k) and S2 (holdout)."""
    sizes = np.asarray(SIZES_TRAIN)
    out = {"per_quantity": {}, "s1_pass": None, "s2_pass": None}
    for qty in ("gamma", "s_density"):
        kc_idx = {}
        for L in SIZES_TRAIN + [L_HOLDOUT]:
            kc_idx[L], kc = common_grid_indices(curves[L]["k"])
        rows = []
        for j, kk in enumerate(kc):
            y = np.array([curves[L][qty][:, kc_idx[L][j]].mean() for L in SIZES_TRAIN])
            se = np.array([curves[L][qty][:, kc_idx[L][j]].std(ddof=1) /
                           np.sqrt(curves[L][qty].shape[0]) for L in SIZES_TRAIN])
            f1 = pointwise_fit(sizes, y, se=se, power=1)
            f2 = pointwise_fit(sizes, y, se=se, power=2)
            best = f1 if f1["aicc"] <= f2["aicc"] else f2
            tv = float(y.max() - y.min())
            converged = tv <= 2.0 * float(se.mean())
            s1_k = (f1["r2"] >= 0.7) or converged
            y176 = float(curves[L_HOLDOUT][qty][:, kc_idx[L_HOLDOUT][j]].mean())
            se176 = float(curves[L_HOLDOUT][qty][:, kc_idx[L_HOLDOUT][j]].std(ddof=1)
                          / np.sqrt(curves[L_HOLDOUT][qty].shape[0]))
            pred176 = float(pointwise_predict(best, L_HOLDOUT))
            rows.append({"k": float(kk), "y": y, "se": se, "fit1": f1, "fit2": f2,
                         "best_power": best["power"], "converged": converged,
                         "s1": bool(s1_k), "holdout_true": y176,
                         "holdout_se": se176, "holdout_pred": pred176,
                         "holdout_rel_err": abs(pred176 - y176) / abs(y176)})
        n_s1 = sum(r["s1"] for r in rows)
        med_err = float(np.median([r["holdout_rel_err"] for r in rows]))
        out["per_quantity"][qty] = {
            "rows": rows, "n_s1_pass": n_s1, "n_k": len(rows),
            "holdout_median_rel_err": med_err,
            "s1": n_s1 >= 5, "s2": med_err <= 0.10,
            "aicc_prefers_power2": int(sum(r["best_power"] == 2 for r in rows)),
        }
    out["s1_pass"] = all(out["per_quantity"][q]["s1"] for q in ("gamma", "s_density"))
    out["s2_pass"] = all(out["per_quantity"][q]["s2"] for q in ("gamma", "s_density"))
    out["gate_s_pass"] = out["s1_pass"] and out["s2_pass"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sim", action="store_true",
                    help="reuse existing runs/measure_*.npz")
    args = ap.parse_args()
    paths = run_measurements(args.skip_sim)

    curves = {}
    for L, p in paths.items():
        curves[L] = analyze_measurement(p)
        c = curves[L]
        print(f"L={L:g}: {len(c['k'])} sectors, med R2(EDMD acf) "
              f"{np.nanmedian(c['r2']):.3f}, strides used "
              f"{sorted(set(c['stride'].tolist()))}, wall {c['wall_s']:.0f}s")
    np.savez_compressed(RUNS / "curves_train.npz",
                        **{f"{key}_L{L:g}": curves[L][key]
                           for L in curves
                           for key in ("k", "gamma", "omega", "s_density", "tau_acf",
                                       "omega_pk", "r2", "stride", "p_mean")})

    gate = gate_s(curves)
    summary = {
        "sizes_train": SIZES_TRAIN, "L_holdout": L_HOLDOUT, "seeds": SEEDS,
        "per_size": {
            f"{L:g}": {
                "n_sectors": len(curves[L]["k"]),
                "median_r2_edmd_acf": float(np.nanmedian(curves[L]["r2"])),
                "frac_r2_above_0.8": float(np.nanmean(curves[L]["r2"] > 0.8)),
                "slow_density_per_L": float((curves[L]["gamma"].mean(axis=0) <= 0.1).sum() / L),
                "wall_s": curves[L]["wall_s"],
            } for L in curves},
        "gate_s": gate,
    }
    save_json("exp0_scaling.json", summary)
    verdict = "PASS" if gate["gate_s_pass"] else "FAIL -> K1 characterization paper"
    print(f"\nGATE S: {verdict}")
    print(f"  S1 smooth-or-converged: gamma {gate['per_quantity']['gamma']['n_s1_pass']}/7, "
          f"S {gate['per_quantity']['s_density']['n_s1_pass']}/7")
    print(f"  S2 holdout med rel err: gamma "
          f"{gate['per_quantity']['gamma']['holdout_median_rel_err']:.3%}, "
          f"S {gate['per_quantity']['s_density']['holdout_median_rel_err']:.3%}")


if __name__ == "__main__":
    main()
