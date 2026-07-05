"""A-E1: synthetic demonstration (Paper A, Figure 1 + headline table).

Controlled generator (heckesel.synth): sampling probability depends on an
unobservable correlated (rho swept 0 -> 0.9) with the outcome noise.

Shows, per info.txt:
(i)   deep ensembles / MC-dropout / GP baselines are miscalibrated in
      selected-against regions;
(ii)  importance weighting with ORACLE propensities does not fix it;
(iii) the Heckman-corrected predictive variance does, when an instrument
      exists (alpha = 1);
(iv)  degradation without the instrument (alpha = 0): honesty curve vs rho.

KILL CONDITION (pre-registered): if oracle-IW matches Heckman under
selection on unobservables, the premise is wrong -> report and stop. The
script prints an explicit verdict block.

Run: python experiments/expA_e1.py [--smoke]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_json, Timer

from heckesel.synth import make_selection_data, f0_smooth
from heckesel.metrics import evaluate_predictive
from heckesel.deep import HeckmanEnsemble, HeckmanTwoStepEnsemble
from heckesel.uq import (DeepEnsembleUQ, IWDeepEnsembleUQ, MCDropoutUQ, GPUQ,
                         BlindTwoHeadUQ)

RHOS = [0.0, 0.3, 0.6, 0.9]
ALPHAS = [0.0, 1.0]
N_SEEDS = 8
N_POOL = 2000
N_TEST = 3000
SIGMA = 0.5


def run_cell(rho: float, alpha: float, seed: int, epochs: int, k: int):
    data = make_selection_data(N_POOL, rho, alpha=alpha, sigma=SIGMA, d=1,
                               seed=seed)
    test = make_selection_data(N_TEST, rho, alpha=alpha, sigma=SIGMA, d=1,
                               seed=10_000 + seed)
    sel = data.s > 0.5
    x_sel, y_sel = data.x[sel], data.y[sel]
    w_all = np.column_stack([data.x, data.z])

    fits = {}
    fits["deep_ensemble"] = DeepEnsembleUQ.fit(x_sel, y_sel, k=k, seed=seed,
                                               epochs=epochs)
    fits["mc_dropout"] = MCDropoutUQ.fit(x_sel, y_sel, seed=seed,
                                         epochs=epochs)
    fits["gp"] = GPUQ.fit(x_sel, y_sel, seed=seed)
    fits["iw_oracle_ensemble"] = IWDeepEnsembleUQ.fit_iw(
        x_sel, y_sel, data.propensity[sel], k=k, seed=seed, epochs=epochs)
    fits["blind_two_head"] = BlindTwoHeadUQ.fit(
        data.x, w_all, data.y, data.s, k=k, seed=seed, epochs=epochs)
    fits["heckman_ens"] = HeckmanEnsemble.fit(
        data.x, w_all, data.y, data.s, k=k, seed=seed, epochs=epochs)
    fits["heckman_2s_ens"] = HeckmanTwoStepEnsemble.fit(
        data.x, w_all, data.y, data.s, k=k, seed=seed, epochs=epochs)

    rows = []
    for name, model in fits.items():
        mu, var = model.predict(test.x)
        m = evaluate_predictive(test.y_full, mu, var, test.prop_x,
                                f0x=test.f0x)
        row = {"rho": rho, "alpha": alpha, "seed": seed, "method": name, **m}
        if name == "heckman_ens":
            row["rho_hat"] = model.rho
            row["sigma_hat"] = model.sigma
        rows.append(row)
    # oracle reference: the true predictive N(f0(x), sigma^2)
    mu0, var0 = test.f0x, np.full(N_TEST, SIGMA**2)
    rows.append({"rho": rho, "alpha": alpha, "seed": seed, "method": "oracle",
                 **evaluate_predictive(test.y_full, mu0, var0, test.prop_x,
                                       f0x=test.f0x)})
    rows.append({"rho": rho, "alpha": alpha, "seed": seed, "method": "_meta",
                 "selected_frac": data.meta["selected_frac"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    rhos, alphas, seeds = RHOS, ALPHAS, range(N_SEEDS)
    if args.smoke:
        rhos, alphas, seeds = [0.8], [1.0], range(1)
        args.epochs = 800

    all_rows = []
    for rho in rhos:
        for alpha in alphas:
            for seed in seeds:
                with Timer(f"rho={rho} alpha={alpha} seed={seed}"):
                    all_rows += run_cell(rho, alpha, seed, args.epochs,
                                         args.k)
    name = "expA_e1_smoke.json" if args.smoke else "expA_e1.json"
    save_json(name, all_rows)

    # ---- pre-registered kill-condition verdict --------------------------
    hi_rho = max(rhos)
    a_on = max(alphas)

    def agg(method, key):
        vals = [r[key] for r in all_rows
                if r["method"] == method and r["rho"] == hi_rho
                and r["alpha"] == a_on and key in r]
        return float(np.mean(vals)) if vals else float("nan")

    iw_cov = agg("iw_oracle_ensemble", "picp90_against")
    hk_cov = agg("heckman_ens", "picp90_against")
    iw_bias = agg("iw_oracle_ensemble", "bias_f0_against")
    hk_bias = agg("heckman_ens", "bias_f0_against")
    print("\n===== KILL-CONDITION CHECK (rho=%.1f, alpha=%.0f) =====" %
          (hi_rho, a_on))
    print(f"selected-against coverage@90: IW-oracle {iw_cov:.3f} "
          f"vs Heckman {hk_cov:.3f} (nominal 0.90)")
    print(f"selected-against bias vs f0:  IW-oracle {iw_bias:+.3f} "
          f"vs Heckman {hk_bias:+.3f}")
    if abs(iw_cov - 0.90) <= abs(hk_cov - 0.90) + 0.01 and \
       abs(iw_bias) <= abs(hk_bias) + 0.01:
        print("VERDICT: KILL CONDITION FIRES -- oracle IW matches Heckman. "
              "Premise wrong; report and stop.")
    else:
        print("VERDICT: premise holds -- oracle IW does not match Heckman "
              "under selection on unobservables.")


if __name__ == "__main__":
    main()
