"""B-E0: synthetic survivor-bias demonstration (Paper B, Figure 1 + GATE).

Power-law curve family with per-config latent parameters + observation
noise; SH censoring; naive per-curve LC surrogate fitted on survivor
prefixes; fitted-parameter and predicted-final bias vs rung, noise level,
and selection pressure eta.

GATE (pre-registered): the bias exists and grows with noise / selection
pressure. If negligible at realistic noise calibrated from LCBench
residuals, that IS the paper's answer (informative null about SH
robustness) -- the script prints the verdict either way.

Design notes:
- The no-selection CONTROL fits the same estimator on the same-length
  prefixes of ALL configs; the survivor-selection effect is the difference
  (isolates selection from generic extrapolation bias of noisy fits).
- Per-curve fits are computed once per (sigma, seed, rung) on all configs
  and reused across eta (the censoring only changes the survivor masks).
- The bias concentrates at EARLY rungs (fresh selection + long
  extrapolation); by late rungs long prefixes pin the fit down. The gate is
  evaluated at the first rung, and per-rung decay is reported as part of
  the finding.

Run: python experiments/expB_e0.py [--smoke]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_json, Timer, ROOT

from heckesel.lc import (POP_MU, default_pop_cov, curve_values, run_sh,
                         fit_pow3_ls, phi_final)

T = 52
RUNGS = [4, 12, 36]
ETAS = [2.0, 3.0, 4.0]
ETA_MAIN = 3.0
N_CONFIGS = 1000
SIGMAS = [0.005, 0.01, 0.02, 0.04]
N_SEEDS = 8


def calibrated_sigma() -> dict | None:
    """LCBench-calibrated noise levels (written by expB_calib.py):
    'sigma_median' = robust lag-1-diff estimate (lower bound; fixed val set
    makes much of the measurement error persistent), 'sigma_pow3_median' =
    residual sd around per-curve pow3 fits (per-epoch fluctuation scale)."""
    cache = ROOT / "data" / "lcbench_calib.json"
    if cache.exists():
        import json
        d = json.loads(cache.read_text())
        return {"sigma_median": d["sigma_median"],
                "sigma_pow3_median": d["sigma_pow3_median"]}
    return None


def sample_population(n: int, seed: int):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(default_pop_cov())
    phi = POP_MU + rng.standard_normal((n, 3)) @ L.T
    y_true = curve_values(phi, np.arange(1, T + 1, dtype=float))
    return phi, y_true


def run_cell(sigma: float, seed: int, n_configs: int, etas):
    phi, y_true = sample_population(n_configs, seed)
    rng = np.random.default_rng(1000 + seed)
    y_obs = y_true + sigma * rng.standard_normal(y_true.shape)
    final_true = y_true[:, -1]

    # naive per-curve fits per rung, on ALL configs (reused across eta)
    preds, fits_a = {}, {}
    for k, t_k in enumerate(RUNGS):
        t = np.arange(1, t_k + 1, dtype=float)
        fitted = np.array([fit_pow3_ls(t, y_obs[i, :t_k])
                           for i in range(n_configs)])
        preds[k] = phi_final(fitted, T)
        fits_a[k] = fitted[:, 0]

    rows = []
    for eta in etas:
        sh = run_sh(y_obs, RUNGS, eta)
        out = {"sigma": sigma, "seed": seed, "eta": eta, "rungs": RUNGS,
               "n_configs": n_configs}
        for k, t_k in enumerate(RUNGS):
            surv = sh.alive[:, k + 1]
            last_val = y_obs[:, t_k - 1]
            eps_rung = y_obs[:, t_k - 1] - y_true[:, t_k - 1]
            r = {}
            for name, mask in (("surv", surv),
                               ("all", np.ones(n_configs, dtype=bool))):
                d = preds[k][mask] - final_true[mask]
                r[f"pred_bias_{name}"] = float(np.mean(d))
                r[f"pred_bias_{name}_sd"] = float(np.std(d))
                r[f"a_bias_{name}"] = float(np.mean(fits_a[k][mask]
                                                    - phi[mask, 0]))
                r[f"last_bias_{name}"] = float(np.mean(last_val[mask]
                                                       - final_true[mask]))
                r[f"eps_rung_{name}"] = float(np.mean(eps_rung[mask]))
                r[f"n_{name}"] = int(mask.sum())
            r["selection_bias"] = r["pred_bias_surv"] - r["pred_bias_all"]
            out[f"rung{k}"] = r
        rows.append(out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    sigmas = list(SIGMAS)
    cal = calibrated_sigma()
    if cal is not None:
        for v in cal.values():
            v = round(v, 4)
            if v > 0 and all(abs(v - s) > 5e-4 for s in sigmas):
                sigmas.append(v)
        sigmas.sort()
    seeds = range(2 if args.smoke else N_SEEDS)
    n_configs = 300 if args.smoke else N_CONFIGS

    rows = []
    for sigma in sigmas:
        for seed in seeds:
            with Timer(f"sigma={sigma} seed={seed}"):
                rows += run_cell(sigma, seed, n_configs, ETAS)
    payload = {"rows": rows, "calibrated_sigma": cal, "etas": ETAS,
               "eta_main": ETA_MAIN, "T": T, "rungs": RUNGS}
    save_json("expB_e0_smoke.json" if args.smoke else "expB_e0.json",
              payload)

    # ---- GATE verdict (rung 0 = first selection, longest extrapolation) --
    print("\n===== B-E0 GATE (rung 0, eta=3) =====")
    def sel_bias(sigma, eta, rung="rung0"):
        v = [r[rung]["selection_bias"] for r in rows
             if r["sigma"] == sigma and r["eta"] == eta]
        return float(np.mean(v)), float(np.std(v))

    for sigma in sigmas:
        m, sd = sel_bias(sigma, ETA_MAIN)
        print(f"sigma={sigma:<7} selection_bias={m:+.4f}+-{sd:.4f}")
    print("--- eta sweep at sigma=0.02 ---")
    for eta in ETAS:
        m, sd = sel_bias(0.02, eta)
        print(f"eta={eta}: selection_bias={m:+.4f}+-{sd:.4f}")

    lo, _ = sel_bias(sigmas[0], ETA_MAIN)
    hi, _ = sel_bias(max(sigmas), ETA_MAIN)
    e_lo, _ = sel_bias(0.02, min(ETAS))
    e_hi, _ = sel_bias(0.02, max(ETAS))
    ok = hi > 0 and hi > lo and e_hi > e_lo > 0
    print("VERDICT:", "GATE PASSES -- bias positive, grows with noise and "
          "selection pressure" if ok else
          "GATE WEAK/FAILS -- inspect; possible informative null")


if __name__ == "__main__":
    main()
