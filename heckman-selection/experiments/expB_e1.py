"""B-E1: corrected surrogate -- joint selection+outcome model (Paper B).

Three questions, per info.txt:
1. POPULATION RECOVERY: can the unselected population over latent curve
   parameters be recovered from survivor-only data? Estimators:
   - eb_surv_naive: EB prior fit on survivors, no correction (common
     practice: extrapolators trained on completed runs);
   - eb_surv_heck_1b: + survival-likelihood correction, ONE bracket
     (weakly identified -- the no-instrument analogue, kept for honesty);
   - eb_surv_heck_mb: + survival terms from THREE brackets with different
     rung schedules (rung-assignment randomness = the exclusion
     restriction);
   - eb_all: prior from all curves incl. killed prefixes (MAR-correct
     reference -- the information a replaying tool actually has).
2. SURVIVOR EXTRAPOLATION (rung-level): predict survivors' final values
   from rung prefixes. Methods: naive per-curve LS, Tobit-corrected
   per-curve fit (ablation), EB posteriors under each prior, and the
   Arellano-Bond-flavoured lag-IV increment extrapolator (secondary).
3. FRESH-CONFIG TRANSFER: predict finals of NEW (unselected) configs from
   short prefixes using priors learned from survivor-only data.

Run: python experiments/expB_e1.py [--smoke]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_json, Timer

from heckesel.lc import (POP_MU, default_pop_cov, curve_values, run_sh,
                         fit_pow3_ls, fit_pow3_tobit, fit_pow3_with_cov,
                         phi_final, EBModel)

T = 52
T_FULL = np.arange(1, T + 1, dtype=float)
BRACKETS = [[4, 12, 36], [12, 36], [36]]
ETA = 3.0
N_PER_BRACKET = 999
SIGMAS = [0.005, 0.01, 0.02]
N_SEEDS = 8


def ab_lag_iv_extrapolate(y_prefix: np.ndarray, T_target: int) -> float:
    """Arellano-Bond-flavoured extrapolator: pooled-per-curve AR(1) on
    increments with the second lag as instrument (2SLS slope), then
    geometric extrapolation of the last increment."""
    d = np.diff(y_prefix)
    if len(d) < 3:
        return float(y_prefix[-1])
    y_t, y_l, z = d[2:], d[1:-1], d[:-2]
    denom = float(z @ y_l)
    gamma = float(z @ y_t) / denom if abs(denom) > 1e-12 else 0.0
    gamma = float(np.clip(gamma, -0.95, 0.95))
    steps = T_target - len(y_prefix)
    inc = d[-1]
    total = inc * gamma * (1 - gamma**steps) / (1 - gamma) \
        if abs(gamma) > 1e-9 else 0.0
    return float(y_prefix[-1] + total)


def make_bracket_data(sigma: float, seed: int, n: int):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(default_pop_cov())
    out = []
    for b, rungs in enumerate(BRACKETS):
        phi = POP_MU + rng.standard_normal((n, 3)) @ L.T
        y_true = curve_values(phi, T_FULL)
        y_obs = y_true + sigma * rng.standard_normal(y_true.shape)
        sh = run_sh(y_obs, rungs, ETA)
        out.append(dict(phi=phi, y_true=y_true, y_obs=y_obs, sh=sh,
                        rungs=rungs))
    return out


def fit_survivor_stats(bracket, sigma):
    """LS fit + Laplace cov on each end-of-run survivor's full curve."""
    surv = np.where(bracket["sh"].alive[:, -1])[0]
    fits = [fit_pow3_with_cov(T_FULL, bracket["y_obs"][i], sigma)
            for i in surv]
    return (surv, np.array([f[0] for f in fits]),
            np.array([f[1] for f in fits]))


def run_cell(sigma: float, seed: int, n: int, quick: bool = False):
    data = make_bracket_data(sigma, seed, n)
    row = {"sigma": sigma, "seed": seed, "n_per_bracket": n}

    # ---------- population recovery -----------------------------------
    groups, ph_parts, V_parts = [], [], []
    for b in data:
        surv, ph, V = fit_survivor_stats(b, sigma)
        groups.append((np.array(b["rungs"], float),
                       np.array(b["sh"].thresholds), len(ph)))
        ph_parts.append(ph)
        V_parts.append(V)
    ph_surv = np.vstack(ph_parts)
    V_surv = np.vstack(V_parts)

    models = {}
    models["eb_surv_naive"] = EBModel.fit(ph_surv, V_surv,
                                          mode="survivor_naive",
                                          sigma_obs=sigma, seed=seed)
    models["eb_surv_heck_mb"] = EBModel.fit(ph_surv, V_surv,
                                            mode="survivor_heckman",
                                            survival_groups=groups,
                                            sigma_obs=sigma, seed=seed)
    models["eb_surv_heck_1b"] = EBModel.fit(
        ph_parts[0], V_parts[0], mode="survivor_heckman",
        survival_groups=[groups[0]], sigma_obs=sigma, seed=seed)

    # MAR-correct reference: all curves of bracket 0 at their observed
    # prefixes (killed curves contribute their short prefixes)
    b0 = data[0]
    ph_all, V_all = [], []
    for i in range(n):
        kr = b0["sh"].kill_rung[i]
        upto = T if kr < 0 else b0["rungs"][kr]
        t = np.arange(1, upto + 1, dtype=float)
        ph_i, V_i = fit_pow3_with_cov(t, b0["y_obs"][i, :upto], sigma)
        ph_all.append(ph_i)
        V_all.append(V_i)
    ph_all, V_all = np.array(ph_all), np.array(V_all)
    models["eb_all"] = EBModel.fit(ph_all, V_all, mode="all",
                                   sigma_obs=sigma, seed=seed)

    row["pop"] = {"naive_mu_a": float(ph_surv[:, 0].mean()),
                  "true_mu_a": float(POP_MU[0]),
                  "true_sd_a": float(np.sqrt(default_pop_cov()[0, 0])),
                  "n_surv": int(len(ph_surv))}
    for name, m in models.items():
        row["pop"][f"{name}_mu_a"] = float(m.mu[0])
        row["pop"][f"{name}_sd_a"] = float(np.sqrt(m.cov[0, 0]))

    # ---------- survivor extrapolation at each rung (bracket 0) --------
    row["extrap"] = {}
    for k, t_k in enumerate(b0["rungs"]):
        surv_k = np.where(b0["sh"].alive[:, k + 1])[0]
        if quick:
            surv_k = surv_k[:80]
        t = np.arange(1, t_k + 1, dtype=float)
        re = np.array(b0["rungs"][:k + 1], float)
        rt = np.array(b0["sh"].thresholds[:k + 1])
        truth = b0["y_true"][surv_k, -1]

        preds = {"naive_ls": [], "tobit": [], "ab_lag_iv": []}
        ph_k, V_k = [], []
        for i in surv_k:
            yp = b0["y_obs"][i, :t_k]
            ph_i, V_i = fit_pow3_with_cov(t, yp, sigma)
            ph_k.append(ph_i)
            V_k.append(V_i)
            preds["naive_ls"].append(phi_final(ph_i, T)[0])
            preds["tobit"].append(phi_final(
                fit_pow3_tobit(t, yp, re, rt, sigma), T)[0])
            preds["ab_lag_iv"].append(ab_lag_iv_extrapolate(yp, T))
        ph_k, V_k = np.array(ph_k), np.array(V_k)
        for name, m in models.items():
            preds[name] = m.predict_final_exact(
                b0["y_obs"][surv_k], t_k, ph_k, V_k, T, seed=seed)

        rr = {}
        for name, p in preds.items():
            p = np.asarray(p, dtype=float)
            rr[name] = {
                "bias": float(np.mean(p - truth)),
                "rmse": float(np.sqrt(np.mean((p - truth)**2))),
                "spearman": float(stats.spearmanr(p, truth).statistic)
                if len(truth) > 2 else float("nan"),
            }
        rr["n"] = int(len(surv_k))
        row["extrap"][f"rung{k}"] = rr

    # ---------- fresh-config transfer (prefix len = first rung epoch) --
    rng = np.random.default_rng(90_000 + seed)
    L = np.linalg.cholesky(default_pop_cov())
    phi_f = POP_MU + rng.standard_normal((300, 3)) @ L.T
    y_true_f = curve_values(phi_f, T_FULL)
    y_obs_f = y_true_f + sigma * rng.standard_normal(y_true_f.shape)
    t_pref = 12
    t = np.arange(1, t_pref + 1, dtype=float)
    ph_f, V_f = [], []
    naive_f = []
    for i in range(len(phi_f)):
        ph_i, V_i = fit_pow3_with_cov(t, y_obs_f[i, :t_pref], sigma)
        ph_f.append(ph_i)
        V_f.append(V_i)
        naive_f.append(phi_final(ph_i, T)[0])
    ph_f, V_f = np.array(ph_f), np.array(V_f)
    truth_f = y_true_f[:, -1]
    row["fresh"] = {"naive_ls": {
        "bias": float(np.mean(np.array(naive_f) - truth_f)),
        "rmse": float(np.sqrt(np.mean((np.array(naive_f) - truth_f)**2))),
        "spearman": float(stats.spearmanr(naive_f, truth_f).statistic)}}
    for name, m in models.items():
        p = m.predict_final_exact(y_obs_f, t_pref, ph_f, V_f, T, seed=seed)
        row["fresh"][name] = {
            "bias": float(np.mean(p - truth_f)),
            "rmse": float(np.sqrt(np.mean((p - truth_f)**2))),
            "spearman": float(stats.spearmanr(p, truth_f).statistic)}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    sigmas = [0.01] if args.smoke else SIGMAS
    seeds = range(1 if args.smoke else N_SEEDS)
    n = 500 if args.smoke else N_PER_BRACKET

    rows = []
    for sigma in sigmas:
        for seed in seeds:
            with Timer(f"sigma={sigma} seed={seed}"):
                rows.append(run_cell(sigma, seed, n, quick=args.smoke))
    save_json("expB_e1_smoke.json" if args.smoke else "expB_e1.json",
              {"rows": rows, "brackets": BRACKETS, "eta": ETA})

    # quick readout
    print("\n===== B-E1 population recovery (mu_a; truth "
          f"{POP_MU[0]:.3f}) =====")
    for key in ["naive_mu_a", "eb_surv_naive_mu_a", "eb_surv_heck_1b_mu_a",
                "eb_surv_heck_mb_mu_a", "eb_all_mu_a"]:
        v = [r["pop"][key] for r in rows]
        print(f"{key:24s} {np.mean(v):.4f} +- {np.std(v):.4f}")
    print("\n===== survivor extrapolation bias at rung0 =====")
    for name in ["naive_ls", "tobit", "ab_lag_iv", "eb_surv_naive",
                 "eb_surv_heck_mb", "eb_all"]:
        v = [r["extrap"]["rung0"][name]["bias"] for r in rows]
        print(f"{name:18s} bias={np.mean(v):+.4f} +- {np.std(v):.4f}")


if __name__ == "__main__":
    main()
