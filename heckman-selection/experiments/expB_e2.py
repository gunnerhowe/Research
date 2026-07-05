"""B-E2/E3/E4: LCBench replay -- naive vs corrected surrogate driving
promotion and final-pick decisions (Paper B).

Protocol (synchronous SH replay on precomputed curves; no training):
- pool of n_pool random configs per (dataset, seed);
- rungs [4, 12, 36] epochs, eta=3, T=52;
- at each rung the surrogate predicts finals from prefixes; top 1/eta of
  alive configs are promoted (last_value = standard SH);
- two final-pick rules:
  'completed'  best OBSERVED final among configs run to completion
               (decisions differ only through promotions);
  'allpick'    best PREDICTED final across ALL configs incl. early-killed
               ones (exposes the winner's curse of naive extrapolation
               from 4-epoch prefixes; the B-E4 null control lives here).

Surrogates (B-E3 baselines faithful, per info.txt):
- last_value          standard SH promotion stat;
- naive_pow3          per-curve LS pow3 extrapolation (DPL-style
                      single-curve power-law fit);
- naive_pow3_vpen     B-E4 null control: naive_pow3 minus kappa * sd where
                      sd is the Laplace delta-method extrapolation sd --
                      the one-parameter variance-inflation hack;
- eb_replay           EB prior fit on the fly from ALL prefixes observed
                      so far in this replay (killed + alive), exact-
                      likelihood posterior predictions (the deployable
                      corrected surrogate; MAR-correct use of the data the
                      tool has anyway).
FT-PFN is cite-compared in the paper (house rule: released weights vs
local run economics); freeze-thaw GP omitted at replay scale and discussed.

Metrics: regret vs budget (epochs), Spearman/RMSE of final-value
predictions at rung 0 across the full pool (held-out completions),
winner's-curse diagnostics for the allpick rule.

sigma_obs per dataset: pow3-residual median from expB_calib.py.

Run: python experiments/expB_e2.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_json, Timer, ROOT

from heckesel.lc import (fit_pow3_with_cov, phi_final, _pow3_jacobian,
                         EBModel)

ETA = 3.0
DATASETS = ["Fashion-MNIST", "adult", "higgs", "jasmine", "vehicle",
            "volkert"]
PD1_TASKS = ["imagenet__resnet", "lm1b__transformer",
             "uniref50__transformer", "svhn_no_extra__wide_resnet"]
N_POOLS = [50, 200]
N_SEEDS = 8
KAPPA = 1.0


def rungs_for(T: int) -> list[int]:
    """Rung epochs as fractions of the horizon (LCBench T=52 -> [4,12,36])."""
    r = [max(3, round(0.077 * T)), round(0.231 * T), round(0.692 * T)]
    out = []
    for v in r:
        while v in out or v < 3:
            v += 1
        out.append(min(v, T - 1))
    return out


def load_data():
    z = np.load(ROOT / "data" / "lcbench_cache.npz")
    calib = json.loads((ROOT / "data" / "lcbench_calib.json").read_text())
    data = {ds: z[ds] for ds in DATASETS}
    sig = {ds: calib["datasets"][ds]["sigma_pow3_median"]
           for ds in DATASETS}
    pd1p = ROOT / "data" / "pd1_cache.npz"
    if pd1p.exists():
        zp = np.load(pd1p)
        calp = json.loads((ROOT / "data" / "pd1_calib.json").read_text())
        for tsk in PD1_TASKS:
            data[tsk] = zp[tsk]
            sig[tsk] = calp["datasets"][tsk]["sigma_pow3_median"]
    return data, sig


def laplace_stats(y_pool: np.ndarray, plens: np.ndarray, sigma: float,
                  T: int):
    """Per-config LS fit + Laplace cov + delta-method extrapolation sd."""
    n = len(y_pool)
    ph = np.zeros((n, 3))
    V = np.zeros((n, 3, 3))
    pred = np.zeros(n)
    sd = np.zeros(n)
    for i in range(n):
        L = int(plens[i])
        t = np.arange(1, L + 1, dtype=float)
        ph[i], V[i] = fit_pow3_with_cov(t, y_pool[i, :L], sigma)
        pred[i] = phi_final(ph[i], T)[0]  # T = task horizon
        g = _pow3_jacobian(ph[i], np.array([float(T)]))[0]
        sd[i] = float(np.sqrt(max(g @ V[i] @ g, 0.0)))
    return ph, V, np.clip(pred, 0.0, 1.2), sd


def surrogate_predictions(method: str, y_pool, plens, sigma, seed, T):
    """Predicted finals for every config from its current prefix."""
    n = len(y_pool)
    if method == "last_value":
        return np.array([y_pool[i, int(plens[i]) - 1] for i in range(n)])
    ph, V, pred, sd = laplace_stats(y_pool, plens, sigma, T)
    if method == "naive_pow3":
        return pred
    if method == "naive_pow3_vpen":
        return pred - KAPPA * sd
    if method == "eb_replay":
        prior = EBModel.fit(ph, V, mode="all", sigma_obs=sigma,
                            steps=800, seed=seed)
        return np.clip(prior.predict_final_exact(
            y_pool, plens, ph, V, T, seed=seed), 0.0, 1.2)
    raise ValueError(method)


def replay(y_pool: np.ndarray, method: str, sigma: float, seed: int):
    """Synchronous SH replay with surrogate-driven promotion."""
    n, T = y_pool.shape
    rungs = rungs_for(T)
    alive = np.ones(n, dtype=bool)
    plens = np.zeros(n, dtype=int)
    budget = 0
    for t_k in rungs:
        run_now = alive & (plens < t_k)
        budget += int(((t_k - plens[run_now])).sum())
        plens[run_now] = t_k
        preds = surrogate_predictions(method, y_pool, np.maximum(plens, 1),
                                      sigma, seed, T)
        alive_idx = np.where(alive)[0]
        n_keep = max(1, int(round(len(alive_idx) / ETA)))
        order = alive_idx[np.argsort(preds[alive_idx])[::-1]]
        keep = set(order[:n_keep].tolist())
        for i in alive_idx:
            if i not in keep:
                alive[i] = False
    # run survivors to completion
    run_now = alive & (plens < T)
    budget += int(((T - plens[run_now])).sum())
    plens[run_now] = T

    finals = y_pool[:, -1]
    best = float(finals.max())
    completed = plens >= T
    pick_completed = int(np.argmax(np.where(completed, finals, -np.inf)))

    preds_end = surrogate_predictions(method, y_pool, plens, sigma, seed,
                                      T) \
        if method != "last_value" else np.array(
            [y_pool[i, plens[i] - 1] for i in range(n)])
    pick_all = int(np.argmax(preds_end))

    return {
        "budget": budget,
        "regret_completed": best - float(finals[pick_completed]),
        "regret_allpick": best - float(finals[pick_all]),
        "allpick_was_killed": bool(~completed[pick_all]),
        "n_completed": int(completed.sum()),
    }


def prediction_metrics(y_pool, sigma, seed):
    """Rank correlation / RMSE / bias of rung-0 predictions vs true finals
    across the full pool (held-out completions)."""
    n, T = y_pool.shape
    plens = np.full(n, rungs_for(T)[0])
    finals = y_pool[:, -1]
    out = {}
    for method in ["last_value", "naive_pow3", "naive_pow3_vpen",
                   "eb_replay"]:
        p = surrogate_predictions(method, y_pool, plens, sigma, seed, T)
        out[method] = {
            "spearman": float(stats.spearmanr(p, finals).statistic),
            "rmse": float(np.sqrt(np.mean((p - finals)**2))),
            "bias": float(np.mean(p - finals)),
            "bias_top_decile": float(np.mean(
                (p - finals)[np.argsort(p)[-max(1, n // 10):]])),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    data, sig = load_data()
    all_tasks = DATASETS + [t_ for t_ in PD1_TASKS if t_ in data]
    datasets = all_tasks[:2] if args.smoke else all_tasks
    pools = [50] if args.smoke else N_POOLS
    seeds = range(2 if args.smoke else N_SEEDS)

    rows = []
    for ds in datasets:
        curves = data[ds]
        ok = ~np.isnan(curves).any(axis=1)
        curves = np.clip(curves[ok], 0.0, 1.0)
        for n_pool in [p for p in pools if p <= len(curves)]:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                pool = curves[rng.choice(len(curves), n_pool,
                                         replace=False)]
                with Timer(f"{ds} n={n_pool} seed={seed}"):
                    row = {"dataset": ds, "n_pool": n_pool, "seed": seed,
                           "sigma": sig[ds],
                           "best": float(pool[:, -1].max()),
                           "pred@rung0": prediction_metrics(pool, sig[ds],
                                                            seed)}
                    for method in ["last_value", "naive_pow3",
                                   "naive_pow3_vpen", "eb_replay"]:
                        row[method] = replay(pool, method, sig[ds], seed)
                    rows.append(row)
    save_json("expB_e2_smoke.json" if args.smoke else "expB_e2.json",
              {"rows": rows, "eta": ETA, "kappa": KAPPA})

    print("\n===== regret (completed-pick) by method =====")
    for method in ["last_value", "naive_pow3", "naive_pow3_vpen",
                   "eb_replay"]:
        v = [r[method]["regret_completed"] for r in rows]
        va = [r[method]["regret_allpick"] for r in rows]
        print(f"{method:18s} completed={np.mean(v):.4f}  "
              f"allpick={np.mean(va):.4f}")


if __name__ == "__main__":
    main()
