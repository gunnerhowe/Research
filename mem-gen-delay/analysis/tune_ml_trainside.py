"""P5: TRAIN-SIDE-ONLY tuning of the multivariate forecaster implementation.

Discipline: the first ml pass showed underfitting visible on TRAIN metrics; per D3 we may
iterate implementation choices against TRAIN ONLY. This script therefore loads EVEN-SEED
runs exclusively (held-out odd seeds never enter memory), and within them uses an inner
split (seeds % 4 == 0 fit / % 4 == 2 validate) to pick: class weighting, iterations,
learning rate, tau grid source, and W multiplier. The winning configuration gets frozen
into the committed spec, then evaluated ONCE on the true test block.
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict_emergence as pe
from predict_emergence_ml import (featnames, quad_names, build_dataset, run_matrix,
                                  predict_run, alarm_lead, evaluate)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "analysis", "out5")
FA_CAP = 0.05


def fit_logistic_w(X, y, pos_weight, l2, iters, lr):
    mu = np.nanmean(X, 0)
    sd = np.nanstd(X, 0) + 1e-9
    Xz = (np.where(np.isnan(X), mu, X) - mu) / sd
    Xz = np.hstack([Xz, np.ones((len(Xz), 1))])
    w = np.zeros(Xz.shape[1])
    sw = np.where(y > 0.5, pos_weight, 1.0)
    sw = sw / sw.mean()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xz @ w, -30, 30)))
        g = Xz.T @ ((p - y) * sw) / len(y) + l2 * w
        w -= lr * g
    return w, mu, sd


def fit_tau_q(train_runs, names, w, mu, sd):
    """tau candidates from quantiles of predicted probabilities on train runs."""
    ps = np.concatenate([predict_run(r, names, w, mu, sd)[pe.WARMUP:] for r in train_runs])
    taus = np.unique(np.quantile(ps, np.linspace(0.5, 0.999, 60)))
    best = None
    for tau in taus:
        sc = evaluate(train_runs, names, w, mu, sd, float(tau))
        if sc["fa_rate"] is not None and sc["fa_rate"] > FA_CAP:
            continue
        if best is None or sc["median_lead"] > best[0]:
            best = (sc["median_lead"], float(tau), sc)
    return best


def main():
    # EVEN seeds only — odd (test) seeds never loaded
    runs = [r for r in pe.load_corpus() if r["seed"] % 2 == 0]
    for r in runs:
        r["_sigs"] = pe.add_slopes(r)
    grid_pw = [1, 5, 20]
    grid_iters = [2000, 8000]
    grid_lr = [0.3]
    grid_wmult = [0.2, 0.5, 1.0]
    results = []
    for domain in ("alg", "mnist"):
        dom = [r for r in runs if r["domain"] == domain]
        fit_runs = [r for r in dom if r["seed"] % 4 == 0]
        val_runs = [r for r in dom if r["seed"] % 4 == 2]
        tg = [r["t_gen"] for r in fit_runs if r["t_gen"] is not None]
        base_W = float(np.median(tg))
        for which, use_quad in (("S", False), ("R", False), ("SR", False), ("SR", True)):
            names = quad_names(domain, which) if use_quad else featnames(domain, which)
            which = which + ("q" if use_quad else "")
            for pw, iters, lr, wm in itertools.product(grid_pw, grid_iters, grid_lr,
                                                       grid_wmult):
                W = wm * base_W
                X, y = build_dataset(fit_runs, names, W)
                if y.sum() < 20:
                    continue
                w, mu, sd = fit_logistic_w(X, y, pw, 1e-3, iters, lr)
                fit = fit_tau_q(fit_runs, names, w, mu, sd)
                if fit is None:
                    continue
                _, tau, _ = fit
                val = evaluate(val_runs, names, w, mu, sd, tau)
                if val["fa_rate"] is not None and val["fa_rate"] > 2 * FA_CAP:
                    continue  # unstable FA across inner split
                results.append(dict(domain=domain, set=which, pos_weight=pw, iters=iters,
                                    lr=lr, w_mult=wm, tau=round(tau, 4),
                                    val_lead=round(val["median_lead"], 1),
                                    val_rel=round(val["median_rel"], 4),
                                    val_miss=val["miss_rate"], val_fa=val["fa_rate"]))
                print(json.dumps(results[-1]), flush=True)
    results.sort(key=lambda r: (r["domain"], r["set"], -r["val_lead"]))
    json.dump(results, open(os.path.join(OUT, "tune_trainside.json"), "w"), indent=2)
    print("\nBEST per domain/set:")
    seen = set()
    for r in results:
        k = (r["domain"], r["set"])
        if k in seen:
            continue
        seen.add(k)
        print(json.dumps(r))


if __name__ == "__main__":
    main()
