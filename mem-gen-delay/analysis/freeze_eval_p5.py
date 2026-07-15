"""P5 FROZEN forecaster spec + one-shot held-out evaluation.

FROZEN (from train-side-only tuning, analysis/out5/tune_log2.txt; odd seeds untouched):
- MNIST: logistic on SRq features (S + R levels/slopes + wnorm^2 + cos_gap x wnorm terms),
  pos_weight=20, iters=8000, lr=0.3, W = 1.0 x median train t_gen, tau fit on train at
  FA <= 5%. Weights refit on ALL even seeds with these hyperparameters.
- ALGORITHMIC: NO multivariate configuration met the FA cap on the inner train split
  (single-signal, linear, and quadratic all fail) -> frozen verdict is the R1 first-pass
  single-signal table (already computed under the prereg spec) + this infeasibility record.
  K1/K2 assessed on that basis.

This file is committed BEFORE the test block below is executed (house prereg discipline);
the test evaluation runs exactly once per D3.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict_emergence as pe
from predict_emergence_ml import (quad_names, build_dataset, predict_run, evaluate)
from tune_ml_trainside import fit_logistic_w, fit_tau_q

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "analysis", "out5")

HP = dict(pos_weight=20, iters=8000, lr=0.3, w_mult=1.0)


def freeze_and_eval(train, test, label):
    names = quad_names("mnist", "SR")
    tg = [r["t_gen"] for r in train if r["t_gen"] is not None]
    W = HP["w_mult"] * float(np.median(tg))
    X, y = build_dataset(train, names, W)
    w, mu, sd = fit_logistic_w(X, y, HP["pos_weight"], 1e-3, HP["iters"], HP["lr"])
    fit = fit_tau_q(train, names, w, mu, sd)
    assert fit is not None, "no feasible tau on train"
    _, tau, tr = fit
    te = evaluate(test, names, w, mu, sd, tau)
    print(f"\n=== {label} ===")
    print(f"W={W:.0f} tau={tau:.4f}")
    print(f"TRAIN: lead {tr['median_lead']:.0f} miss {tr['miss_rate']:.2f} FA {tr['fa_rate']}")
    print(f"TEST : lead {te['median_lead']:.0f} rel {te['median_rel']:.3f} "
          f"miss {te['miss_rate']:.2f} FA {te['fa_rate']} "
          f"(nPos {te['n_pos']}, nNeg {te['n_neg']})")
    return dict(label=label, W=W, tau=float(tau),
                train={k: tr[k] for k in ("median_lead", "miss_rate", "fa_rate")},
                test=te, weights={n: round(float(v), 4) for n, v in
                                  zip(names + ["bias"], w)})


def main():
    runs = [r for r in pe.load_corpus() if r["domain"] == "mnist"]
    for r in runs:
        r["_sigs"] = pe.add_slopes(r)
    out = {}
    # R1: even -> odd
    out["r1"] = freeze_and_eval([r for r in runs if r["seed"] % 2 == 0],
                                [r for r in runs if r["seed"] % 2 == 1],
                                "R1 mnist frozen SRq (train=even, TEST=odd)")
    # R2: control-even -> prior arms (interventional shift)
    out["r2"] = freeze_and_eval([r for r in runs if r["family"] == "control"
                                 and r["seed"] % 2 == 0],
                                [r for r in runs if r["family"] == "prior"],
                                "R2 mnist frozen SRq (train=control even, TEST=prior arms)")
    json.dump(out, open(os.path.join(OUT, "frozen_eval.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(OUT, 'frozen_eval.json')}")


if __name__ == "__main__":
    main()
