"""E3b — mechanism probe: matched-ABSOLUTE-theta vs matched-events views.

The E3 finding is that budget training rescales dynamics (sigma_delta
shrinks) without improving the accuracy-events Pareto front under
scale-adapted thresholds (theta proportional to each model's own
sigma_delta): send-on-delta event rates are approximately scale-free in
theta/sigma_delta. This probe makes the distinction explicit: evaluate the
budget-trained model at the BASE model's absolute theta grid. At matched
absolute theta the budget model produces far fewer events (its traces are
smoother in absolute units) -- the apparent "savings" that a fixed-theta
comparison would report -- while the matched-events comparison (E3 fronts)
shows no gain. Both views are reported; the field's fixed-theta comparisons
overstate what smoothness training buys.

Writes results/exp3b_<task>.json.
"""

import argparse

import numpy as np
import torch

from common import CKPT, DEVICE, RESULTS, log, save_json

from exp1_prediction import record_gru_traces
from exp2_allocation import dense_weighted_rate, measure
from exp3_budget_training import (CAL_SUBSET, EV_SUBSET, H, X_SWEEP,
                                  load_task, subset)

from eventrice.delta import GRUClassifier
from eventrice.rice import channel_moments

SEEDS = (0, 1, 2)
ARM = "budget_rho0.35"


def run_task(task):
    train_xy, (xva, yva), (xte, yte), in_size, n_cls = load_task(task)
    xcal, _ = subset(xva, yva, CAL_SUBSET)
    xte_s, yte_s = subset(xte, yte, EV_SUBSET, seed=77)
    xcal = xcal.to(DEVICE)
    dense_w = dense_weighted_rate(in_size)

    results = {}
    for seed in SEEDS:
        base = GRUClassifier(in_size, H, 2, n_cls).to(DEVICE)
        base.load_state_dict(torch.load(CKPT / f"{task}_base_s{seed}.pt",
                                        weights_only=True))
        base.eval()
        tuned = GRUClassifier(in_size, H, 2, n_cls).to(DEVICE)
        tuned.load_state_dict(torch.load(CKPT / f"{task}_{ARM}_s{seed}.pt",
                                         weights_only=True))
        tuned.eval()

        cal = record_gru_traces(base, xcal)
        streams = {"input": xcal.cpu(), "h1": cal[0], "h2": cal[1]}
        sd_base = {n: float(np.mean(channel_moments(t.numpy())["sigma_delta"]))
                   for n, t in streams.items()}

        rows = []
        for x in X_SWEEP:
            th = {n: x * sd_base[n] for n in sd_base}  # BASE-scaled thetas
            acc_b, wev_b, _, _ = measure(base, th, xte_s, yte_s, in_size)
            acc_t, wev_t, _, _ = measure(tuned, th, xte_s, yte_s, in_size)
            rows.append(dict(x=x, thetas=th,
                             base=dict(acc=acc_b, frac=wev_b / dense_w),
                             tuned=dict(acc=acc_t, frac=wev_t / dense_w)))
            log(f"[{task} s{seed}] abs-theta x={x}: base "
                f"({wev_b/dense_w:.3f}, {acc_b:.4f}) tuned "
                f"({wev_t/dense_w:.3f}, {acc_t:.4f})")
        results[f"s{seed}"] = dict(sd_base=sd_base, arm=ARM, rows=rows)
    save_json(results, RESULTS / f"exp3b_{task}.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all", choices=["all", "sc2", "psmnist"])
    args = p.parse_args()
    if args.task in ("all", "sc2"):
        run_task("sc2")
    if args.task in ("all", "psmnist"):
        run_task("psmnist")


if __name__ == "__main__":
    main()
