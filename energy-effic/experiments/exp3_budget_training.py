"""E3 — budget training (the headline, if it works).

Per task (SC2, psMNIST) and seed: fine-tune the trained base model with

    L = L_task + lambda * mean_l BUDGET_l(h_l traces)

where BUDGET_l is the one-sided crossing budget (levels on the layer's
quantile ladder from a calibration pass, budgets = rho * calibration profile;
rho < 1 is the demanded crossing reduction and is the swept Pareto knob;
lambda fixed after a one-off sanity check).

Every model (base = post-hoc arm, and each fine-tuned arm) is then evaluated
CLOSED LOOP as a delta net over a theta sweep theta_s = x * sigma_delta_s
with sigma_delta from that model's OWN calibration traces (fair deployment
protocol: thresholds are re-tuned per model). Fronts: test accuracy vs
MAC-weighted event rate (and modeled energy).

Question: does training-for-temporal-sparsity beat post-hoc thresholding of
the same base model at matched event budget? exp4 adds the null controls.

Writes results/exp3_<task>.json (shared machinery imported by exp4).
"""

import argparse
import copy

import numpy as np
import torch

from common import CKPT, DEVICE, RESULTS, log, save_json

from exp1_prediction import record_gru_traces
from exp2_allocation import dense_weighted_rate, measure, stream_weights

from eventrice import data as D
from eventrice.budget import TemporalCrossingBudget
from eventrice.delta import GRUClassifier
from eventrice.rice import channel_moments
from eventrice.train import evaluate_classifier, finetune_classifier

SEEDS = (0, 1, 2)
H = 128
X_SWEEP = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
RHO_GRID = [0.7, 0.5, 0.35, 0.2]
LAMBDA = 3.0
FT_EPOCHS = 5
CAL_SUBSET = 1024
EV_SUBSET = 2048


def load_task(task):
    if task == "sc2":
        xtr, ytr = D.load_sc2("train", DEVICE)
        xva, yva = D.load_sc2("val", "cpu")
        xte, yte = D.load_sc2("test", "cpu")
        return (xtr, ytr), (xva, yva), (xte, yte), 40, 35
    xtr, ytr = D.load_psmnist("train", DEVICE)
    xva, yva = D.load_psmnist("val", "cpu")
    xte, yte = D.load_psmnist("test", "cpu")
    return (xtr, ytr), (xva, yva), (xte, yte), 28, 10


def subset(x, y, n, seed=99):
    g = torch.Generator().manual_seed(seed)
    i = torch.randperm(len(x), generator=g)[:n]
    return x[i], y[i]


def make_budget_reg(model, xcal, rho, n_levels=16, eps_scale=0.15):
    """Per-layer one-sided budgets from the model's calibration traces."""
    traces = record_gru_traces(model, xcal)
    budgets = [TemporalCrossingBudget.from_calibration(
        tr.transpose(1, 2), rho, n_levels=n_levels,
        eps_scale=eps_scale).to(DEVICE) for tr in traces]

    def reg(trs):
        return sum(b(tr.transpose(1, 2)) for b, tr in zip(budgets, trs)) / len(budgets)

    return reg


def theta_sweep_front(model, xcal, xte, yte, in_size):
    """Closed-loop x-sweep with the model's own sigma_delta calibration."""
    cal = record_gru_traces(model, xcal)
    streams = {"input": xcal.cpu(), "h1": cal[0], "h2": cal[1]}
    sd = {n: float(np.mean(channel_moments(t.numpy())["sigma_delta"]))
          for n, t in streams.items()}
    dense_w = dense_weighted_rate(in_size)
    rows = []
    for x in X_SWEEP:
        th = {n: x * sd[n] for n in sd}
        acc, wev, en, rates = measure(model, th, xte, yte, in_size)
        rows.append(dict(x=x, thetas=th, acc=acc, weighted_events=wev,
                         energy_pj_per_step=en, frac_of_dense=wev / dense_w,
                         rates=rates))
    return rows, sd


def finetune_arm(base_model, reg, reg_weight, train_xy, val_xy, seed,
                 epochs=FT_EPOCHS, lr=3e-4):
    model = copy.deepcopy(base_model)
    hist = finetune_classifier(model, train_xy, val_xy, reg, reg_weight,
                               epochs=epochs, lr=lr, seed=seed, log=log)
    return model, hist


def run_task(task, seeds=SEEDS):
    train_xy, (xva, yva), (xte, yte), in_size, n_cls = load_task(task)
    xcal, _ = subset(xva, yva, CAL_SUBSET)
    xte_s, yte_s = subset(xte, yte, EV_SUBSET, seed=77)
    xcal = xcal.to(DEVICE)
    val_gpu = (xva.to(DEVICE), yva.to(DEVICE))

    results = {}
    for seed in seeds:
        base = GRUClassifier(in_size, H, 2, n_cls).to(DEVICE)
        base.load_state_dict(torch.load(CKPT / f"{task}_base_s{seed}.pt",
                                        weights_only=True))
        base.eval()
        arms = {}

        front, sd0 = theta_sweep_front(base, xcal, xte_s, yte_s, in_size)
        arms["posthoc"] = dict(front=front, sd=sd0,
                               dense_acc=evaluate_classifier(base, xte_s.to(DEVICE),
                                                             yte_s.to(DEVICE)))
        log(f"[{task} s{seed}] posthoc dense acc {arms['posthoc']['dense_acc']:.4f}")

        for rho in RHO_GRID:
            reg = make_budget_reg(base, xcal, rho)
            model, hist = finetune_arm(base, reg, LAMBDA, train_xy, val_gpu,
                                       seed)
            front, sd = theta_sweep_front(model, xcal, xte_s, yte_s, in_size)
            arms[f"budget_rho{rho}"] = dict(
                rho=rho, lam=LAMBDA, history=hist, front=front, sd=sd,
                dense_acc=evaluate_classifier(model, xte_s.to(DEVICE),
                                              yte_s.to(DEVICE)))
            torch.save(model.state_dict(),
                       CKPT / f"{task}_budget_rho{rho}_s{seed}.pt")
            log(f"[{task} s{seed}] budget rho={rho} dense acc "
                f"{arms[f'budget_rho{rho}']['dense_acc']:.4f}")
        results[f"s{seed}"] = arms
        save_json(results, RESULTS / f"exp3_{task}.json")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all", choices=["all", "sc2", "psmnist"])
    p.add_argument("--seeds", default=None,
                   help="comma-separated; default 0-7 for sc2, 0-2 psmnist")
    args = p.parse_args()
    for task, default_seeds in (("sc2", range(8)), ("psmnist", range(3))):
        if args.task in ("all", task):
            seeds = (tuple(int(s) for s in args.seeds.split(","))
                     if args.seeds else tuple(default_seeds))
            run_task(task, seeds)


if __name__ == "__main__":
    main()
