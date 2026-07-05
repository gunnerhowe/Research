"""E4 — null controls (the experiment that makes it a paper).

(i)  L1 penalty on one-step activation deltas |h_t - h_{t-1}| — the obvious
     cheap alternative. If indistinguishable from the budget at matched
     events, the crossing machinery adds nothing: reported honestly.
     Mechanism test: delta-magnitude histograms under both (L1 shrinks ALL
     deltas; the budget prices only crossings and is silent within budget).
(ii) Global activity/rate regularizer (spiking-style), mean h^2.
(iii) Post-hoc threshold sweep on the same architecture = exp3's "posthoc"
     arm (shared machinery; no re-run here).

Same protocol as exp3: fine-tune base model, then closed-loop theta-sweep
front with per-model calibration. Writes results/exp4_<task>.json.
"""

import argparse

import numpy as np
import torch

from common import CKPT, DEVICE, RESULTS, log, save_json

from exp1_prediction import record_gru_traces
from exp3_budget_training import (EV_SUBSET, CAL_SUBSET, SEEDS, H,
                                  finetune_arm, load_task, subset,
                                  theta_sweep_front)

from eventrice.budget import L1DeltaPenalty, RatePenalty
from eventrice.delta import GRUClassifier
from eventrice.train import evaluate_classifier

L1_GRID = [0.03, 0.1, 0.3, 1.0]
RATE_GRID = [0.03, 0.1, 0.3]
LAMBDA_UNIT = 1.0  # grids ARE the reg weights


def run_task(task, seeds=SEEDS):
    train_xy, (xva, yva), (xte, yte), in_size, n_cls = load_task(task)
    xcal, _ = subset(xva, yva, CAL_SUBSET)
    xte_s, yte_s = subset(xte, yte, EV_SUBSET, seed=77)
    xcal = xcal.to(DEVICE)
    val_gpu = (xva.to(DEVICE), yva.to(DEVICE))

    l1_mod, rate_mod = L1DeltaPenalty(), RatePenalty()

    def l1_reg(trs):
        return sum(l1_mod(t) for t in trs) / len(trs)

    def rate_reg(trs):
        return sum(rate_mod(t) for t in trs) / len(trs)

    results = {}
    for seed in seeds:
        base = GRUClassifier(in_size, H, 2, n_cls).to(DEVICE)
        base.load_state_dict(torch.load(CKPT / f"{task}_base_s{seed}.pt",
                                        weights_only=True))
        base.eval()
        arms = {}
        for lam in L1_GRID:
            model, hist = finetune_arm(base, l1_reg, lam, train_xy, val_gpu,
                                       seed)
            front, sd = theta_sweep_front(model, xcal, xte_s, yte_s, in_size)
            arms[f"l1delta_{lam}"] = dict(
                lam=lam, history=hist, front=front, sd=sd,
                dense_acc=evaluate_classifier(model, xte_s.to(DEVICE),
                                              yte_s.to(DEVICE)))
            torch.save(model.state_dict(),
                       CKPT / f"{task}_l1delta{lam}_s{seed}.pt")
            log(f"[{task} s{seed}] l1delta lam={lam} dense acc "
                f"{arms[f'l1delta_{lam}']['dense_acc']:.4f}")
        for lam in RATE_GRID:
            model, hist = finetune_arm(base, rate_reg, lam, train_xy, val_gpu,
                                       seed)
            front, sd = theta_sweep_front(model, xcal, xte_s, yte_s, in_size)
            arms[f"rate_{lam}"] = dict(
                lam=lam, history=hist, front=front, sd=sd,
                dense_acc=evaluate_classifier(model, xte_s.to(DEVICE),
                                              yte_s.to(DEVICE)))
            log(f"[{task} s{seed}] rate lam={lam} dense acc "
                f"{arms[f'rate_{lam}']['dense_acc']:.4f}")
        results[f"s{seed}"] = arms
        save_json(results, RESULTS / f"exp4_{task}.json")

    # ---- mechanism test: delta-magnitude histograms ----
    hist_out = {}
    for seed in seeds:
        row = {}
        for name, ck in (("base", f"{task}_base_s{seed}.pt"),
                         ("budget", f"{task}_budget_rho0.35_s{seed}.pt"),
                         ("l1delta", f"{task}_l1delta{L1_GRID[1]}_s{seed}.pt")):
            m = GRUClassifier(in_size, H, 2, n_cls).to(DEVICE)
            m.load_state_dict(torch.load(CKPT / ck, weights_only=True))
            m.eval()
            traces = record_gru_traces(m, xcal)
            d = torch.cat([(t[:, 1:] - t[:, :-1]).abs().flatten()
                           for t in traces]).numpy()
            bins = np.geomspace(1e-6, 2.0, 61)
            cnt, _ = np.histogram(d, bins=bins)
            row[name] = dict(bins=bins.tolist(), counts=cnt.tolist(),
                             mean_abs_delta=float(d.mean()),
                             p95=float(np.quantile(d, 0.95)),
                             p99=float(np.quantile(d, 0.99)))
        hist_out[f"s{seed}"] = row
        log(f"[{task} s{seed}] histograms done")
    save_json(hist_out, RESULTS / f"exp4_{task}_histograms.json")


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
