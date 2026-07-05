"""E2 — analytic per-stream threshold allocation vs sweep incumbent.

Deployment knob: per-stream send-on-delta thresholds (input, h1, h2) of the
delta-GRU. Incumbent practice tunes them by search: N random configs, each
costing a full validation evaluation. Ours: ONE calibration pass records
traces; per-stream event-rate curves R_s(theta) come from open-loop
send-on-delta simulation on the cached traces (predictor (b); no further
network forwards); thresholds are then allocated ANALYTICALLY by inverting
the curves:

  - uniform-x:  theta_s = x* sigma_delta_s, with the single x* solved (by
    bisection on the predicted aggregate curve) to hit the global
    MAC-weighted event budget;
  - prop-share: each stream's weighted event budget is proportional to its
    dense MAC share; theta_s inverts that stream's own predicted curve;
  - (strawman) single-theta: one ABSOLUTE theta everywhere -- no per-stream
    statistics, shows why they matter.

Costs are counted in forward sequence-evaluations. Random search evaluates
its configs on the val subset; its Pareto front is then re-evaluated on test.
Analytic fronts go straight to test (their tuning cost is the calibration
pass only). Also logged: realized vs predicted event rates for the analytic
configs (closes the loop with E1).

Writes results/exp2_<task>.json.
"""

import argparse

import numpy as np
import torch

from common import CKPT, DEVICE, RESULTS, log, save_json

from exp1_prediction import delta_eval, record_gru_traces, sod_sim_rate

from eventrice import data as D
from eventrice.delta import GRUClassifier
from eventrice.energy import E_MAC_PJ, E_SRAM_PJ
from eventrice.rice import channel_moments

SEEDS = (0, 1, 2)
H = 128
N_RANDOM = 48
VAL_SUBSET = 1024
EV_SUBSET = 2048
BUDGET_FRACS = [0.6, 0.45, 0.3, 0.2, 0.12, 0.07, 0.04]


def stream_weights(in_size):
    """MACs per event of each stream (fan-out): input -> W_ih(l0); h1 -> W_hh(l0)
    + W_ih(l1); h2 -> W_hh(l1). Head ignored (dense, tiny; stated in paper)."""
    return dict(input=3 * H, h1=3 * H + 3 * H, h2=3 * H)


def dense_weighted_rate(in_size):
    """Dense MAC-weighted 'event' rate per step: every component fires."""
    w = stream_weights(in_size)
    return in_size * w["input"] + H * w["h1"] + H * w["h2"]


def rate_curves(cal_traces, sd, theta_grid_x):
    """R_s(theta) per stream via open-loop SOD sim on cached traces.
    Returns dict name -> (thetas, rates)."""
    curves = {}
    for name, tr in cal_traces.items():
        thetas = theta_grid_x * sd[name]
        rates = np.array([sod_sim_rate(tr, float(t)) for t in thetas])
        curves[name] = (thetas, rates)
    return curves


def weighted_rate_at_x(x, curves, sd, w, n_ch):
    tot = 0.0
    for name, (thetas, rates) in curves.items():
        th = x * sd[name]
        r = np.interp(th, thetas, rates, left=rates[0], right=rates[-1])
        tot += n_ch[name] * w[name] * r
    return tot


def alloc_uniform_x(budget_w, curves, sd, w, n_ch):
    lo, hi = 1e-3, 20.0
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        if weighted_rate_at_x(mid, curves, sd, w, n_ch) > budget_w:
            lo = mid
        else:
            hi = mid
    x = (lo * hi) ** 0.5
    return {name: x * sd[name] for name in curves}, x


def alloc_prop_share(budget_w, curves, w, n_ch, dense_share):
    out = {}
    for name, (thetas, rates) in curves.items():
        target_r = budget_w * dense_share[name] / (n_ch[name] * w[name])
        # rates decreasing in theta: invert by interp on reversed arrays
        r = np.clip(target_r, rates.min(), rates.max())
        out[name] = float(np.interp(-r, -rates, thetas))
    return out


def alloc_single_theta(budget_w, curves, w, n_ch):
    def wrate(th):
        tot = 0.0
        for name, (thetas, rates) in curves.items():
            r = np.interp(th, thetas, rates, left=rates[0], right=rates[-1])
            tot += n_ch[name] * w[name] * r
        return tot
    lo, hi = 1e-4, 50.0
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        if wrate(mid) > budget_w:
            lo = mid
        else:
            hi = mid
    th = (lo * hi) ** 0.5
    return {name: th for name in curves}


def measure(model, thetas_dict, x, y, in_size):
    """Closed-loop evaluation: (acc, weighted event rate per step, energy)."""
    d = model.as_delta([(thetas_dict["input"], thetas_dict["h1"]),
                        (thetas_dict["h1"], thetas_dict["h2"])])
    rates, acc = delta_eval(d, model, x, y)
    w = stream_weights(in_size)
    n_ch = dict(input=in_size, h1=H, h2=H)
    wev = sum(n_ch[n] * w[n] * rates[n] for n in rates)
    energy = sum(n_ch[n] * rates[n] * (w[n] * (E_MAC_PJ + E_SRAM_PJ)
                                       + 2 * E_SRAM_PJ) for n in rates)
    return acc, float(wev), float(energy), rates


def pareto_front(points):
    """points: list of (events, acc, payload). Returns the non-dominated set
    sorted by events (lower events, higher acc dominate)."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front, best = [], -1.0
    for p in pts:
        if p[1] > best:
            front.append(p)
            best = p[1]
    return front


def run_task(task):
    if task == "sc2":
        xva, yva = D.load_sc2("val", "cpu")
        xte, yte = D.load_sc2("test", "cpu")
        in_size, n_cls = 40, 35
    else:
        xva, yva = D.load_psmnist("val", "cpu")
        xte, yte = D.load_psmnist("test", "cpu")
        in_size, n_cls = 28, 10
    g = torch.Generator().manual_seed(99)
    vi = torch.randperm(len(xva), generator=g)[:VAL_SUBSET]
    ti = torch.randperm(len(xte), generator=g)[:EV_SUBSET]
    xva_s, yva_s = xva[vi], yva[vi]
    xte_s, yte_s = xte[ti], yte[ti]
    w = stream_weights(in_size)
    n_ch = dict(input=in_size, h1=H, h2=H)
    dense_w = dense_weighted_rate(in_size)
    dense_share = {n: n_ch[n] * w[n] / dense_w for n in n_ch}

    results = {}
    for seed in SEEDS:
        model = GRUClassifier(in_size, H, 2, n_cls).to(DEVICE)
        model.load_state_dict(torch.load(CKPT / f"{task}_base_s{seed}.pt",
                                         weights_only=True))
        model.eval()

        # calibration: ONE pass over the val subset (cost = VAL_SUBSET fwd)
        cal = record_gru_traces(model, xva_s)
        cal_traces = {"input": xva_s, "h1": cal[0], "h2": cal[1]}
        sd, tv = {}, {}
        for name, tr in cal_traces.items():
            m = channel_moments(tr.numpy())
            sd[name] = float(np.mean(m["sigma_delta"]))
            tv[name] = float(np.mean(m["tv_rate"]))
        theta_grid_x = np.geomspace(0.02, 8.0, 25)
        curves = rate_curves(cal_traces, sd, theta_grid_x)

        methods = {}
        for frac in BUDGET_FRACS:
            budget_w = frac * dense_w
            th_u, xstar = alloc_uniform_x(budget_w, curves, sd, w, n_ch)
            th_p = alloc_prop_share(budget_w, curves, w, n_ch, dense_share)
            th_s = alloc_single_theta(budget_w, curves, w, n_ch)
            for mname, th in (("uniform_x", th_u), ("prop_share", th_p),
                              ("single_theta", th_s)):
                acc, wev, en, rates = measure(model, th, xte_s, yte_s, in_size)
                pred_w = sum(n_ch[n] * w[n] * np.interp(
                    th[n], curves[n][0], curves[n][1]) for n in th)
                methods.setdefault(mname, []).append(dict(
                    budget_frac=frac, thetas=th, acc=acc,
                    weighted_events=wev, energy_pj_per_step=en,
                    frac_of_dense=wev / dense_w,
                    predicted_weighted=float(pred_w),
                    predicted_frac=float(pred_w / dense_w), rates=rates))
            log(f"[{task} s{seed}] frac {frac}: uniform_x acc "
                f"{methods['uniform_x'][-1]['acc']:.4f} realized "
                f"{methods['uniform_x'][-1]['frac_of_dense']:.3f}")

        # incumbent: random search on val
        rng = np.random.default_rng(1000 + seed)
        rs_configs, rs_val = [], []
        for i in range(N_RANDOM):
            th = {n: float(rng.uniform(np.log(0.05), np.log(3.0)))
                  for n in ("input", "h1", "h2")}
            th = {n: float(np.exp(v) * sd[n]) for n, v in th.items()}
            acc, wev, en, _ = measure(model, th, xva_s, yva_s, in_size)
            rs_configs.append(th)
            rs_val.append((wev, acc, i))
        # fronts at increasing tuning budgets, re-evaluated on TEST
        rs_fronts = {}
        for k in (6, 12, 24, 48):
            front = pareto_front(rs_val[:k])
            rows = []
            for wev, acc_val, i in front:
                acc, wev_t, en, rates = measure(model, rs_configs[i],
                                                xte_s, yte_s, in_size)
                rows.append(dict(config=rs_configs[i], val_acc=acc_val,
                                 acc=acc, weighted_events=wev_t,
                                 energy_pj_per_step=en,
                                 frac_of_dense=wev_t / dense_w))
            rs_fronts[f"k{k}"] = rows
        log(f"[{task} s{seed}] random search done")

        results[f"s{seed}"] = dict(
            sd=sd, tv=tv, dense_weighted=dense_w,
            curves={n: dict(thetas=c[0].tolist(), rates=c[1].tolist())
                    for n, c in curves.items()},
            analytic=methods, random_search=rs_fronts,
            cost_forwards=dict(
                calibration=VAL_SUBSET,
                analytic_extra=0,
                random_search_per_config=VAL_SUBSET,
                random_search_total=N_RANDOM * VAL_SUBSET))
    save_json(results, RESULTS / f"exp2_{task}.json")


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
