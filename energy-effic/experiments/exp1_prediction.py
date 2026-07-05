"""E1 — prediction on real trained networks (GATE P).

Two measured quantities, per task and seed:

1. CROSSING-RATE PROFILES of per-channel activation traces on the evaluation
   split, at channel-standardized levels mean_c + k sigma_c, vs three
   predictors fitted on a disjoint CALIBRATION split:
     (a) Rice from fitted spectral moments (Gaussian assumption),
         + the discrete-time Gaussian refinement (appendix),
     (b) empirical crossing profile (no Gaussianity) -- the load-bearing,
         differentiable one; GATE P: median abs relative error <= 10%,
     (c) Gaussian-iid baseline (no temporal structure).
   Plus split-half noise floors and the kurtosis/Rice-error correlation
   (where does Gaussianity break).

2. DELTA-EVENT RATES of the faithful delta network (closed loop, all streams
   thresholded at theta = x * sigma_delta(layer), x swept) vs:
     - analytic: per-channel TV/theta ladder bound x the universal gamma
       curve calibrated once in E0 (Rice-side: TV from lambda2),
     - empirical: send-on-delta simulation on the CALIBRATION traces,
     - open-loop simulation on eval traces (isolates feedback error).

Writes results/exp1_<task>.json.
"""

import argparse

import numpy as np
import torch

from common import CKPT, DEVICE, RESULTS, load_json, log, save_json

from eventrice import data as D
from eventrice.delta import GRUClassifier, TransformerLM, delta_encode_trace
from eventrice.rice import (SodCorrection, channel_moments,
                            gaussian_discrete_rate, gaussian_tv_rate,
                            iid_rate, rice_rate, sod_rate_bound)

SEEDS = (0, 1, 2)
K_SIGMA = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
X_GRID = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
RATE_FLOOR = 0.005  # rates/step below this are inside the noise floor


# ------------------------------------------------------------------ helpers


def per_channel_hard_rates(traces, levels):
    """traces (B,T,C) torch float; levels (C,L) numpy -> (C,L) rates/step."""
    a = traces.permute(2, 0, 1)                      # (C,B,T)
    C = a.shape[0]
    a0 = a[..., :-1].reshape(C, 1, -1)
    a1 = a[..., 1:].reshape(C, 1, -1)
    u = torch.from_numpy(levels).to(a0).unsqueeze(-1)  # (C,L,1)
    crossed = ((a0 - u) * (a1 - u) < 0).float()
    return crossed.mean(dim=2).cpu().numpy()


def profile_analysis(cal, ev):
    """cal/eval: (B,T,C) torch cpu float32. Returns metrics dict."""
    mom = channel_moments(cal.numpy())
    sig = np.sqrt(np.maximum(mom["lam0"], 1e-30))
    levels = mom["mean"][:, None] + K_SIGMA[None, :] * sig[:, None]  # (C,L)
    meas = per_channel_hard_rates(ev, levels)
    # noise floor: split eval batch in halves
    h = ev.shape[0] // 2
    m1 = per_channel_hard_rates(ev[:h], levels)
    m2 = per_channel_hard_rates(ev[h:], levels)

    C = levels.shape[0]
    pred = {}
    pred["rice"] = np.stack([rice_rate(levels[c], mom["mean"][c],
                                       mom["lam0"][c], mom["lam2"][c])
                             for c in range(C)])
    pred["discrete"] = np.stack([gaussian_discrete_rate(
        levels[c], mom["mean"][c], mom["lam0"][c], mom["rho1"][c])
        for c in range(C)])
    pred["iid"] = np.stack([iid_rate(levels[c], mom["mean"][c], mom["lam0"][c])
                            for c in range(C)])
    pred["empirical"] = per_channel_hard_rates(cal, levels)

    ok = meas > RATE_FLOOR
    floor = np.abs(m1 - m2)[ok] / meas[ok]
    out = dict(n_channels=C, n_levels_used=int(ok.sum()),
               frac_levels_used=float(ok.mean()),
               split_half_floor_median=float(np.median(floor)),
               kurt=mom["kurt"].tolist())
    for name, p in pred.items():
        rel = np.abs(p - meas)[ok] / meas[ok]
        per_ch = [float(np.median(np.abs(p[c][ok[c]] - meas[c][ok[c]])
                                  / meas[c][ok[c]]))
                  if ok[c].any() else None for c in range(C)]
        out[name] = dict(median_rel_err=float(np.median(rel)),
                         mean_rel_err=float(np.mean(rel)),
                         frac_within_10pct=float(np.mean(rel <= 0.10)),
                         per_channel_median=per_ch)
    # exemplar channels for figures: most active, median kurtosis, max kurtosis
    ex_ids = [int(np.argmax(meas.max(axis=1))),
              int(np.argsort(mom["kurt"])[C // 2]),
              int(np.argmax(mom["kurt"]))]
    out["examples"] = [dict(channel=c, kurt=float(mom["kurt"][c]),
                            k_sigma=K_SIGMA.tolist(),
                            levels=levels[c].tolist(),
                            measured=meas[c].tolist(),
                            **{n: pred[n][c].tolist() for n in pred})
                       for c in ex_ids]
    return out, mom


def event_predictors(mom, theta, gamma):
    """Per-layer predicted SOD event rate at threshold theta: mean over
    channels of the gamma-corrected TV/theta bound (empirical TV) and the
    Gaussian-TV (Rice-side) version."""
    b_emp = sod_rate_bound(theta, mom["tv_rate"])
    b_gau = sod_rate_bound(theta, gaussian_tv_rate(mom["lam2"]))
    g = gamma(theta / np.maximum(mom["sigma_delta"], 1e-30))
    return float(np.mean(np.minimum(b_emp * g, 1.0))), \
        float(np.mean(np.minimum(b_gau * g, 1.0)))


def sod_sim_rate(traces, theta, device=DEVICE, chunk=512):
    """Open-loop send-on-delta event rate on recorded traces."""
    tot, n = 0.0, 0
    for i in range(0, traces.shape[0], chunk):
        t = traces[i:i + chunk].to(device)
        _, ev = delta_encode_trace(t, theta)
        tot += ev.float().sum().item()
        n += ev.numel()
    return tot / n


# ------------------------------------------------------------------ GRU E1


def run_gru_task(task):
    if task == "sc2":
        xcal, _ = D.load_sc2("val", "cpu")
        xev, yev = D.load_sc2("test", "cpu")
        in_size, n_cls = 40, 35
    else:
        xcal, _ = D.load_psmnist("val", "cpu")
        xev, yev = D.load_psmnist("test", "cpu")
        in_size, n_cls = 28, 10
    g = torch.Generator().manual_seed(1234)
    cal_idx = torch.randperm(len(xcal), generator=g)[:2048]
    ev_idx = torch.randperm(len(xev), generator=g)[:2048]
    xcal = xcal[cal_idx]
    xev, yev = xev[ev_idx], yev[ev_idx]

    gamma = SodCorrection.from_json(
        {"x_grid": load_json(RESULTS / "exp0_sod_gamma.json")["x_grid"],
         "gamma": load_json(RESULTS / "exp0_sod_gamma.json")["pooled_gamma"]})

    results = {}
    for seed in SEEDS:
        model = GRUClassifier(in_size, 128, 2, n_cls).to(DEVICE)
        model.load_state_dict(torch.load(CKPT / f"{task}_base_s{seed}.pt",
                                         weights_only=True))
        model.eval()

        cal_traces = record_gru_traces(model, xcal)
        ev_traces = record_gru_traces(model, xev)
        streams = {"input": (xcal, xev)}
        for l in range(model.num_layers):
            streams[f"h{l+1}"] = (cal_traces[l], ev_traces[l])

        profiles, moments = {}, {}
        for name, (c, e) in streams.items():
            profiles[name], moments[name] = profile_analysis(c, e)
            log(f"[{task} s{seed}] {name}: emp {profiles[name]['empirical']['median_rel_err']:.3f} "
                f"rice {profiles[name]['rice']['median_rel_err']:.3f} "
                f"iid {profiles[name]['iid']['median_rel_err']:.3f}")

        # ---- delta-event prediction (closed loop, joint thresholds) ----
        sd = {n: float(np.mean(m["sigma_delta"])) for n, m in moments.items()}
        event_rows = []
        for x in X_GRID:
            thetas = [(x * sd["input"], x * sd["h1"]),
                      (x * sd["h1"], x * sd["h2"])]
            dmodel = model.as_delta(thetas)
            ev_meas, acc = delta_eval(dmodel, model, xev, yev)
            row = dict(x=x, thetas=[list(t) for t in thetas], acc=acc)
            for li, name in enumerate(["input", "h1", "h2"]):
                th = x * sd[name]
                mrate = ev_meas[name]
                p_emp, p_gau = event_predictors(moments[name], th, gamma)
                ol_ev = sod_sim_rate(streams[name][1], th)
                ol_cal = sod_sim_rate(streams[name][0], th)
                row[name] = dict(theta=th, measured=mrate,
                                 pred_analytic_emp_tv=p_emp,
                                 pred_analytic_gauss_tv=p_gau,
                                 pred_openloop_cal=ol_cal,
                                 openloop_eval=ol_ev)
            event_rows.append(row)
            log(f"[{task} s{seed}] x={x}: acc {acc:.4f} "
                f"h1 meas {event_rows[-1]['h1']['measured']:.4f} "
                f"pred {event_rows[-1]['h1']['pred_analytic_emp_tv']:.4f}")
        results[f"s{seed}"] = dict(profiles=profiles, events=event_rows)
    save_json(results, RESULTS / f"exp1_{task}.json")
    return results


def record_gru_traces(model, x, batch=512):
    outs = [[] for _ in range(model.num_layers)]
    with torch.no_grad():
        for i in range(0, len(x), batch):
            _, tr = model(x[i:i + batch].to(DEVICE), return_traces=True)
            for l, t in enumerate(tr):
                outs[l].append(t.cpu())
    return [torch.cat(o) for o in outs]


def delta_eval(dmodel, model, x, y, batch=512):
    """Closed-loop delta run: per-stream event rates + accuracy."""
    ev = dict(input=[0.0, 0.0], h1=[0.0, 0.0], h2=[0.0, 0.0])
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].to(DEVICE)
            out, stats = dmodel(xb)
            logits = model.head(out.mean(dim=1))
            correct += (logits.argmax(1).cpu() == y[i:i + batch]).sum().item()
            # stream events: input = layer0 x; h1 = layer0 h (== layer1 x
            # stream, separate anchor but same trace); h2 = layer1 h
            ev["input"][0] += stats[0]["events_x"]
            ev["input"][1] += stats[0]["dense_x"]
            ev["h1"][0] += stats[0]["events_h"] + stats[1]["events_x"]
            ev["h1"][1] += stats[0]["dense_h"] + stats[1]["dense_x"]
            ev["h2"][0] += stats[1]["events_h"]
            ev["h2"][1] += stats[1]["dense_h"]
    rates = {k: v[0] / v[1] for k, v in ev.items()}
    return rates, correct / len(x)


# ------------------------------------------------------------------ LM E1


def run_lm_task():
    tr_ids, va_ids, te_ids, vocab = D.load_enwik8()
    gamma = SodCorrection.from_json(
        {"x_grid": load_json(RESULTS / "exp0_sod_gamma.json")["x_grid"],
         "gamma": load_json(RESULTS / "exp0_sod_gamma.json")["pooled_gamma"]})
    seq, B = 256, 64

    def windows(ids, seed):
        g = torch.Generator().manual_seed(seed)
        hi = len(ids) - seq - 1
        starts = torch.randint(0, hi, (B,), generator=g)
        return torch.stack([ids[s:s + seq] for s in starts])

    results = {}
    for seed in SEEDS:
        model = TransformerLM(vocab, dim=128, n_layers=2, n_heads=4, ffn=512,
                              seq_len=seq).to(DEVICE)
        model.load_state_dict(torch.load(CKPT / f"enwik8_base_s{seed}.pt",
                                         weights_only=True))
        model.eval()
        xcal = windows(va_ids, 111).to(DEVICE)
        xev = windows(te_ids, 222).to(DEVICE)

        model.set_thetas(0.0)
        with torch.no_grad():
            _, cal_tr = model(xcal, return_traces=True)
            _, ev_tr = model(xev, return_traces=True)
        cal_tr = {k: v.cpu() for k, v in cal_tr.items()}
        ev_tr = {k: v.cpu() for k, v in ev_tr.items()}

        profiles, moments = {}, {}
        for name in cal_tr:
            profiles[name], moments[name] = profile_analysis(cal_tr[name],
                                                             ev_tr[name])
            log(f"[enwik8 s{seed}] {name}: emp {profiles[name]['empirical']['median_rel_err']:.3f} "
                f"rice {profiles[name]['rice']['median_rel_err']:.3f} "
                f"iid {profiles[name]['iid']['median_rel_err']:.3f}")

        sd = {n: float(np.mean(m["sigma_delta"])) for n, m in moments.items()}
        event_rows = []
        for x in X_GRID:
            theta = [{k.split(".")[1][:-3]: x * sd[f"l{l}.{k.split('.')[1]}"]
                      for k in [f"l{l}.qkv_in", f"l{l}.out_in",
                                f"l{l}.fc1_in", f"l{l}.fc2_in"]}
                     for l in range(2)]
            model.set_thetas(theta)
            model.reset_event_counts()
            with torch.no_grad():
                logits = model(xev[:, :-1])
                nll = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    xev[:, 1:].reshape(-1))
            bpc = float(nll.item() / np.log(2.0))
            stats = model.event_stats()
            row = dict(x=x, bpc=bpc, streams={})
            for s in stats:
                name = f"l{s['layer']}.{s['name']}_in"
                th = x * sd[name]
                p_emp, p_gau = event_predictors(moments[name], th, gamma)
                row["streams"][name] = dict(
                    theta=th, measured=s["events_x"] / s["dense_x"],
                    pred_analytic_emp_tv=p_emp,
                    pred_analytic_gauss_tv=p_gau,
                    pred_openloop_cal=sod_sim_rate(cal_tr[name], th),
                    openloop_eval=sod_sim_rate(ev_tr[name], th))
            event_rows.append(row)
            log(f"[enwik8 s{seed}] x={x} l0.qkv meas "
                f"{row['streams']['l0.qkv_in']['measured']:.4f} pred "
                f"{row['streams']['l0.qkv_in']['pred_analytic_emp_tv']:.4f}")
        results[f"s{seed}"] = dict(profiles=profiles, events=event_rows)
    save_json(results, RESULTS / "exp1_enwik8.json")
    return results


def gate_p(all_results):
    """GATE P: predictor (b) median abs rel err <= 10% per task (pooled over
    streams, seeds)."""
    verdict = {}
    for task, res in all_results.items():
        errs = [p["empirical"]["median_rel_err"]
                for r in res.values() for p in r["profiles"].values()]
        rice = [p["rice"]["median_rel_err"]
                for r in res.values() for p in r["profiles"].values()]
        verdict[task] = dict(empirical_median=float(np.median(errs)),
                             empirical_max=float(np.max(errs)),
                             rice_median=float(np.median(rice)),
                             passed=bool(np.median(errs) <= 0.10))
    verdict["passed"] = all(v["passed"] for v in verdict.values()
                            if isinstance(v, dict))
    save_json(verdict, RESULTS / "exp1_gate_p.json")
    log(f"GATE P: {'PASS' if verdict['passed'] else 'FAIL'} {verdict}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all",
                   choices=["all", "sc2", "psmnist", "enwik8"])
    args = p.parse_args()
    all_results = {}
    if args.task in ("all", "sc2"):
        all_results["sc2"] = run_gru_task("sc2")
    if args.task in ("all", "psmnist"):
        all_results["psmnist"] = run_gru_task("psmnist")
    if args.task in ("all", "enwik8"):
        all_results["enwik8"] = run_lm_task()
    if args.task == "all":
        gate_p(all_results)


if __name__ == "__main__":
    main()
