"""P5 R1/R2: emergence-timing forecasting benchmark on the existing run corpus.

Spec pre-registered in plan_p5.md (commit 2cf62e3) BEFORE this file existed. Protocol:
threshold-alarm forecasters per signal; thresholds + directions fit on TRAIN runs (even
seeds) subject to false-alarm rate <= 5% on train negatives (never-generalizing runs);
all reported numbers from TEST runs (odd seeds). Primary metric: median lead time
(t_gen - t_alarm, misses score 0) at the <=5%-FA operating point. R2: train on
control-family runs only, test on content-bearing prior arms (interventional shift).

Implementation details fixed before any test-block inspection (disclosed per D3):
- alarm eligibility starts at eval index 5 (init transients: logit_scale ~40 at 8x init);
- MNIST logs contain no CE, so the oscillation feature there uses train_acc;
- signals missing from a run's logs are skipped for that run;
- candidate thresholds = 2..98% quantiles (49 points) of pooled train values, both
  directions; ties broken toward the larger train median lead.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out5")
GRIDS = ["grid", "grid2", "grid3", "grid4", "grid4b"]
WARMUP = 5
OSC_WIN = 50
SLOPE_K = 5
FA_CAP = 0.05

SIG_S = {"alg": ["wnorm", "train_ce", "test_ce", "logit_scale", "conf", "osc"],
         "mnist": ["wnorm", "logit_scale", "conf", "osc"]}
SIG_R = {"alg": ["fourier_top8", "fourier_gini", "cos_gap", "fisher"],
         "mnist": ["cos_gap"]}

# R2 family map: content-bearing prior arms are the interventional TEST set; everything
# else (incl. wrong/shuffled-content aux arms) is control-family TRAIN, per plan_p5.md.
PRIOR_TOKENS = ("supcon_true", "comm", "supcon_aug", "supcon_label", "supcon_nn")


def load_corpus():
    runs = []
    for grid in GRIDS:
        gdir = os.path.join(ROOT, "runs", grid)
        for name in sorted(os.listdir(gdir)):
            sp = os.path.join(gdir, name, "summary.json")
            mp = os.path.join(gdir, name, "metrics.jsonl")
            if not (os.path.exists(sp) and os.path.exists(mp)):
                continue
            s = json.load(open(sp))
            recs = [json.loads(l) for l in open(mp) if l.strip()]
            if not recs or "test_agr" in recs[0]:      # exclude sequence-grammar runs
                continue
            domain = "alg" if "epoch" in recs[0] else "mnist"
            tkey = "epoch" if domain == "alg" else "step"
            times = np.array([r[tkey] for r in recs], float)
            sigs = {}
            for k in set(SIG_S[domain] + SIG_R[domain]) - {"osc"}:
                if k in recs[0]:
                    sigs[k] = np.array([r.get(k, np.nan) for r in recs], float)
            # oscillation feature (Notsawo-style): trailing-window detrended std
            base = "train_ce" if domain == "alg" else "train_acc"
            x = np.array([r.get(base, np.nan) for r in recs], float)
            osc = np.full(len(x), np.nan)
            for i in range(len(x)):
                lo = max(0, i - OSC_WIN + 1)
                w = x[lo:i + 1]
                if len(w) >= 10 and np.isfinite(w).all():
                    t = np.arange(len(w), dtype=float)
                    A = np.vstack([t, np.ones_like(t)]).T
                    coef, *_ = np.linalg.lstsq(A, w, rcond=None)
                    osc[i] = float(np.std(w - A @ coef))
            sigs["osc"] = osc
            cond = str(s.get("condition", ""))
            runs.append(dict(
                grid=grid, name=name, domain=domain, seed=int(s.get("seed", 0)),
                condition=cond, task=s.get("task", "add"),
                family="prior" if any(t in cond or t in name for t in PRIOR_TOKENS)
                       else "control",
                t_gen=s.get("t_gen"), t_end=float(times[-1]),
                times=times, sigs=sigs))
    return runs


def add_slopes(run):
    out = {}
    for k, v in run["sigs"].items():
        out[k] = v
        d = np.full(len(v), np.nan)
        d[SLOPE_K:] = (v[SLOPE_K:] - v[:-SLOPE_K])
        out["d." + k] = d
    return out


def alarm_time(times, x, direction, theta):
    """First eligible eval where the signal crosses; None if never."""
    for i in range(WARMUP, len(x)):
        if not np.isfinite(x[i]):
            continue
        if (direction == ">=" and x[i] >= theta) or (direction == "<=" and x[i] <= theta):
            return times[i]
    return None


def score(runs, sig, direction, theta):
    leads, misses, fa, npos, nneg = [], 0, 0, 0, 0
    for r in runs:
        x = r["_sigs"].get(sig)
        if x is None:
            continue
        ta = alarm_time(r["times"], x, direction, theta)
        if r["t_gen"] is not None:
            npos += 1
            if ta is None:
                misses += 1
                leads.append(0.0)
            else:
                leads.append(max(0.0, r["t_gen"] - ta))
        else:
            nneg += 1
            if ta is not None:
                fa += 1
    med = float(np.median(leads)) if leads else 0.0
    rel = float(np.median([l / r["t_gen"] for l, r in
                           zip(leads, [r for r in runs if r["t_gen"] is not None
                                       and r["_sigs"].get(sig) is not None])])) if leads else 0.0
    return dict(median_lead=med, median_rel_lead=rel,
                miss_rate=misses / npos if npos else None,
                fa_rate=fa / nneg if nneg else None, n_pos=npos, n_neg=nneg)


def fit_threshold(train_runs, sig):
    vals = np.concatenate([r["_sigs"][sig][WARMUP:] for r in train_runs
                           if r["_sigs"].get(sig) is not None])
    vals = vals[np.isfinite(vals)]
    if len(vals) < 50:
        return None
    qs = np.quantile(vals, np.linspace(0.02, 0.98, 49))
    best = None
    for direction in (">=", "<="):
        for theta in qs:
            sc = score(train_runs, sig, direction, float(theta))
            if sc["fa_rate"] is not None and sc["fa_rate"] > FA_CAP:
                continue
            if sc["n_pos"] == 0:
                continue
            key = (sc["median_lead"],)
            if best is None or key > best[0]:
                best = (key, direction, float(theta), sc)
    return best


def run_block(title, train, test, domain):
    print(f"\n{'=' * 100}\n{title}  [{domain}]  train n={len(train)} "
          f"(pos {sum(1 for r in train if r['t_gen'] is not None)}, "
          f"neg {sum(1 for r in train if r['t_gen'] is None)}); "
          f"test n={len(test)} "
          f"(pos {sum(1 for r in test if r['t_gen'] is not None)}, "
          f"neg {sum(1 for r in test if r['t_gen'] is None)})\n{'=' * 100}")
    rows = []
    sigsets = {"S": SIG_S[domain], "R": SIG_R[domain]}
    for setname, base_sigs in sigsets.items():
        for sig in [s for b in base_sigs for s in (b, "d." + b)]:
            fit = fit_threshold(train, sig)
            if fit is None:
                continue
            _, direction, theta, tr_sc = fit
            te_sc = score(test, sig, direction, theta)
            rows.append(dict(set=setname, signal=sig, dir=direction, theta=round(theta, 5),
                             train_med_lead=round(tr_sc["median_lead"], 1),
                             test_med_lead=round(te_sc["median_lead"], 1),
                             test_rel_lead=round(te_sc["median_rel_lead"], 4),
                             test_miss=round(te_sc["miss_rate"], 3) if te_sc["miss_rate"] is not None else None,
                             test_fa=round(te_sc["fa_rate"], 3) if te_sc["fa_rate"] is not None else None,
                             n_pos=te_sc["n_pos"], n_neg=te_sc["n_neg"]))
    rows.sort(key=lambda r: -r["train_med_lead"])
    hdr = f"{'set':3s} {'signal':16s} {'dir':3s} {'theta':>10s} {'trainLead':>10s} " \
          f"{'TESTlead':>10s} {'rel':>7s} {'miss':>6s} {'FA':>6s} {'nPos':>5s} {'nNeg':>5s}"
    print(hdr)
    for r in rows:
        print(f"{r['set']:3s} {r['signal']:16s} {r['dir']:3s} {r['theta']:>10.4g} "
              f"{r['train_med_lead']:>10.0f} {r['test_med_lead']:>10.0f} "
              f"{r['test_rel_lead']:>7.3f} {str(r['test_miss']):>6s} {str(r['test_fa']):>6s} "
              f"{r['n_pos']:>5d} {r['n_neg']:>5d}")
    # champions chosen on TRAIN lead (no test peeking), compared on TEST
    champ = {}
    for setname in ("S", "R"):
        cand = [r for r in rows if r["set"] == setname]
        if cand:
            champ[setname] = max(cand, key=lambda r: r["train_med_lead"])
    if "S" in champ and "R" in champ:
        s, p = champ["S"], champ["R"]
        gain = (p["test_med_lead"] - s["test_med_lead"]) / s["test_med_lead"] if s["test_med_lead"] else float("inf")
        print(f"\nCHAMPIONS (train-selected): S={s['signal']} testLead={s['test_med_lead']:.0f} | "
              f"R={p['signal']} testLead={p['test_med_lead']:.0f} | R-vs-S gain={gain:+.1%}")
    return rows, champ


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = load_corpus()
    for r in runs:
        r["_sigs"] = add_slopes(r)
    print(f"corpus: {len(runs)} runs "
          f"(alg {sum(1 for r in runs if r['domain'] == 'alg')}, "
          f"mnist {sum(1 for r in runs if r['domain'] == 'mnist')}); "
          f"negatives {sum(1 for r in runs if r['t_gen'] is None)}")
    from collections import Counter
    print("family x domain:", Counter((r["domain"], r["family"]) for r in runs))

    results = {}
    for domain in ("alg", "mnist"):
        dom = [r for r in runs if r["domain"] == domain]
        # ---- R1: even/odd seed split ----
        train = [r for r in dom if r["seed"] % 2 == 0]
        test = [r for r in dom if r["seed"] % 2 == 1]
        rows, champ = run_block("R1 in-distribution (train=even seeds, test=odd seeds)",
                                train, test, domain)
        results[f"r1_{domain}"] = dict(rows=rows,
                                       champions={k: v["signal"] for k, v in champ.items()})
        # ---- R2: interventional shift ----
        ctrl_even = [r for r in dom if r["family"] == "control" and r["seed"] % 2 == 0]
        prior_all = [r for r in dom if r["family"] == "prior"]
        rows2, champ2 = run_block("R2 interventional shift (train=control even, test=PRIOR arms)",
                                  ctrl_even, prior_all, domain)
        results[f"r2_{domain}"] = dict(rows=rows2,
                                       champions={k: v["signal"] for k, v in champ2.items()})
    json.dump(results, open(os.path.join(OUT, "r1r2_stats.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(OUT, 'r1r2_stats.json')}")


if __name__ == "__main__":
    main()
