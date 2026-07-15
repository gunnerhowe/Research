"""P5 R2b: mechanism-factored two-gate forecaster (pre-registered, commit 5595d3e).

Alarm iff CONTENT gate fires AND the norm-VIABILITY gate is open at the same eval.
Viability windows are mechanism constants fixed a priori in the prereg (NOT fit):
  algorithmic: ||W|| in [40, inf)   (c35 inversion traps live below 40; Papers 2-3)
  mnist:       ||W|| in (0, 110]    (wrong-structure norm explosion above ~110; Paper 4)
Only the content-signal threshold is fit, train-side, at FA <= 5% with the veto active.
Evaluation is one-shot: alg train=even -> TEST=odd (the domain that was infeasible);
mnist train=control-even -> TEST=prior arms (the shift where the multivariate collapsed).
K8: alg still infeasible -> boundary stands. Prediction (ii): mnist two-gate retains
>= 50% of cos_gap's shift lead (>= 2,200) at lower FA.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict_emergence as pe

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "analysis", "out5")
WINDOW = {"alg": (40.0, np.inf), "mnist": (0.0, 110.0)}
CONTENT = {"alg": ["fourier_top8", "fourier_gini", "cos_gap", "fisher",
                   "d.fourier_top8", "d.cos_gap", "d.fisher"],
           "mnist": ["cos_gap", "d.cos_gap"]}
FA_CAP = 0.05


def alarm_time_veto(run, sig, direction, theta, lo, hi):
    x = run["_sigs"].get(sig)
    w = run["_sigs"].get("wnorm")
    if x is None or w is None:
        return None
    for i in range(pe.WARMUP, len(x)):
        if not (np.isfinite(x[i]) and lo <= w[i] <= hi):
            continue
        if (direction == ">=" and x[i] >= theta) or (direction == "<=" and x[i] <= theta):
            return run["times"][i]
    return None


def score(runs, sig, direction, theta, lo, hi):
    leads, miss, fa, npos, nneg = [], 0, 0, 0, 0
    for r in runs:
        ta = alarm_time_veto(r, sig, direction, theta, lo, hi)
        if r["t_gen"] is not None:
            npos += 1
            l = 0.0 if ta is None else max(0.0, r["t_gen"] - ta)
            miss += int(ta is None)
            leads.append(l)
        else:
            nneg += 1
            fa += int(ta is not None)
    return dict(median_lead=float(np.median(leads)) if leads else 0.0,
                median_rel=float(np.median([l / r["t_gen"] for l, r in zip(
                    leads, [r for r in runs if r["t_gen"] is not None])])) if leads else 0.0,
                miss=miss / npos if npos else None, fa=fa / nneg if nneg else None,
                n_pos=npos, n_neg=nneg)


def fit(train, sig, lo, hi):
    vals = np.concatenate([r["_sigs"][sig][pe.WARMUP:] for r in train
                           if r["_sigs"].get(sig) is not None])
    vals = vals[np.isfinite(vals)]
    if len(vals) < 50:
        return None
    best = None
    for direction in (">=", "<="):
        for theta in np.quantile(vals, np.linspace(0.02, 0.98, 49)):
            sc = score(train, sig, direction, float(theta), lo, hi)
            if sc["fa"] is not None and sc["fa"] > FA_CAP:
                continue
            if best is None or sc["median_lead"] > best[0]:
                best = (sc["median_lead"], direction, float(theta), sc)
    return best


def block(domain, train, test, label):
    lo, hi = WINDOW[domain]
    print(f"\n=== R2b two-gate [{domain}] {label} | window [{lo}, {hi}] ===")
    print(f"train n={len(train)} (neg {sum(1 for r in train if r['t_gen'] is None)}); "
          f"test n={len(test)} (neg {sum(1 for r in test if r['t_gen'] is None)})")
    rows = []
    for sig in CONTENT[domain]:
        f = fit(train, sig, lo, hi)
        if f is None:
            print(f"  {sig:16s} no feasible threshold under veto")
            continue
        trlead, d, th, tr = f
        te = score(test, sig, d, th, lo, hi)
        rows.append(dict(signal=sig, dir=d, theta=round(th, 5),
                         train_lead=round(trlead, 1),
                         test_lead=round(te["median_lead"], 1),
                         test_rel=round(te["median_rel"], 4),
                         test_miss=te["miss"], test_fa=te["fa"],
                         n_pos=te["n_pos"], n_neg=te["n_neg"]))
        print(f"  {sig:16s} {d}{th:.4g}: train lead {trlead:.0f} -> TEST lead "
              f"{te['median_lead']:.0f} (rel {te['median_rel']:.3f}) miss {te['miss']:.2f} "
              f"FA {te['fa']}")
    champ = max(rows, key=lambda r: r["train_lead"]) if rows else None
    if champ:
        print(f"  CHAMPION (train-selected): {champ['signal']} -> TEST lead "
              f"{champ['test_lead']:.0f}, miss {champ['test_miss']:.2f}, FA {champ['test_fa']}")
    return dict(rows=rows, champion=champ)


def main():
    runs = pe.load_corpus()
    for r in runs:
        r["_sigs"] = pe.add_slopes(r)
    out = {}
    alg = [r for r in runs if r["domain"] == "alg"]
    out["alg"] = block("alg",
                       [r for r in alg if r["seed"] % 2 == 0],
                       [r for r in alg if r["seed"] % 2 == 1],
                       "train=even, TEST=odd (previously infeasible)")
    mn = [r for r in runs if r["domain"] == "mnist"]
    out["mnist"] = block("mnist",
                         [r for r in mn if r["family"] == "control" and r["seed"] % 2 == 0],
                         [r for r in mn if r["family"] == "prior"],
                         "train=control-even, TEST=prior arms (shift)")
    json.dump(out, open(os.path.join(OUT, "r2b_twogate.json"), "w"), indent=2)
    ch = out["alg"]["champion"]
    print(f"\nK8 (alg still infeasible) fires: {ch is None or ch['test_lead'] <= 0}")


if __name__ == "__main__":
    main()
