"""P6 R4: split-conformal emergence-time intervals anchored at the precursor alarm.

Spec frozen in plan_p6.md (calibration seeds 1-15, test seeds 16-30, alpha=0.1).
Methods: OFFSET (t_pv + median lead) and LINEAR (t_event ~ a + b t_pv), both wrapped with
split-conformal absolute-residual intervals. Loss-anchored contrast included: accurate but
issued ~concurrently with the event. K4: widths on the order of the 16,000-step budget.
"""
import json
import os

import numpy as np

GRID = "runs/grid6r2"
ALPHA = 0.1
LOSS_THETA = 2.0769


def load_seed(s):
    summ = json.load(open(f"{GRID}/rep_s{s}/summary.json"))
    recs = [json.loads(l) for l in open(f"{GRID}/rep_s{s}/metrics.jsonl") if l.strip()]
    t_pv = next(r["step"] for r in recs if r["prevtok_by_layer"][0] >= 0.10)
    t_loss = next(r["step"] for r in recs if r["train_loss"] <= LOSS_THETA)
    return dict(seed=s, t_event=summ["t_event"], t_pv=t_pv, t_loss=t_loss)


def conformal_q(residuals, alpha):
    n = len(residuals)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(np.abs(residuals))[min(k, n) - 1])


def evaluate(cal, test, anchor_key, method):
    a = np.array([r[anchor_key] for r in cal], float)
    y = np.array([r["t_event"] for r in cal], float)
    if method == "offset":
        delta = float(np.median(y - a))
        pred = lambda x: x + delta
        params = dict(delta=delta)
    else:
        b, c = np.polyfit(a, y, 1)
        pred = lambda x: b * x + c
        params = dict(slope=float(b), intercept=float(c))
    q = conformal_q(y - np.array([pred(x) for x in a]), ALPHA)
    cover, widths, leads = 0, [], []
    rows = []
    for r in test:
        p = pred(r[anchor_key])
        lo, hi = p - q, p + q
        cover += lo <= r["t_event"] <= hi
        widths.append(hi - lo)
        leads.append(r["t_event"] - r[anchor_key])
        rows.append(dict(seed=r["seed"], anchor=r[anchor_key], point=round(p, 1),
                         lo=round(lo, 1), hi=round(hi, 1), t_event=r["t_event"],
                         covered=bool(lo <= r["t_event"] <= hi)))
    return dict(method=method, anchor=anchor_key, params=params, q=q,
                coverage=f"{cover}/{len(test)}", coverage_frac=cover / len(test),
                median_width=float(np.median(widths)),
                median_anchor_lead=float(np.median(leads)), test_rows=rows)


def main():
    data = [load_seed(s) for s in range(1, 31)]
    cal = [d for d in data if d["seed"] <= 15]
    test = [d for d in data if d["seed"] >= 16]
    out = {}
    for anchor in ("t_pv", "t_loss"):
        for method in ("offset", "linear"):
            r = evaluate(cal, test, anchor, method)
            out[f"{anchor}_{method}"] = r
            print(f"{anchor:7s} {method:6s}: coverage {r['coverage']} "
                  f"(nominal 90%), median width {r['median_width']:.0f} steps, "
                  f"median anchor lead {r['median_anchor_lead']:.0f}")
    budget = 16000
    k4 = all(v["median_width"] > 0.5 * budget for v in out.values()
             if v["anchor"] == "t_pv")
    out["K4_fires"] = bool(k4)
    print(f"\nK4 (precursor intervals vacuous ~budget) fires: {k4}")
    os.makedirs("analysis/out6", exist_ok=True)
    json.dump(out, open("analysis/out6/r4_scored.json", "w"), indent=2)


if __name__ == "__main__":
    main()
