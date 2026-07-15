"""P5 R3: score the frozen forecasters on the prospective runs (grid5r3).

Frozen artifacts (committed 5595d3e, BEFORE grid5r3 existed):
  A) R1-SRq forecaster: weights deterministically refit from committed code on grid4/4b
     even seeds (all families), tau = frozen_eval.json r1.tau, W = r1.W.
  B) R2-SRq forecaster: control-even training pool, tau/W from frozen_eval.json r2.
  C) Single-signal thresholds from out5/r1r2_stats.json r2_mnist rows (control-even):
     cos_gap and d.cos_gap.

Predictions under test (STATUS prereg):
  P-R3a  A's median lead on new baseline-family positives >= 5,200 (50% of 10,400); K4 below.
  P-R3b  C(cos_gap)'s median lead on new prior arms >= 2,200 (50% of 4,400).
  P-R3c  B misses >= 40% of new prior arms (the robustness-vs-FA tradeoff, prospectively).
  P-R3d  structural negatives (shufpair s5-7): zero alarms from A, B, and C.
         c92 runs exempt (censoring-noise class), reported descriptively.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict_emergence as pe
from predict_emergence_ml import quad_names, build_dataset, predict_run
from tune_ml_trainside import fit_logistic_w

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "analysis", "out5")


def load(grids):
    saved = pe.GRIDS
    pe.GRIDS = grids
    runs = [r for r in pe.load_corpus() if r["domain"] == "mnist"]
    pe.GRIDS = saved
    for r in runs:
        r["_sigs"] = pe.add_slopes(r)
    return runs


def srq_model(train, W):
    names = quad_names("mnist", "SR")
    X, y = build_dataset(train, names, W)
    w, mu, sd = fit_logistic_w(X, y, 20, 1e-3, 8000, 0.3)
    return names, w, mu, sd


def srq_alarm(run, model, tau):
    names, w, mu, sd = model
    p = predict_run(run, names, w, mu, sd)
    for i in range(pe.WARMUP, len(p)):
        if p[i] >= tau:
            return run["times"][i]
    return None


def sig_alarm(run, sig, direction, theta):
    x = run["_sigs"].get(sig)
    return None if x is None else pe.alarm_time(run["times"], x, direction, theta)


def lead(run, ta):
    if run["t_gen"] is None:
        return None
    return 0.0 if ta is None else max(0.0, run["t_gen"] - ta)


def main():
    fz = json.load(open(os.path.join(OUT, "frozen_eval.json")))
    stats = json.load(open(os.path.join(OUT, "r1r2_stats.json")))
    thr = {r["signal"]: (r["dir"], r["theta"]) for r in stats["r2_mnist"]["rows"]}

    hist = load(["grid4", "grid4b"])
    new = load(["grid5r3"])
    A = srq_model([r for r in hist if r["seed"] % 2 == 0], fz["r1"]["W"])
    B = srq_model([r for r in hist if r["family"] == "control" and r["seed"] % 2 == 0],
                  fz["r2"]["W"])
    tauA, tauB = fz["r1"]["tau"], fz["r2"]["tau"]

    base_fam = [r for r in new if r["family"] == "control" and "c92" not in r["name"]
                and r["t_gen"] is not None]
    prior = [r for r in new if r["family"] == "prior"]
    shuf = [r for r in new if "shufpair" in r["name"]]
    c92 = [r for r in new if "c92" in r["name"]]

    print(f"new runs: {len(new)} (baseline-family pos {len(base_fam)}, prior {len(prior)}, "
          f"structural neg {len(shuf)}, censoring-noise {len(c92)})")
    res = {}

    # P-R3a
    leadsA = [lead(r, srq_alarm(r, A, tauA)) for r in base_fam]
    res["P_R3a"] = dict(leads={r["name"]: l for r, l in zip(base_fam, leadsA)},
                        median=float(np.median(leadsA)), bar=5200,
                        passed=bool(np.median(leadsA) >= 5200))
    # P-R3b (prior arms that grok)
    ppos = [r for r in prior if r["t_gen"] is not None]
    d, th = thr["cos_gap"]
    leadsC = [lead(r, sig_alarm(r, "cos_gap", d, th)) for r in ppos]
    res["P_R3b"] = dict(leads={r["name"]: l for r, l in zip(ppos, leadsC)},
                        median=float(np.median(leadsC)), bar=2200,
                        passed=bool(np.median(leadsC) >= 2200))
    d2, th2 = thr["d.cos_gap"]
    leadsC2 = [lead(r, sig_alarm(r, "d.cos_gap", d2, th2)) for r in ppos]
    res["dcos_gap_secondary"] = dict(median=float(np.median(leadsC2)))
    # P-R3c
    missB = [srq_alarm(r, B, tauB) is None or
             (r["t_gen"] is not None and srq_alarm(r, B, tauB) > r["t_gen"]) for r in ppos]
    res["P_R3c"] = dict(miss_rate=float(np.mean(missB)), bar=0.40,
                        passed=bool(np.mean(missB) >= 0.40))
    # P-R3d
    alarms = {}
    for r in shuf:
        alarms[r["name"]] = dict(
            A=srq_alarm(r, A, tauA) is not None,
            B=srq_alarm(r, B, tauB) is not None,
            C=sig_alarm(r, "cos_gap", d, th) is not None)
    res["P_R3d"] = dict(alarms=alarms,
                        passed=not any(v for a in alarms.values() for v in a.values()))
    # c92 descriptive
    res["c92_descriptive"] = {r["name"]: dict(
        A=srq_alarm(r, A, tauA) is not None, B=srq_alarm(r, B, tauB) is not None,
        C=sig_alarm(r, "cos_gap", d, th) is not None,
        t_gen=r["t_gen"]) for r in c92}

    for k in ("P_R3a", "P_R3b", "P_R3c", "P_R3d"):
        print(f"\n{k}: {json.dumps(res[k], indent=1)}")
    print(f"\nc92 (exempt): {json.dumps(res['c92_descriptive'])}")
    print(f"\nd.cos_gap secondary median lead: {res['dcos_gap_secondary']['median']:.0f}")
    json.dump(res, open(os.path.join(OUT, "r3_scored.json"), "w"), indent=2)
    print(f"\nK4 fires: {not res['P_R3a']['passed']}")


if __name__ == "__main__":
    main()
