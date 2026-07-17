"""P6 R7: one-shot scoring of the THIRD-AXIS gate (new language, lang_seed 888).

Frozen forecaster (unchanged since commit 377511b, calibrated on the ORIGINAL language's
fleet): alarm at first eval with layer-0 prevtok >= 0.10; interval
[t_alarm + 825, t_alarm + 1125]. Event: copy_adv >= 2.0 x 2 consecutive (in summary).
Pre-registered (commit 4a5083f) bars, mirroring R5:
  P7a: >= 9/10 positives event, 0/3 negatives event.
  P7b: alarm strictly pre-event >= 9/10; conjunction FA 0/3.
  P7c: interval covers t_event on >= 7/10.
  secondary: Spearman(t_pv, t_event) >= 0.5.
  K7-gate: coverage <= 5/10 OR pre-event <= 7/10 -> gap constant is language-local.
"""
import json
import os

import numpy as np

GRID = "runs/grid6r7"
LO, HI = 825, 1125
POS = range(201, 211)
NEG = range(201, 204)


def spearman(x, y):
    def rank(a):
        o = np.argsort(a)
        rk = np.empty(len(a))
        rk[o] = np.arange(len(a))
        return rk
    return float(np.corrcoef(rank(np.asarray(x, float)), rank(np.asarray(y, float)))[0, 1])


def main():
    rows = []
    for s in POS:
        summ = json.load(open(f"{GRID}/rep_s{s}/summary.json"))
        recs = [json.loads(l) for l in open(f"{GRID}/rep_s{s}/metrics.jsonl") if l.strip()]
        t_pv = next((r["step"] for r in recs if r["prevtok_by_layer"][0] >= 0.10), None)
        ev = summ["t_event"]
        covered = (ev is not None and t_pv is not None and t_pv + LO <= ev <= t_pv + HI)
        rows.append(dict(seed=s, t_event=ev, t_pv=t_pv,
                         interval=[None if t_pv is None else t_pv + LO,
                                   None if t_pv is None else t_pv + HI],
                         lead=None if (ev is None or t_pv is None) else ev - t_pv,
                         pre_event=bool(t_pv is not None and ev is not None and t_pv < ev),
                         covered=bool(covered)))
        print(json.dumps(rows[-1]))
    neg = []
    for s in NEG:
        summ = json.load(open(f"{GRID}/norep_s{s}/summary.json"))
        recs = [json.loads(l) for l in open(f"{GRID}/norep_s{s}/metrics.jsonl") if l.strip()]
        cj = next((r["step"] for r in recs if r["prevtok_by_layer"][0] >= 0.10
                   and r["indist_adv"] >= 0.10), None)
        bare = next((r["step"] for r in recs if r["prevtok_by_layer"][0] >= 0.10), None)
        neg.append(dict(seed=s, t_event=summ["t_event"], conj_alarm=cj, bare_alarm=bare))
        print(json.dumps(neg[-1]))
    n_event = sum(1 for r in rows if r["t_event"] is not None)
    n_pre = sum(1 for r in rows if r["pre_event"])
    n_cov = sum(1 for r in rows if r["covered"])
    ok = [r for r in rows if r["t_event"] is not None and r["t_pv"] is not None]
    rho = spearman([r["t_pv"] for r in ok], [r["t_event"] for r in ok]) if len(ok) >= 3 else None
    leads = [r["lead"] for r in rows if r["lead"] is not None]
    res = dict(
        rows=rows, negatives=neg,
        P7a=dict(events=f"{n_event}/10",
                 neg_events=sum(1 for n in neg if n["t_event"] is not None),
                 passed=bool(n_event >= 9 and all(n["t_event"] is None for n in neg))),
        P7b=dict(pre_event=f"{n_pre}/10",
                 conj_fa=sum(1 for n in neg if n["conj_alarm"] is not None),
                 bare_fa=sum(1 for n in neg if n["bare_alarm"] is not None),
                 passed=bool(n_pre >= 9 and all(n["conj_alarm"] is None for n in neg))),
        P7c=dict(coverage=f"{n_cov}/10", passed=bool(n_cov >= 7)),
        secondary_spearman=None if rho is None else round(rho, 4),
        median_lead=float(np.median(leads)) if leads else None,
        K7_gate_fires=bool(n_cov <= 5 or n_pre <= 7))
    print("\nP7a:", res["P7a"], "\nP7b:", res["P7b"], "\nP7c:", res["P7c"])
    print("secondary Spearman:", res["secondary_spearman"],
          "| median lead:", res["median_lead"], "| K7-gate fires:", res["K7_gate_fires"])
    os.makedirs("analysis/out6", exist_ok=True)
    json.dump(res, open("analysis/out6/r7_scored.json", "w"), indent=2)


if __name__ == "__main__":
    main()
