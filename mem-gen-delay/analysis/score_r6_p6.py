"""P6 R6: one-shot scoring of the TRAP-LANGUAGE rung (prereg commit 4a5083f; constants
finalized at f032a48 after disclosed smoke iterations; seed 0 = smoke, excluded).

The modal-trigram language makes previous-token context pay for the task itself, so the
precursor should form even where induction cannot (norep). Frozen predictions:
  P-T1 (the trap is real): bare precursor rule (layer-0 prevtok >= 0.10) false-alarms on
        >= 8/10 trigram norep negatives.
  P-T2: conjunction (prevtok >= 0.10 AND indist_adv >= 0.10) restores FA <= 1/10.
  P-T3: among anchors {t_pv, t_ind, t_conj, t_prefix (first max-prefix >= 0.05)}, at
        least one achieves Spearman >= 0.5 AND median lead >= 300 AND FA <= 1/10.
  K-T:  no anchor does -> boundary; forecasting in trap languages needs new signals.
"""
import json
import os

import numpy as np

GRID = "runs/grid6r6"


def spearman(x, y):
    def rank(a):
        o = np.argsort(a)
        rk = np.empty(len(a))
        rk[o] = np.arange(len(a))
        return rk
    return float(np.corrcoef(rank(np.asarray(x, float)), rank(np.asarray(y, float)))[0, 1])


def load(name):
    summ = json.load(open(f"{GRID}/{name}/summary.json"))
    recs = [json.loads(l) for l in open(f"{GRID}/{name}/metrics.jsonl") if l.strip()]
    return summ, recs


def anchors(recs):
    t_pv = next((r["step"] for r in recs if r["prevtok_by_layer"][0] >= 0.10), None)
    t_ind = next((r["step"] for r in recs if r["indist_adv"] >= 0.10), None)
    t_conj = next((r["step"] for r in recs if r["prevtok_by_layer"][0] >= 0.10
                   and r["indist_adv"] >= 0.10), None)
    t_prefix = next((r["step"] for r in recs if max(r["prefix_by_layer"]) >= 0.05), None)
    return dict(t_pv=t_pv, t_ind=t_ind, t_conj=t_conj, t_prefix=t_prefix)


def main():
    pos = []
    for s in range(1, 11):
        summ, recs = load(f"rep_s{s}")
        a = anchors(recs)
        pos.append(dict(seed=s, t_event=summ["t_event"], **a))
        print(json.dumps(pos[-1]))
    neg = []
    for s in range(1, 11):
        summ, recs = load(f"norep_s{s}")
        a = anchors(recs)
        neg.append(dict(seed=s, t_event=summ["t_event"], **a))
        print(json.dumps(neg[-1]))

    res = dict(positives=pos, negatives=neg)
    # P-T1: bare precursor FA on negatives
    bare_fa = sum(1 for n in neg if n["t_pv"] is not None)
    res["P_T1"] = dict(bare_fa=f"{bare_fa}/10", passed=bool(bare_fa >= 8))
    # P-T2: conjunction FA
    conj_fa = sum(1 for n in neg if n["t_conj"] is not None)
    res["P_T2"] = dict(conj_fa=f"{conj_fa}/10", passed=bool(conj_fa <= 1))
    # P-T3: anchor race — (rho, median lead, FA) triples
    ev = [p["t_event"] for p in pos]
    assert all(e is not None for e in ev)
    table = {}
    any_pass = False
    for key in ("t_pv", "t_ind", "t_conj", "t_prefix"):
        vals = [p[key] for p in pos]
        fa = sum(1 for n in neg if n[key] is not None)
        if any(v is None for v in vals):
            table[key] = dict(note=f"missing on {sum(1 for v in vals if v is None)}/10 positives",
                              fa=f"{fa}/10", qualifies=False)
            continue
        rho = spearman(vals, ev)
        leads = [e - v for e, v in zip(ev, vals)]
        med = float(np.median(leads))
        q = bool(rho >= 0.5 and med >= 300 and fa <= 1)
        any_pass = any_pass or q
        table[key] = dict(rho=round(rho, 4), median_lead=med,
                          lead_range=[min(leads), max(leads)], fa=f"{fa}/10",
                          qualifies=q)
    res["P_T3"] = dict(anchors=table, passed=bool(any_pass))
    res["K_T_fires"] = not any_pass
    print("\nP-T1:", res["P_T1"], "\nP-T2:", res["P_T2"])
    for k, v in table.items():
        print(f"  {k}: {json.dumps(v)}")
    print("P-T3 passed:", res["P_T3"]["passed"], "| K-T fires:", res["K_T_fires"])
    os.makedirs("analysis/out6", exist_ok=True)
    json.dump(res, open("analysis/out6/r6_scored.json", "w"), indent=2)


if __name__ == "__main__":
    main()
