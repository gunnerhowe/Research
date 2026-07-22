"""P8-R2b sealed one-shot scorer (prereg in plan_p8.md, commit 69f4f00). Refuses on
partial fleet or existing scorefile. Adjudicates the TRUE-DENIAL arm:
  VOID RULE : step-0 max-over-heads L0 prevtok must be < 0.15 on 3/3, else arm VOID.
  P-R2c'    : burned2 median t_reevent >= 300 (2.0x the faithful median of 150).
  K-R2c'    : fires if burned2 median <= 187.5 (1.25x) WITH the manipulation valid.
  P-R2d'    : burned2 median anchor lead >= 300.
Anchor (frozen, unchanged): first eval with prevtok_by_layer[0] >= 0.10 AND
indist_adv >= 0.10. Faithful baseline reused from p8r2_scored.json (median 150).
"""
import json
import os

import numpy as np

FAITHFUL_MEDIAN = 150.0
SEEDS = (501, 502, 503)


def main():
    outp = "analysis/out8/p8r2b_scored.json"
    if os.path.exists(outp):
        raise RuntimeError("p8r2b_scored.json exists — one-shot already ran")
    runs = {}
    for s in SEEDS:
        d = f"runs/grid8r2/reburn2_s{s}"
        if not os.path.exists(f"{d}/summary.json"):
            raise RuntimeError(f"fleet incomplete: reburn2_s{s}")
        recs = [json.loads(l) for l in open(f"{d}/metrics.jsonl")]
        summ = json.load(open(f"{d}/summary.json"))
        ta = next((r["step"] for r in recs
                   if r["prevtok_by_layer"][0] >= 0.10 and r["indist_adv"] >= 0.10), None)
        te = summ["t_event"]
        runs[s] = dict(
            prevtok0_start=recs[0]["prevtok_by_layer"][0],
            t_reevent=te, t_anchor=ta,
            lead=(te - ta) if (te is not None and ta is not None) else None,
            heads_start=[round(x, 3) for x in recs[0]["prevtok_by_head"][0]])

    manip_ok = all(runs[s]["prevtok0_start"] < 0.15 for s in SEEDS)
    tes = [runs[s]["t_reevent"] for s in SEEDS if runs[s]["t_reevent"] is not None]
    mb = float(np.median(tes)) if tes else None
    leads = [runs[s]["lead"] for s in SEEDS if runs[s]["lead"] is not None]
    ml = float(np.median(leads)) if leads else None

    v = {}
    v["manipulation"] = (f"{'VALID' if manip_ok else 'VOID — arm not scored'} "
                         f"(step-0 prevtok {[round(runs[s]['prevtok0_start'], 3) for s in SEEDS]})")
    if manip_ok:
        ratio = (mb / FAITHFUL_MEDIAN) if mb else None
        v["P_R2c_prime_compensating_law"] = (
            f"{'PASS' if (mb is not None and mb >= 300) else 'FAIL'} "
            f"(burned2 median {mb}, faithful {FAITHFUL_MEDIAN}, "
            f"ratio {None if ratio is None else round(ratio, 2)}; evented {len(tes)}/3)")
        v["K_R2c_prime_uncovered_cell"] = (
            "FIRES — fast return even from genuine scaffold denial"
            if (mb is not None and mb <= 187.5) else "no-fire")
        v["P_R2d_prime_watchable_rebuild"] = (
            f"{'PASS' if (ml is not None and ml >= 300) else 'FAIL'} (median lead {ml})")
    res = dict(faithful_median=FAITHFUL_MEDIAN, runs={str(s): runs[s] for s in SEEDS},
               verdicts=v)
    json.dump(res, open(outp, "w"), indent=1)
    print(json.dumps(v, indent=1))
    for s in SEEDS:
        r = runs[s]
        print(f"  reburn2_s{s}: start prevtok {r['prevtok0_start']:.3f} heads {r['heads_start']} "
              f"t_reevent={r['t_reevent']} anchor={r['t_anchor']} lead={r['lead']}")


if __name__ == "__main__":
    main()
