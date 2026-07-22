"""P8-R2 sealed one-shot scorer (prereg plan_p8.md commit b127fa1). Refuses on partial
fleet or existing scorefile. Adjudicates P-R2a..d / K-R2a..c.

T0 = 6,300 (first-emergence median, grid6r2 rep n=30, reused). Frozen rules:
  event      : copy_adv >= 2.0 sustain 2 (in-run t_event of each reteach run)
  anchor     : first eval with prevtok_by_layer[0] >= 0.10 AND indist_adv >= 0.10
  scaffold   : layer-0 max-over-heads prevtok (prevtok_by_layer[0])
  guard valid: copy_adv <= 0.25 AND indist_adv <= 0.05 over last 5 evals
"""
import json
import os

import numpy as np

T0 = 6300.0
GRID = "runs/grid8r2"
SEEDS = (501, 502, 503)


def load(name):
    d = f"{GRID}/{name}"
    recs = [json.loads(l) for l in open(f"{d}/metrics.jsonl") if l.strip()]
    summ = json.load(open(f"{d}/summary.json"))
    return recs, summ


def anchor_step(recs):
    for r in recs:
        if r["prevtok_by_layer"][0] >= 0.10 and r["indist_adv"] >= 0.10:
            return r["step"]
    return None


def main():
    os.makedirs("analysis/out8", exist_ok=True)
    outp = "analysis/out8/p8r2_scored.json"
    if os.path.exists(outp):
        raise RuntimeError("p8r2_scored.json exists — one-shot already ran")
    need = ([f"spec_s{s}" for s in SEEDS] + [f"guard_s{s}" for s in SEEDS] +
            [f"guardnr_s{s}" for s in SEEDS] + [f"refaith_s{s}" for s in SEEDS] +
            [f"reburn_s{s}" for s in SEEDS] + [f"hold_s{s}" for s in SEEDS])
    missing = [n for n in need if not os.path.exists(f"{GRID}/{n}/summary.json")]
    if missing:
        raise RuntimeError(f"fleet incomplete: {missing}")

    res = dict(T0=T0, arms={})
    # guard validity + scaffold survival
    gv, surv, survnr = {}, {}, {}
    for s in SEEDS:
        for tag, store in (("guard", surv), ("guardnr", survnr)):
            recs, _ = load(f"{tag}_s{s}")
            last5 = recs[-5:]
            valid = (max(r["copy_adv"] for r in last5) <= 0.25 and
                     max(r["indist_adv"] for r in last5) <= 0.05)
            store[s] = dict(prevtok_end=last5[-1]["prevtok_by_layer"][0],
                            copy_end=round(float(np.median([r["copy_adv"] for r in last5])), 4),
                            guard_valid=bool(valid))
    res["arms"]["guard_shufrep"] = surv
    res["arms"]["guard_norep"] = survnr

    # reteach arms
    def arm(prefix):
        out = {}
        for s in SEEDS:
            recs, summ = load(f"{prefix}_s{s}")
            te = summ["t_event"]
            ta = anchor_step(recs)
            out[s] = dict(t_reevent=te, t_anchor=ta,
                          lead=(te - ta) if (te is not None and ta is not None) else None,
                          prevtok0_start=recs[0]["prevtok_by_layer"][0])
        return out
    faith = arm("refaith")
    burn = arm("reburn")
    res["arms"]["refaith"] = faith
    res["arms"]["reburn"] = burn

    # negatives: anchor alarms on hold streams
    holds = {}
    for s in SEEDS:
        recs, _ = load(f"hold_s{s}")
        holds[s] = dict(anchor=anchor_step(recs),
                        max_copy=round(max(r["copy_adv"] for r in recs), 4))
    res["arms"]["hold"] = holds

    def med(d, k):
        v = [x[k] for x in d.values() if x[k] is not None]
        return float(np.median(v)) if v else None

    mf, mb = med(faith, "t_reevent"), med(burn, "t_reevent")
    v = {}
    v["guard_validity"] = {s: surv[s]["guard_valid"] for s in SEEDS}
    n_surv = sum(surv[s]["prevtok_end"] >= 0.5 for s in SEEDS)
    n_dead = sum(surv[s]["prevtok_end"] < 0.10 for s in SEEDS)
    v["P_R2a_scaffold_survives"] = (f"{'PASS' if n_surv >= 2 else 'FAIL'} "
                                    f"({n_surv}/3 >= 0.5; prevtok {[round(surv[s]['prevtok_end'],3) for s in SEEDS]})")
    v["K_R2a"] = "FIRES — premise dead" if n_dead >= 2 else "no-fire"
    v["P_R2b_fast_relearn"] = (f"{'PASS' if (mf is not None and mf <= 3150) else 'FAIL'} "
                               f"(faithful median {mf} vs bar 3150)")
    v["K_R2b_not_fast"] = ("FIRES — relearn not fast at 2L"
                           if (mf is None or mf >= 5040) else "no-fire")
    ratio = (mb / mf) if (mf and mb) else None
    v["P_R2c_compensating_law"] = (
        f"{'PASS' if (ratio is not None and ratio >= 2.0 and mb >= 3800) else 'FAIL'} "
        f"(burned {mb}, faithful {mf}, ratio {None if ratio is None else round(ratio, 2)})")
    v["K_R2c_disguise_free"] = ("FIRES — disguise costs nothing (uncovered cell)"
                                if (ratio is not None and ratio <= 1.25) else "no-fire")
    bl = med(burn, "lead")
    n_fa = sum(holds[s]["anchor"] is not None for s in SEEDS)
    v["P_R2d_warning_returns"] = (
        f"{'PASS' if (bl is not None and bl >= 300 and n_fa == 0) else 'FAIL'} "
        f"(burned median lead {bl}; hold FA {n_fa}/3)")
    v["faithful_lead_reported"] = med(faith, "lead")
    res["verdicts"] = v
    json.dump(res, open(outp, "w"), indent=1)
    print(json.dumps(v, indent=1))
    print("\nper-run:")
    for tag, d in (("refaith", faith), ("reburn", burn)):
        for s in SEEDS:
            print(f"  {tag}_s{s}: t_reevent={d[s]['t_reevent']} anchor={d[s]['t_anchor']} "
                  f"lead={d[s]['lead']} prevtok0@start={d[s]['prevtok0_start']:.3f}")


if __name__ == "__main__":
    main()
