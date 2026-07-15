"""P5 R5: score the label-free probe battery (prereg: STATUS block at commit 2365021).

Protocol (frozen before grid5r5 completed): threshold-alarm machinery on grid5r5 runs;
fit each signal's threshold on SEED-8 runs (FA cap over the seed-8 negatives), validate on
SEED-9 runs. Signals: label-free {eff_rank, part_ratio, top1_frac} (+slopes) vs the
task-aware probe {cos_gap} (+slope) ON THE SAME RUNS. K6: best label-free validated lead
< 50% of cos_gap's validated lead -> early warning needs capability-specific probes.
Exploratory scale (n=2 seeds/arm) — reported as such.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict_emergence as pe

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "analysis", "out5")
LABEL_FREE = ["eff_rank", "part_ratio", "top1_frac"]
TASK_AWARE = ["cos_gap"]


def load_r5():
    saved = pe.GRIDS
    pe.GRIDS = ["grid5r5"]
    runs = [r for r in pe.load_corpus() if r["domain"] == "mnist"]
    pe.GRIDS = saved
    for r in runs:
        # spectra fields are not in pe's declared signal sets; attach them manually
        import json as _j
        recs = [_j.loads(l) for l in open(os.path.join(
            "runs", "grid5r5", r["name"], "metrics.jsonl")) if l.strip()]
        for k in LABEL_FREE:
            if k in recs[0]:
                r["sigs"][k] = np.array([x.get(k, np.nan) for x in recs], float)
        r["_sigs"] = pe.add_slopes(r)
    return runs


def main():
    runs = load_r5()
    fit_runs = [r for r in runs if r["seed"] == 8]
    val_runs = [r for r in runs if r["seed"] == 9]
    print(f"grid5r5: {len(runs)} runs; fit(seed8) n={len(fit_runs)} "
          f"(neg {sum(1 for r in fit_runs if r['t_gen'] is None)}), "
          f"val(seed9) n={len(val_runs)} "
          f"(neg {sum(1 for r in val_runs if r['t_gen'] is None)})")
    rows = []
    for fam, sigs in (("label_free", LABEL_FREE), ("task_aware", TASK_AWARE)):
        for base in sigs:
            for sig in (base, "d." + base):
                fit = pe.fit_threshold(fit_runs, sig)
                if fit is None:
                    print(f"  {fam:10s} {sig:14s}: no feasible threshold on seed 8")
                    continue
                _, d, th, tr = fit
                va = pe.score(val_runs, sig, d, th)
                rows.append(dict(family=fam, signal=sig, dir=d, theta=round(th, 5),
                                 fit_lead=round(tr["median_lead"], 1),
                                 val_lead=round(va["median_lead"], 1),
                                 val_rel=round(va["median_rel_lead"], 4),
                                 val_miss=va["miss_rate"], val_fa=va["fa_rate"]))
                print(f"  {fam:10s} {sig:14s} {d}{th:.4g}: fit lead {tr['median_lead']:>7.0f}"
                      f" -> val lead {va['median_lead']:>7.0f} (rel {va['median_rel_lead']:.3f})"
                      f" miss {va['miss_rate']:.2f} FA {va['fa_rate']}")
    # champions chosen on FIT lead (no val peeking), K6 on validated leads
    champs = {}
    for fam in ("label_free", "task_aware"):
        cand = [r for r in rows if r["family"] == fam]
        if cand:
            champs[fam] = max(cand, key=lambda r: r["fit_lead"])
    out = dict(rows=rows, champions=champs)
    if len(champs) == 2:
        lf, ta = champs["label_free"], champs["task_aware"]
        ratio = lf["val_lead"] / ta["val_lead"] if ta["val_lead"] else float("inf")
        out["k6"] = dict(label_free=lf["signal"], lf_val_lead=lf["val_lead"],
                         task_aware=ta["signal"], ta_val_lead=ta["val_lead"],
                         recovery=round(ratio, 3), fires=bool(ratio < 0.5))
        print(f"\nK6: label-free champion {lf['signal']} val lead {lf['val_lead']:.0f} vs "
              f"task-aware {ta['signal']} {ta['val_lead']:.0f} -> recovery {ratio:.1%}; "
              f"K6 fires: {ratio < 0.5}")
    json.dump(out, open(os.path.join(OUT, "r5_scored.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
