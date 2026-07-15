"""P5 R5b: apply the FROZEN thresholds to the confirmation grid (prereg commit 10082d3).

Frozen (fit on grid5r5 seed 8, unchanged since): d.top1_frac >= 0.006197 (label-free) and
cos_gap >= 0.1107 (task-aware). Applied causally to grid5r5b (seeds 10-12; 6 positives,
3 structural negatives). P-R5b-a: label-free median lead on positives >= 18,100 with 0/3
false alarms. P-R5b-b: cos_gap retains >= 3,600. K9: any label-free false alarm OR lead
below the bar -> superiority claim dies.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict_emergence as pe

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "analysis", "out5")
FROZEN = {"d.top1_frac": (">=", 0.006197), "cos_gap": (">=", 0.1107)}
BARS = {"d.top1_frac": 18100, "cos_gap": 3600}


def main():
    saved = pe.GRIDS
    pe.GRIDS = ["grid5r5b"]
    runs = [r for r in pe.load_corpus() if r["domain"] == "mnist"]
    pe.GRIDS = saved
    for r in runs:
        recs = [json.loads(l) for l in open(os.path.join(
            "runs", "grid5r5b", r["name"], "metrics.jsonl")) if l.strip()]
        for k in ("eff_rank", "part_ratio", "top1_frac"):
            if k in recs[0]:
                r["sigs"][k] = np.array([x.get(k, np.nan) for x in recs], float)
        r["_sigs"] = pe.add_slopes(r)
    pos = [r for r in runs if r["t_gen"] is not None]
    neg = [r for r in runs if r["t_gen"] is None]
    print(f"grid5r5b: {len(runs)} runs ({len(pos)} positives, {len(neg)} structural negatives)")
    out = {}
    for sig, (d, th) in FROZEN.items():
        leads, fa_names = [], []
        for r in pos:
            ta = pe.alarm_time(r["times"], r["_sigs"][sig], d, th)
            leads.append(0.0 if ta is None else max(0.0, r["t_gen"] - ta))
        for r in neg:
            if pe.alarm_time(r["times"], r["_sigs"][sig], d, th) is not None:
                fa_names.append(r["name"])
        med = float(np.median(leads))
        rel = float(np.median([l / r["t_gen"] for l, r in zip(leads, pos)]))
        passed = med >= BARS[sig] and (sig != "d.top1_frac" or not fa_names)
        out[sig] = dict(per_run_leads={r["name"]: l for r, l in zip(pos, leads)},
                        median_lead=med, median_rel=rel, bar=BARS[sig],
                        false_alarms=fa_names, passed=bool(passed))
        print(f"\n{sig} {d}{th}: median lead {med:.0f} (rel {rel:.3f}), bar {BARS[sig]}, "
              f"FA on {fa_names or 'none'} -> {'PASS' if passed else 'FAIL'}")
        for r, l in zip(pos, leads):
            print(f"   {r['name']:22s} t_gen {r['t_gen']:>7} lead {l:>8.0f}")
    k9 = not out["d.top1_frac"]["passed"]
    out["K9_fires"] = k9
    json.dump(out, open(os.path.join(OUT, "r5b_scored.json"), "w"), indent=2)
    print(f"\nK9 (label-free superiority dies) fires: {k9}")


if __name__ == "__main__":
    main()
