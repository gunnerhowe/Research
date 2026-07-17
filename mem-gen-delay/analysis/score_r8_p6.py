"""P6 R8: one-shot scoring of the gap-origin probe (prereg commit 4a5083f).

Gap = t_event - t_pv (t_pv = first eval with layer-0 prevtok >= 0.10). Cells: lr in
{5e-4, 2e-3} x seeds 301-305 (batch 64); batch in {32, 128} x seeds 311-315 (lr 1e-3).
Reference: the 30-seed fleet (lr 1e-3, batch 64; median gap 975).
Frozen discrimination rule: an axis CONTROLS the gap if its low/high median-gap ratio
>= 2 while the other axis' ratio <= 1.5. H1 (optimization clock): lr ratio ~4, batch ~1.
H2 (token clock): batch ratio ~4, lr ~1. K8b: both < 2 or both >= 2 -> no clean single
clock; gap origin reported open/mixed.
"""
import json
import os

import numpy as np

GRID = "runs/grid6r8"


def gap_of(path):
    summ = json.load(open(f"{path}/summary.json"))
    recs = [json.loads(l) for l in open(f"{path}/metrics.jsonl") if l.strip()]
    t_pv = next((r["step"] for r in recs if r["prevtok_by_layer"][0] >= 0.10), None)
    ev = summ["t_event"]
    return dict(t_event=ev, t_pv=t_pv,
                gap=None if (ev is None or t_pv is None) else ev - t_pv)


def main():
    cells = {}
    for tag, seeds in [("lr5e-4", range(301, 306)), ("lr2e-3", range(301, 306)),
                       ("b32", range(311, 316)), ("b128", range(311, 316))]:
        rows = []
        for s in seeds:
            r = gap_of(f"{GRID}/{tag}_s{s}")
            rows.append(dict(seed=s, **r))
            print(f"{tag}_s{s}: event={r['t_event']} t_pv={r['t_pv']} gap={r['gap']}")
        gaps = [r["gap"] for r in rows if r["gap"] is not None]
        evs = [r["t_event"] for r in rows if r["t_event"] is not None]
        cells[tag] = dict(rows=rows, median_gap=float(np.median(gaps)) if gaps else None,
                          median_event=float(np.median(evs)) if evs else None,
                          n_event=len(evs))
    # reference from the original fleet
    ref_gaps, ref_evs = [], []
    for s in range(1, 31):
        r = gap_of(f"runs/grid6r2/rep_s{s}")
        ref_gaps.append(r["gap"])
        ref_evs.append(r["t_event"])
    cells["ref_lr1e-3_b64"] = dict(median_gap=float(np.median(ref_gaps)),
                                   median_event=float(np.median(ref_evs)), n_event=30)
    print("\nper-cell medians:")
    for k, v in cells.items():
        print(f"  {k:16s} median_gap={v['median_gap']} median_event={v['median_event']} "
              f"(n={v['n_event']})")
    lr_ratio = cells["lr5e-4"]["median_gap"] / cells["lr2e-3"]["median_gap"]
    b_ratio = cells["b32"]["median_gap"] / cells["b128"]["median_gap"]
    lr_controls = lr_ratio >= 2 and b_ratio <= 1.5
    b_controls = b_ratio >= 2 and lr_ratio <= 1.5
    k8b = not (lr_controls or b_controls)
    res = dict(cells={k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                      for k, v in cells.items()},
               per_run={k: v.get("rows") for k, v in cells.items() if "rows" in v},
               lr_ratio=round(lr_ratio, 3), batch_ratio=round(b_ratio, 3),
               lr_controls=bool(lr_controls), batch_controls=bool(b_controls),
               K8b_fires=bool(k8b))
    print(f"\nlr ratio (5e-4 / 2e-3): {lr_ratio:.2f}  |  batch ratio (b32 / b128): {b_ratio:.2f}")
    print(f"H1 lr-controls: {lr_controls} | H2 batch-controls: {b_controls} | K8b fires: {k8b}")
    os.makedirs("analysis/out6", exist_ok=True)
    json.dump(res, open("analysis/out6/r8_scored.json", "w"), indent=2)


if __name__ == "__main__":
    main()
