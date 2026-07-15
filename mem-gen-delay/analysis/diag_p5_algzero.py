"""P5 diagnostic: why did the algorithmic domain return zero median lead?

Three candidate stories (see 2026-07-14 session notes):
  A) mixture artifact — pooling 400-epoch and 40,000-epoch groks makes median-lead-in-epochs
     collapse (fast runs cannot have lead at eval cadence);
  B) warmup artifact — the 5-eval warmup eats fast runs' entire window;
  C) real finding — coherent wrong-structure negatives (wrong/band arms build wrong-task
     structure) bind the FA<=5% constraint and push probe thresholds past usefulness.

TRAIN-SIDE ONLY (even seeds): diagnostics must not touch held-out test seeds.
Read-only w.r.t. the pre-registered spec; informs a disclosed amendment if warranted.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_emergence import (load_corpus, add_slopes, fit_threshold, score,
                               alarm_time, WARMUP, SIG_S, SIG_R)


def leads_per_run(runs, sig, direction, theta):
    out = []
    for r in runs:
        x = r["_sigs"].get(sig)
        if x is None or r["t_gen"] is None:
            continue
        ta = alarm_time(r["times"], x, direction, theta)
        lead = 0.0 if ta is None else max(0.0, r["t_gen"] - ta)
        out.append((r["grid"], r["name"], r["condition"], r["t_gen"], lead))
    return out


def main():
    runs = [r for r in load_corpus() if r["domain"] == "alg" and r["seed"] % 2 == 0]
    for r in runs:
        r["_sigs"] = add_slopes(r)
    pos = [r for r in runs if r["t_gen"] is not None]
    neg = [r for r in runs if r["t_gen"] is None]
    print(f"alg TRAIN runs: {len(runs)} (pos {len(pos)}, neg {len(neg)})")

    # ---- 1) t_gen distribution + cadence/warmup exposure ----
    tg = np.array([r["t_gen"] for r in pos], float)
    ee = np.array([np.median(np.diff(r["times"])) for r in pos])
    warm_end = WARMUP * ee
    print(f"\n1) t_gen quartiles: {np.percentile(tg, [0, 25, 50, 75, 100]).astype(int)}")
    print(f"   eval cadence (median diff): {sorted(set(ee.astype(int)))}")
    print(f"   positives with t_gen <= warmup end: {(tg <= warm_end).sum()}/{len(tg)}")
    print(f"   positives with t_gen <= 2000: {(tg <= 2000).sum()}; <= 5000: {(tg <= 5000).sum()}")

    # ---- 2) who binds the FA constraint? refit key probe thresholds with negative SUBSETS ----
    def neg_subset(kinds):
        return [r for r in runs if r["t_gen"] is not None or
                any(k in r["condition"] or k in r["name"] for k in kinds)]

    subsets = {
        "all negatives (as prereg)": runs,
        "shuffled-only negatives": neg_subset(["shuffled", "shufpair"]),
        "no wrong/band negatives": [r for r in runs if r["t_gen"] is not None or not
                                    any(k in r["condition"] or k in r["name"]
                                        for k in ["wrong", "band"])],
    }
    for sig in ["fourier_top8", "cos_gap", "d.fisher", "test_ce", "wnorm"]:
        print(f"\n2) signal {sig}:")
        for label, sub in subsets.items():
            nn = sum(1 for r in sub if r["t_gen"] is None)
            fit = fit_threshold(sub, sig)
            if fit is None:
                print(f"   {label:32s} (neg n={nn:2d}): no feasible threshold")
                continue
            _, d, th, sc = fit
            print(f"   {label:32s} (neg n={nn:2d}): {d}{th:.4g} -> train median lead "
                  f"{sc['median_lead']:.0f}, miss {sc['miss_rate']:.2f}")

    # ---- 3) per-condition lead breakdown at the prereg threshold (slow vs fast runs) ----
    fit = fit_threshold(runs, "fourier_top8")
    if fit:
        _, d, th, _ = fit
        rows = leads_per_run(runs, "fourier_top8", d, th)
        arr = np.array([(t, l) for _, _, _, t, l in rows], float)
        slow = arr[arr[:, 0] >= 10000]
        fast = arr[arr[:, 0] < 10000]
        print(f"\n3) fourier_top8 at prereg-style threshold {d}{th:.4g}:")
        print(f"   slow groks (t_gen>=10k): n={len(slow)}, median lead {np.median(slow[:,1]):.0f}, "
              f"mean {slow[:,1].mean():.0f}, frac>0 {(slow[:,1]>0).mean():.2f}")
        print(f"   fast groks (t_gen<10k):  n={len(fast)}, median lead {np.median(fast[:,1]):.0f}, "
              f"frac>0 {(fast[:,1]>0).mean():.2f}")
        # relative leads
        rel_slow = slow[:, 1] / slow[:, 0]
        print(f"   slow-grok RELATIVE lead: median {np.median(rel_slow):.3f}")

    # ---- 4) which negatives alarm first for fourier_top8 (threshold sweep) ----
    print("\n4) negative-run signal maxima (fourier_top8, post-warmup) — who fires highest:")
    peaks = []
    for r in neg:
        x = r["_sigs"].get("fourier_top8")
        if x is None:
            continue
        peaks.append((float(np.nanmax(x[WARMUP:])), r["condition"], r["name"]))
    peaks.sort(reverse=True)
    for p, c, n in peaks[:8]:
        print(f"   {p:.4f}  {c:18s} {n}")
    pos_at_tgen = []
    for r in pos:
        x = r["_sigs"].get("fourier_top8")
        if x is None:
            continue
        i = int(np.searchsorted(r["times"], r["t_gen"]))
        pos_at_tgen.append(float(x[min(i, len(x) - 1)]))
    print(f"   positives' fourier_top8 AT t_gen: median {np.median(pos_at_tgen):.4f}, "
          f"10th pct {np.percentile(pos_at_tgen, 10):.4f}")


if __name__ == "__main__":
    main()
