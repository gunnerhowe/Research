"""P5 R4: cross-domain transfer of forecasting thresholds (plan_p5.md, K5).

Fit per-signal alarm thresholds on one domain, apply to the other. Shared signals only,
z-standardized PER DOMAIN over pooled post-warmup values (unsupervised calibration — uses
signal values, never outcomes). Fit side: all runs of the source domain, FA <= 5% on source
negatives. Target side: report median lead, miss, and FA at the transferred threshold.
K5: no signal achieves positive target median lead with target FA <= 5% in either
direction -> forecasting calibration is regime-specific (scope limit, reported as such).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict_emergence as pe

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "analysis", "out5")
SHARED = ["cos_gap", "wnorm", "logit_scale", "conf", "osc"]


def zstats(runs, sig):
    vals = np.concatenate([r["_sigs"][sig][pe.WARMUP:] for r in runs
                           if r["_sigs"].get(sig) is not None])
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()), float(vals.std() + 1e-12)


def standardize(runs, sigs):
    for sig in sigs:
        mu, sd = zstats(runs, sig)
        for r in runs:
            v = r["_sigs"].get(sig)
            if v is not None:
                r["_sigs"]["z." + sig] = (v - mu) / sd


def main():
    runs = pe.load_corpus()
    for r in runs:
        r["_sigs"] = pe.add_slopes(r)
    sigs = [s for b in SHARED for s in (b, "d." + b)]
    for domain in ("alg", "mnist"):
        standardize([r for r in runs if r["domain"] == domain], sigs)
    zsigs = ["z." + s for s in sigs]
    out = {}
    for src, tgt in (("alg", "mnist"), ("mnist", "alg")):
        S = [r for r in runs if r["domain"] == src]
        T = [r for r in runs if r["domain"] == tgt]
        print(f"\n=== R4 transfer {src} -> {tgt} (src n={len(S)}, tgt n={len(T)}, "
              f"tgt neg={sum(1 for r in T if r['t_gen'] is None)}) ===")
        rows = []
        for sig in zsigs:
            fit = pe.fit_threshold(S, sig)
            if fit is None:
                continue
            _, d, th, tr = fit
            te = pe.score(T, sig, d, th)
            rows.append(dict(signal=sig, dir=d, theta=round(th, 4),
                             src_lead=round(tr["median_lead"], 1),
                             tgt_lead=round(te["median_lead"], 1),
                             tgt_rel=round(te["median_rel_lead"], 4),
                             tgt_miss=te["miss_rate"], tgt_fa=te["fa_rate"]))
            print(f"  {sig:16s} {d}{th:+.3f}: src lead {tr['median_lead']:>7.0f} -> "
                  f"tgt lead {te['median_lead']:>7.0f} (rel {te['median_rel_lead']:.3f}) "
                  f"miss {te['miss_rate']:.2f} FA {te['fa_rate']}")
        transfers = [r for r in rows if r["tgt_lead"] > 0 and r["tgt_fa"] is not None
                     and r["tgt_fa"] <= 0.05]
        out[f"{src}_to_{tgt}"] = dict(rows=rows, transfers=[r["signal"] for r in transfers])
        print(f"  SIGNALS THAT TRANSFER (lead>0, tgt FA<=5%): "
              f"{[r['signal'] for r in transfers] or 'NONE'}")
    k5 = all(not v["transfers"] for v in out.values())
    out["K5_fires"] = k5
    json.dump(out, open(os.path.join(OUT, "r4_transfer.json"), "w"), indent=2)
    print(f"\nK5 (no transfer either direction) fires: {k5}")


if __name__ == "__main__":
    main()
