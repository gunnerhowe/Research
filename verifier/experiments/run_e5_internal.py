"""E5 (internal reconciliation): show, from the E0 2x2, that one mechanism —
judges scoring novelty SIGNAL (G) rather than SUBSTANCE (S) — produces BOTH tails
that the two benchmarks observed:
  - RQ-Bench OVER-rating: low-substance-but-signaled ideas score high;
  - RINoBench compression/decoupling: genuine (high-substance) novelty framed
    plainly is under-credited, so Y decouples from true novelty S.

Metrics per judge (from results/e0_results.scores.csv):
  - variance of the judge's novelty score explained by G vs S (balanced 2x2);
  - over-rating rate  P(Y[low-substance] > mean Y[high-substance]);
  - under-rating rate P(Y[high-substance] < mean Y[low-substance]);
  - Y-vs-S decoupling: how weakly the score tracks true substance.

This is our controlled evidence for the mechanism; external validation on the
RQ-Bench / RINoBench corpora themselves is future work.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def summarize(d: pd.DataFrame) -> dict:
    hi_s = d[d.s == 1]["y"]
    lo_s = d[d.s == 0]["y"]
    over = float((lo_s > hi_s.mean()).mean())          # non-novel scoring above avg novel
    under = float((hi_s < lo_s.mean()).mean())          # novel scoring below avg non-novel
    # balanced-2x2 variance shares from cell means
    cm = d.groupby("cell")["y"].mean()
    bS = 0.5 * ((cm["HL"] + cm["HH"]) - (cm["LL"] + cm["LH"]))
    bG = 0.5 * ((cm["LH"] + cm["HH"]) - (cm["LL"] + cm["HL"]))
    bSG = (cm["HH"] - cm["HL"]) - (cm["LH"] - cm["LL"])
    denom = bS**2 + bG**2 + bSG**2
    return {
        "over_rating_rate": over, "under_rating_rate": under,
        "var_share_G": float(bG**2 / denom) if denom else float("nan"),
        "var_share_S": float(bS**2 / denom) if denom else float("nan"),
        "corr_Y_S": float(np.corrcoef(d.s, d.y)[0, 1]),
        "corr_Y_G": float(np.corrcoef(d.g, d.y)[0, 1]),
        "cell_means": {c: float(cm[c]) for c in ("LL", "LH", "HL", "HH")},
    }


def main():
    df = pd.read_csv(ROOT / "results" / "e0_results.scores.csv")
    out = {"per_judge": {}, "note": "one signal-tracking mechanism yields both the "
           "over-rating (RQ-Bench) and the score/substance decoupling (RINoBench)."}
    for jid in sorted(df.judge.unique()):
        s = summarize(df[df.judge == jid])
        out["per_judge"][jid] = s
        print(f"{jid.split('/')[-1]}:")
        print(f"  over-rating  P(non-novel > avg novel) = {s['over_rating_rate']:.2f}")
        print(f"  under-rating P(novel < avg non-novel)  = {s['under_rating_rate']:.2f}")
        print(f"  variance share  G={s['var_share_G']:.2f}  S={s['var_share_S']:.2f}   "
              f"corr(Y,G)={s['corr_Y_G']:.2f} corr(Y,S)={s['corr_Y_S']:.2f}")
    out["pooled"] = summarize(df)
    p = out["pooled"]
    print(f"\nPOOLED: over={p['over_rating_rate']:.2f} under={p['under_rating_rate']:.2f} "
          f"varG={p['var_share_G']:.2f} varS={p['var_share_S']:.2f}")
    (ROOT / "results" / "e5_reconciliation.json").write_text(json.dumps(out, indent=2))
    print("wrote results/e5_reconciliation.json")


if __name__ == "__main__":
    main()
