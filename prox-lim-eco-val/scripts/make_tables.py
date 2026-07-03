"""Render LaTeX results tables from runs/summary.csv (+ paired tests).

Writes paper/tables/main_results.tex. Rerun after adding seeds.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper/tables"
OUT.mkdir(parents=True, exist_ok=True)

ORDER = ["base", "tpp", "pois", "marg", "push", "det"]
LABELS = {"base": r"Base (FM only)", "tpp": r"\textbf{TPP-aux (ours)}",
          "pois": r"Poisson-TPP-aux", "marg": r"Invariant-stats",
          "push": r"Pushforward", "det": r"Deterministic AR"}
COLS = [("iet_ks", "IET KS", 3), ("iet_w1", "IET $W_1$", 3),
        ("rate_ratio", "rate ratio", 3), ("rl_logerr", "RP log-err", 3),
        ("tpp_ll", "TPP-LL", 4), ("marg_w1", "state $W_1$", 3),
        ("crps20", "CRPS@20", 3)]

df = pd.read_csv(ROOT / "runs/summary.csv")


def cell(sub, key, prec, best):
    v = sub[key].astype(float)
    m, s = v.mean(), v.std()
    txt = f"{m:.{prec}f} \\pm {s:.{prec}f}"
    return f"$\\mathbf{{{txt}}}$" if best else f"${txt}$"


# per-metric best condition (closest to target: 1 for rate_ratio, max for
# tpp_ll, min otherwise), det excluded from "best" competition
def best_cond(key):
    scores = {}
    for c in ORDER[:-1]:
        v = df[df.condition == c][key].astype(float)
        if key == "rate_ratio":
            scores[c] = abs(v.mean() - 1)
        elif key == "tpp_ll":
            scores[c] = -v.mean()
        else:
            scores[c] = v.mean()
    return min(scores, key=scores.get)


bests = {k: best_cond(k) for k, _, _ in COLS}
lines = [r"\begin{tabular}{l" + "c" * len(COLS) + "}", r"\toprule",
         "condition & " + " & ".join(h for _, h, _ in COLS) + r" \\",
         r"\midrule"]
for c in ORDER:
    sub = df[df.condition == c]
    if sub.empty:
        continue
    cells = [cell(sub, k, p, bests[k] == c) for k, _, p in COLS]
    lines.append(LABELS[c] + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "main_results.tex").write_text("\n".join(lines))
n = df.groupby("condition").size().min()
print(f"wrote main_results.tex (n={n} seeds/condition)")
print("\n".join(lines))
