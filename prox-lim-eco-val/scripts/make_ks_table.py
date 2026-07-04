"""Render the KS results table -> paper/tables/ks_results.tex."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper/tables"
OUT.mkdir(parents=True, exist_ok=True)

ORDER = ["base", "tpp", "pois", "marg", "margtpp"]
LABELS = {"base": r"Base (FM only)", "tpp": r"TPP-aux",
          "pois": r"Poisson-TPP-aux", "marg": r"Invariant-stats (incl.\ $u_t$)",
          "margtpp": r"Invariant-stats + TPP-aux"}
COLS = [("iet_ks", "IET KS", 3), ("iet_w1", "IET $W_1$", 1),
        ("rate_ratio", "rate ratio", 2), ("rl_logerr", "RP log-err", 2),
        ("tpp_ll", "TPP-LL", 3), ("marg_w1", "state $W_1$", 3),
        ("psd_logdist", "PSD log-dist", 2)]


def rows_for(cond):
    if cond == "base":
        dirs = [f"ks_grid_basephase_s{s}" for s in (1, 2, 3)]
    else:
        dirs = [f"ks_ft_{cond}_s{s}" for s in (1, 2, 3)]
    out = []
    for d in dirs:
        p = ROOT / f"runs/{d}/metrics.json"
        if p.exists():
            out.append(json.load(open(p)))
    return out


lines = [r"\begin{tabular}{l" + "c" * len(COLS) + "}", r"\toprule",
         "condition & " + " & ".join(h for _, h, _ in COLS) + r" \\",
         r"\midrule"]
for c in ORDER:
    ms = rows_for(c)
    if not ms:
        continue
    cells = []
    for k, _, p in COLS:
        v = np.array([float(m[k]) for m in ms])
        if len(v) > 1:
            cells.append(f"${v.mean():.{p}f} \\pm {v.std():.{p}f}$")
        else:
            cells.append(f"${v.mean():.{p}f}$")
    label = LABELS[c] + (r"\;$^\dagger$" if len(ms) == 1 else "")
    lines.append(label + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "ks_results.tex").write_text("\n".join(lines))
print("\n".join(lines))
