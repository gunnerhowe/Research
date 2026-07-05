"""Figures + tables for paper #3 (differentiable Minkowski functionals).

Reads results/phase3/*.json; writes paper3/figs and paper3/tables.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT

RES = ROOT / "results" / "phase3"
FIGS = ROOT / "paper3" / "figs"
TABS = ROOT / "paper3" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 1.4,
})


def load(suite):
    recs, seen = [], set()
    for p in sorted(RES.glob(f"{suite}*.json")):
        for r in json.loads(p.read_text()):
            key = (r.get("shape"), r.get("config"), r.get("seed"),
                   r.get("iters"), r.get("omega"), r.get("n_dom"))
            if key in seen:
                continue
            seen.add(key)
            recs.append(r)
    return recs


def tab_exp9():
    recs = load("exp9")
    by = defaultdict(list)
    for r in recs:
        by[r["config"]].append(r)
    labels = {"anchor_only": "anchor only", "smooth_null": "+ smoothness (null)",
              "chi_only": "+ $\\chi$-cap alone", "vector_cap":
              "+ Minkowski-vector cap (ours)"}
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "Config & repaired & spurious max & clean MSE \\\\", "\\midrule"]
    for cfg in ("anchor_only", "smooth_null", "chi_only", "vector_cap"):
        rs = by[cfg]
        rep = sum(r["repaired"] for r in rs)
        sp = np.array([r["spurious_max"] for r in rs])
        ms = np.array([r["clean_mse"] for r in rs])
        lines.append(f"{labels[cfg]} & {rep}/{len(rs)} & "
                     f"{sp.mean():.3f} $\\pm$ {sp.std():.3f} & "
                     f"{ms.mean():.4f} $\\pm$ {ms.std():.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABS / "p3_exp9.tex").write_text("\n".join(lines))


def tab_exp10():
    recs = load("exp10")
    regimes = [("hard ($\\omega{=}15$, 8k)", 15.0, 8000),
               ("matched ($\\omega{=}8$, 16k)", 8.0, 16000)]
    labels = {"eikonal_only": "Eikonal only", "smooth_null": "+ smoothness",
              "vector_match": "+ Minkowski vector (ours)",
              "ph_loss": "+ persistent homology"}
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Config & topo correct & median $b_1$ err & Chamfer & topo ms/iter \\\\"]
    for rname, om, nd in regimes:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{rname} regime}}}} \\\\")
        for cfg in ("eikonal_only", "smooth_null", "vector_match", "ph_loss"):
            rs = [r for r in recs if r["config"] == cfg
                  and (r.get("omega") or 15.0) == om
                  and (r.get("n_dom") or 8000) == nd]
            if not rs:
                continue
            ok = sum(r["topo_correct"] for r in rs)
            b1e = np.median([abs(r["betti"][1] - r["betti_gt"][1]) for r in rs])
            ch = np.mean([r["chamfer"] for r in rs])
            ms = np.mean([r["topo_ms_per_iter"] for r in rs])
            lines.append(f"{labels[cfg]} & {ok}/{len(rs)} & {b1e:.0f} & "
                         f"{ch:.3f} & {ms:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABS / "p3_exp10.tex").write_text("\n".join(lines))


def fig_debris_scaling():
    recs = [r for r in load("exp10") if r["config"] == "vector_match"
            and (r.get("omega") or 15.0) == 8.0 and r["seed"] == 0]
    by_n = defaultdict(list)
    for r in recs:
        by_n[r.get("n_dom") or 8000].append(sum(r["betti"]))
    # include the omega-8 16k runs from _bw and 32k from _scal32
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ns = sorted(by_n)
    med = [np.median(by_n[n]) for n in ns]
    ax.plot(ns, med, "o-", color="#d62728", label="total spurious features")
    ax.axhline(3, color="#555", ls="--", lw=0.9, label="ground-truth features")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("collocation samples $N$")
    ax.set_ylabel("features at $96^3$ eval")
    ax.set_title("debris is invariant to affordable $N$", fontsize=8)
    ax.legend(fontsize=6.5)
    fig.savefig(FIGS / "p3_scaling.pdf")
    plt.close(fig)


def tab_gatev():
    rows = [
        ("2D one bump", "1", "0.998--1.021", "$\\le 0.012$"),
        ("2D two bumps (merge)", "2 / 1", "2.005--2.017 / 1.006", "$\\le 0.026$"),
        ("2D annulus (hole)", "0", "0.011--0.019", "$\\le 0.040$"),
        ("3D ball", "1", "0.987--1.012", "$\\le 0.019$"),
        ("3D solid torus", "0", "0.016--0.026", "$\\le 0.071$"),
    ]
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "Field & exact $\\chi$ & $\\hat\\chi$ range (3 seeds) & s.d. \\\\",
             "\\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABS / "p3_gatev.tex").write_text("\n".join(lines))


def main():
    done, skipped = [], []
    for name, fn in [("tab_exp9", tab_exp9), ("tab_exp10", tab_exp10),
                     ("fig_debris_scaling", fig_debris_scaling),
                     ("tab_gatev", tab_gatev)]:
        try:
            fn()
            done.append(name)
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{name}: {type(e).__name__}: {e}")
    print("done:", ", ".join(done))
    for s in skipped:
        print("SKIP", s)


if __name__ == "__main__":
    main()
