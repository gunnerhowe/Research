"""Generate all publication figures and LaTeX tables from the result JSONs.

    python -m scripts.make_figures

Writes PDFs/PNGs to paper/figures/ and .tex tables to paper/tables/.
Robust to partial results (skips a figure if its data is missing).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analyze import (load_synthetic, load_charlm, agg, significance_synthetic,
                             VARIANT_ORDER, PRETTY)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
TAB = os.path.join(ROOT, "paper", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "font.family": "serif",
})
COLORS = {
    "nope": "#9e9e9e", "sinusoidal": "#8c564b", "learned": "#e377c2",
    "rope": "#1f77b4", "alibi": "#2ca02c", "t5": "#ff7f0e", "cable": "#17becf",
    "semrf": "#d62728",
}
TASK_TITLE = {"assoc_recall": "Associative Recall", "temporal_recency": "Temporal Recency",
              "selective_copy": "Selective Copy"}


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(FIG, name))


def fig_extrapolation_synthetic(edf):
    if edf.empty:
        return
    tasks = [t for t in ["assoc_recall", "temporal_recency", "selective_copy"] if t in edf["task"].unique()]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = edf[edf["task"] == task]
        for v in VARIANT_ORDER:
            vv = sub[sub["variant"] == v]
            if vv.empty:
                continue
            g = agg(vv, ["seq_len"], "token_acc").sort_values("seq_len")
            ax.plot(g["seq_len"], g["mean"], "-o", ms=4, color=COLORS[v], label=PRETTY[v])
            ax.fill_between(g["seq_len"], g["mean"] - g["std"], g["mean"] + g["std"],
                            color=COLORS[v], alpha=0.15)
        trained = sub["seq_len"].min()
        ax.axvline(trained, ls=":", color="k", alpha=0.5)
        ax.set_title(TASK_TITLE.get(task, task))
        ax.set_xlabel("evaluation sequence length")
        ax.set_ylabel("answer token accuracy")
        ax.set_ylim(-0.02, 1.02)
    axes[0][-1].legend(fontsize=9, loc="best")
    fig.suptitle("Length extrapolation: train short, evaluate long (mean ± std over seeds)", y=1.02)
    _save(fig, "fig_extrapolation_synthetic")


def fig_distance(ddf):
    if ddf.empty:
        return
    tasks = [t for t in ["assoc_recall", "temporal_recency"] if t in ddf["task"].unique()]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = ddf[ddf["task"] == task]
        for v in VARIANT_ORDER:
            vv = sub[sub["variant"] == v].copy()
            if vv.empty:
                continue
            # merge per-seed buckets: bin distances onto a common grid, then
            # average accuracy across seeds within each bin (weighted by n)
            nb = 8
            lo, hi = vv["distance"].min(), vv["distance"].max()
            edges = np.linspace(lo, hi + 1e-9, nb + 1)
            vv["bin"] = np.clip(np.digitize(vv["distance"], edges) - 1, 0, nb - 1)
            g = vv.groupby("bin").apply(
                lambda d: pd.Series({
                    "distance": np.average(d["distance"], weights=d["n"]),
                    "acc": np.average(d["acc"], weights=d["n"]),
                }), include_groups=False).reset_index(drop=True).sort_values("distance")
            ax.plot(g["distance"], g["acc"], "-o", ms=3, color=COLORS[v], label=PRETTY[v], alpha=0.85)
        ax.set_title(TASK_TITLE.get(task, task))
        ax.set_xlabel("retrieval distance (tokens)")
        ax.set_ylabel("accuracy")
        ax.set_ylim(-0.02, 1.02)
    axes[0][-1].legend(fontsize=9)
    fig.suptitle("Accuracy vs. retrieval distance", y=1.02)
    _save(fig, "fig_distance_synthetic")


def fig_charlm_extrap(cedf):
    if cedf.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # left: all methods, log-y (catastrophic failures visible)
    ax = axes[0]
    for v in VARIANT_ORDER:
        vv = cedf[(cedf["variant"] == v) & (cedf["bpc"].notna())]
        if vv.empty:
            continue
        g = agg(vv, ["eval_len"], "bpc").sort_values("eval_len")
        ax.plot(g["eval_len"], g["mean"], "-o", ms=4, color=COLORS[v], label=PRETTY[v])
        ax.fill_between(g["eval_len"], g["mean"] - g["std"], g["mean"] + g["std"], color=COLORS[v], alpha=0.15)
    ax.axvline(512, ls=":", color="k", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("evaluation context length")
    ax.set_ylabel("bits per character (test, log scale)")
    ax.set_title("All methods")
    ax.legend(fontsize=8, ncol=2)

    # right: zoom on the extrapolating methods
    ax = axes[1]
    for v in ["alibi", "t5", "cable", "semrf"]:
        vv = cedf[(cedf["variant"] == v) & (cedf["bpc"].notna())]
        if vv.empty:
            continue
        g = agg(vv, ["eval_len"], "bpc").sort_values("eval_len")
        ax.plot(g["eval_len"], g["mean"], "-o", ms=5, color=COLORS[v], label=PRETTY[v])
        ax.fill_between(g["eval_len"], g["mean"] - g["std"], g["mean"] + g["std"], color=COLORS[v], alpha=0.2)
    ax.axvline(512, ls=":", color="k", alpha=0.5, label="train length")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("evaluation context length")
    ax.set_ylabel("bits per character (test)")
    ax.set_ylim(1.30, 1.55)
    ax.set_title("Zoom: additive-bias methods")
    ax.legend(fontsize=9)
    fig.suptitle("enwik8 length extrapolation (train at 512)", y=1.02)
    _save(fig, "fig_charlm_extrapolation")


def fig_ablation(edf):
    """SemRF ablations + K-sweep on mean extrapolation acc at longest length."""
    if edf.empty:
        return
    abls = ["semrf", "semrf_no_time", "semrf_no_sem", "semrf_no_res", "semrf_hard"]
    have = [a for a in abls if a in edf["variant"].unique()]
    if len(have) < 2:
        return
    # average over tasks of the longest-length accuracy
    rows = []
    for task in edf["task"].unique():
        sub = edf[edf["task"] == task]
        Lmax = sub["seq_len"].max()
        atL = sub[sub["seq_len"] == Lmax]
        for a in have:
            vv = atL[atL["variant"] == a]
            if not vv.empty:
                rows.append({"task": task, "variant": a, "acc": vv["token_acc"].mean()})
    adf = pd.DataFrame(rows)
    if adf.empty:
        return
    labels = {"semrf": "Full", "semrf_no_time": "− time", "semrf_no_sem": "− anchor",
              "semrf_no_res": "− residual", "semrf_hard": "hard assign"}
    task_order = [t for t in ["assoc_recall", "temporal_recency", "selective_copy"]
                  if t in adf["task"].unique()]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    width = 0.8 / len(have)
    shades = {"semrf": "#d62728", "semrf_no_time": "#1f77b4", "semrf_no_sem": "#9467bd",
              "semrf_no_res": "#8c564b", "semrf_hard": "#e377c2"}
    x = np.arange(len(task_order))
    for i, a in enumerate(have):
        vals = [adf[(adf["task"] == t) & (adf["variant"] == a)]["acc"].mean() for t in task_order]
        ax.bar(x + (i - len(have) / 2 + 0.5) * width, vals, width * 0.92,
               color=shades.get(a, "#333"), label=labels.get(a, a), alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_TITLE.get(t, t) for t in task_order])
    ax.set_ylabel("extrapolation accuracy (longest length)")
    ax.set_title("SemRF ablations, per task")
    ax.legend(fontsize=9, ncol=len(have), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_ylim(0, 1.05)
    _save(fig, "fig_ablation")

    # K sweep
    ks = {"semrf_K8": 8, "semrf": 16, "semrf_K64": 64}
    krows = []
    for task in edf["task"].unique():
        sub = edf[edf["task"] == task]
        Lmax = sub["seq_len"].max()
        atL = sub[sub["seq_len"] == Lmax]
        for var, k in ks.items():
            vv = atL[atL["variant"] == var]
            if not vv.empty:
                krows.append({"K": k, "acc": vv["token_acc"].mean()})
    kdf = pd.DataFrame(krows)
    if not kdf.empty and kdf["K"].nunique() >= 2:
        g = kdf.groupby("K")["acc"].agg(["mean", "std"]).reset_index().sort_values("K")
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.errorbar(g["K"], g["mean"], yerr=g["std"].fillna(0), fmt="-o", color="#d62728", capsize=4)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("number of anchors K")
        ax.set_ylabel("extrapolation accuracy")
        ax.set_title("Effect of anchor count")
        _save(fig, "fig_ksweep")


# ------------------------------ tables ------------------------------------- #
def _fmt(m, s):
    return f"{m:.3f}\\,$\\pm$\\,{s:.3f}"


def table_synthetic_final(fdf):
    if fdf.empty:
        return
    tasks = [t for t in ["assoc_recall", "temporal_recency", "selective_copy"]
             if t in fdf["task"].unique()]
    if not tasks:
        return
    lines = [r"\begin{tabular}{l" + "c" * len(tasks) + "}", r"\toprule",
             "Method & " + " & ".join(TASK_TITLE.get(t, t) for t in tasks) + r" \\", r"\midrule"]
    for v in VARIANT_ORDER:
        cells = []
        for t in tasks:
            sub = fdf[(fdf["task"] == t) & (fdf["variant"] == v)]
            cells.append(_fmt(sub["token_acc"].mean(), sub["token_acc"].std(ddof=0)) if not sub.empty else "--")
        name = PRETTY[v]
        if v == "semrf":
            name = r"\textbf{" + name + "}"
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "table_synthetic_final.tex"), "w").write("\n".join(lines))
    print("wrote table_synthetic_final.tex")


def table_charlm(cdf):
    if cdf.empty:
        return
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"Method & Params & Test bpc $\downarrow$ \\", r"\midrule"]
    for v in VARIANT_ORDER:
        sub = cdf[cdf["variant"] == v]
        if sub.empty:
            continue
        params = f"{sub['n_params'].iloc[0]/1e6:.1f}M"
        bpc = f"{sub['test_bpc'].mean():.4f}\\,$\\pm$\\,{sub['test_bpc'].std(ddof=0):.4f}"
        name = r"\textbf{SemRF (ours)}" if v == "semrf" else PRETTY[v]
        lines.append(f"{name} & {params} & {bpc} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "table_charlm.tex"), "w").write("\n".join(lines))
    print("wrote table_charlm.tex")


def main():
    fdf, edf, ddf = load_synthetic()
    cdf, cedf = load_charlm()
    fig_extrapolation_synthetic(edf)
    fig_distance(ddf)
    fig_charlm_extrap(cedf)
    fig_ablation(edf)
    table_synthetic_final(fdf)
    table_charlm(cdf)
    print("done.")


if __name__ == "__main__":
    main()
