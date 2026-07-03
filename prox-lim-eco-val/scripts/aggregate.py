"""Aggregate run metrics into tables and figures.

Usage: python scripts/aggregate.py [--pattern "ft_{cond}_s{seed}"]
                                   [--conditions base tpp pois marg push det]
                                   [--seeds 1 2 3 4 5] [--data_dir data/dt02]
Writes runs/summary.csv, runs/summary.md, and paper/figs/*.pdf
"""

import argparse
import json
from pathlib import Path


def savefig(fig, figdir, name):
    fig.savefig(figdir / f"{name}.pdf")
    fig.savefig(figdir / f"{name}.png", dpi=150)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent

KEY_METRICS = ["iet_ks", "iet_w1", "fano_logerr", "rl_logerr", "tpp_ll",
               "rate_ratio", "marg_w1", "psd_logdist", "acf_rmse", "div_frac"]
LABELS = {"base": "Base (FM only)", "tpp": "TPP-aux (ours)",
          "pois": "Poisson-TPP-aux", "marg": "Invariant-stats (Jiang)",
          "push": "Pushforward", "det": "Deterministic AR",
          "shuf": "Shuffled-TPP-aux", "tpp_mle": "TPP-MLE (no ratio)"}
COLORS = {"base": "#888888", "tpp": "#d62728", "pois": "#1f77b4",
          "marg": "#2ca02c", "push": "#9467bd", "det": "#8c564b",
          "shuf": "#e377c2", "tpp_mle": "#7f7f7f"}


def collect(conditions, seeds, pattern):
    rows = []
    for c in conditions:
        for s in seeds:
            p = ROOT / "runs" / pattern.format(cond=c, seed=s) / "metrics.json"
            if p.exists():
                m = json.load(open(p))
                row = {k: m.get(k, np.nan) for k in KEY_METRICS}
                row.update(condition=c, seed=s,
                           crps1=m.get("crps", {}).get("1", np.nan),
                           crps20=m.get("crps", {}).get("20", np.nan),
                           iet_cv=m.get("events", {}).get("iet_cv", np.nan),
                           agg_cv=m.get("agg", {}).get("cv", np.nan))
                rows.append(row)
    return pd.DataFrame(rows)


def summary_table(df):
    agg = df.groupby("condition").agg(["mean", "std", "count"])
    lines = ["| condition | " + " | ".join(
        f"{k} (mean±sd)" for k in KEY_METRICS + ["crps20"]) + " |",
        "|" + "---|" * (len(KEY_METRICS) + 2)]
    for c in df["condition"].unique():
        sub = df[df.condition == c]
        cells = [LABELS.get(c, c)]
        for k in KEY_METRICS + ["crps20"]:
            v = sub[k].astype(float)
            cells.append(f"{v.mean():.4f}±{v.std():.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired_tests(df, target="tpp", ref="base"):
    """Paired (by seed) Wilcoxon + t-test for each metric, target vs ref."""
    out = {}
    a = df[df.condition == target].set_index("seed").sort_index()
    b = df[df.condition == ref].set_index("seed").sort_index()
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return out
    for k in KEY_METRICS + ["crps20"]:
        x = a.loc[common, k].astype(float).values
        y = b.loc[common, k].astype(float).values
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 3 and not np.allclose(x[ok], y[ok]):
            t = sps.ttest_rel(x[ok], y[ok])
            try:
                w = sps.wilcoxon(x[ok], y[ok])
                wp = w.pvalue
            except ValueError:
                wp = np.nan
            out[k] = dict(delta=float(np.mean(x[ok] - y[ok])),
                          t_p=float(t.pvalue), wilcoxon_p=float(wp))
    return out


def fig_iet(conditions, seeds, pattern, ddir, figdir):
    refs = torch.load(ddir / "eval_refs.pt", weights_only=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    bins = np.linspace(0, 20, 80)
    axes[0].hist(refs["iets"], bins=bins, density=True, histtype="stepfilled",
                 alpha=0.25, color="k", label="ground truth")
    gt_sorted = np.sort(refs["iets"])
    for c in conditions:
        allsets = []
        for s in seeds:
            p = ROOT / "runs" / pattern.format(cond=c, seed=s) / "curves.npz"
            if p.exists():
                d = np.load(p)
                if len(d["iets"]):
                    allsets.append(d["iets"])
        if not allsets:
            continue
        iets = np.concatenate(allsets)
        axes[0].hist(iets, bins=bins, density=True, histtype="step",
                     color=COLORS.get(c), label=LABELS.get(c, c))
        q = np.linspace(0.001, 0.999, 300)
        axes[1].plot(np.quantile(gt_sorted, q), np.quantile(iets, q),
                     color=COLORS.get(c), lw=1.2, label=LABELS.get(c, c))
    axes[1].plot([0, 25], [0, 25], "k--", lw=0.8)
    axes[0].set_xlabel("inter-event time (MTU)")
    axes[0].set_ylabel("density")
    axes[1].set_xlabel("GT IET quantile")
    axes[1].set_ylabel("model IET quantile")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, figdir, "iet")
    plt.close(fig)


def fig_fano(conditions, seeds, pattern, ddir, figdir):
    refs = torch.load(ddir / "eval_refs.pt", weights_only=False)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    w = np.array(list(refs["fano"].keys()), dtype=float)
    ax.plot(w, list(refs["fano"].values()), "k-o", ms=3, label="ground truth")
    for c in conditions:
        curves = []
        for s in seeds:
            p = ROOT / "runs" / pattern.format(cond=c, seed=s) / "curves.npz"
            if p.exists():
                d = np.load(p)
                curves.append(d["fano"])
        if curves:
            arr = np.stack(curves)
            m, sd = arr.mean(0), arr.std(0)
            ax.plot(w, m, color=COLORS.get(c), marker="s", ms=2.5,
                    label=LABELS.get(c, c))
            ax.fill_between(w, m - sd, m + sd, color=COLORS.get(c), alpha=0.15)
    ax.axhline(1.0, color="gray", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("window size (MTU)")
    ax.set_ylabel("Fano factor")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, figdir, "fano")
    plt.close(fig)


def fig_hazard(conditions, seeds, pattern, ddir, figdir, dt):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from metrics import hazard_curve
    refs = torch.load(ddir / "eval_refs.pt", weights_only=False)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    lag = (np.arange(60) + 1) * dt
    ax.plot(lag, hazard_curve(refs["iets"], dt), "k-", lw=2, label="ground truth")
    for c in conditions:
        allsets = []
        for s in seeds:
            p = ROOT / "runs" / pattern.format(cond=c, seed=s) / "curves.npz"
            if p.exists():
                d = np.load(p)
                if len(d["iets"]):
                    allsets.append(d["iets"])
        if allsets:
            ax.plot(lag, hazard_curve(np.concatenate(allsets), dt),
                    color=COLORS.get(c), lw=1.2, label=LABELS.get(c, c))
    ax.set_xlabel("time since last event (MTU)")
    ax.set_ylabel("event hazard")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, figdir, "hazard")
    plt.close(fig)


def fig_dissociation(df, figdir):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for c in df.condition.unique():
        sub = df[df.condition == c]
        ax.scatter(sub["marg_w1"], sub["iet_w1"], s=22,
                   color=COLORS.get(c), label=LABELS.get(c, c))
    ax.set_xlabel("marginal error (state W1)")
    ax.set_ylabel("event-timing error (IET W1)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, figdir, "dissociation")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+",
                    default=["base", "tpp", "pois", "marg", "push", "det"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--pattern", default="ft_{cond}_s{seed}")
    ap.add_argument("--data_dir", default="data/dt02")
    args = ap.parse_args()
    ddir = ROOT / args.data_dir
    figdir = ROOT / "paper/figs"
    figdir.mkdir(parents=True, exist_ok=True)

    df = collect(args.conditions, args.seeds, args.pattern)
    if df.empty:
        print("no runs found")
        return
    df.to_csv(ROOT / "runs/summary.csv", index=False)
    md = summary_table(df)
    tests = {c: paired_tests(df, target=c)
             for c in args.conditions if c != "base"}
    with open(ROOT / "runs/summary.md", "w") as f:
        f.write(md + "\n\n## paired tests vs base\n" +
                json.dumps(tests, indent=2, default=float))
    print(md)
    print(json.dumps(tests, indent=2, default=float))

    refs = torch.load(ddir / "eval_refs.pt", weights_only=False)
    fig_iet(args.conditions, args.seeds, args.pattern, ddir, figdir)
    fig_fano(args.conditions, args.seeds, args.pattern, ddir, figdir)
    fig_hazard(args.conditions, args.seeds, args.pattern, ddir, figdir,
               refs["dt"])
    fig_dissociation(df, figdir)
    print(f"figures -> {figdir}")


if __name__ == "__main__":
    main()
