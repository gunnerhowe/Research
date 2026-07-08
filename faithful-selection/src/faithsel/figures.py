"""Paper figures. Every figure regenerates from committed raw JSONL +
results JSONs; no hand-drawn numbers."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})

C_NAIVE = "#c0392b"
C_CORR = "#2471a3"
C_TRUE = "#1e8449"
C_GREY = "#7f8c8d"


def fig_confound(df, outcome="R_TE", path="paper/figs/fig1_confound.pdf"):
    """The confound, visually: latent reliance distribution by verbalization
    status, plus verbalization rate by instrument arm (first stage)."""
    d = df[df["parse_ok"] & (df["hint_type"] != "placebo")]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.7))

    ax = axes[0]
    r1 = d.loc[d["V"] == 1, outcome].values
    r0 = d.loc[d["V"] == 0, outcome].values
    bins = np.linspace(np.percentile(d[outcome], 1),
                       np.percentile(d[outcome], 99), 40)
    ax.hist(r0, bins=bins, alpha=0.6, density=True, color=C_GREY,
            label=f"V=0 (n={len(r0)})")
    ax.hist(r1, bins=bins, alpha=0.6, density=True, color=C_NAIVE,
            label=f"V=1 (n={len(r1)})")
    ax.axvline(r0.mean(), color=C_GREY, lw=1.5, ls="--")
    ax.axvline(r1.mean(), color=C_NAIVE, lw=1.5, ls="--")
    ax.set_xlabel(f"latent reliance {outcome} (log-odds)")
    ax.set_ylabel("density")
    ax.set_title("(a) reliance by verbalization")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    rates = [d.loc[d["z"] == 0, "V"].mean(), d.loc[d["z"] == 1, "V"].mean()]
    ns = [(d["z"] == 0).sum(), (d["z"] == 1).sum()]
    err = [1.96 * np.sqrt(r * (1 - r) / n) for r, n in zip(rates, ns)]
    ax.bar([0, 1], rates, yerr=err, color=[C_GREY, C_CORR], width=0.55,
           capsize=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["concise\n(Z=0)", "verbose\n(Z=1)"])
    ax.set_ylabel("P(V=1)")
    ax.set_title("(b) instrument first stage")

    ax = axes[2]
    means = [d.loc[d["z"] == 0, outcome].mean(),
             d.loc[d["z"] == 1, outcome].mean()]
    errs = [1.96 * d.loc[d["z"] == z, outcome].sem() for z in (0, 1)]
    ax.bar([0, 1], means, yerr=errs, color=[C_GREY, C_CORR], width=0.55,
           capsize=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["concise\n(Z=0)", "verbose\n(Z=1)"])
    ax.set_ylabel(f"mean {outcome}")
    ax.set_title("(c) exclusion: Z moves V, not R")
    fig.savefig(path)
    plt.close(fig)


def fig_correction(results: dict, path="paper/figs/fig2_correction.pdf",
                   outcome="R_TE"):
    """Naive vs corrected vs ground truth, population + hidden estimands,
    bootstrap CIs."""
    rep = results["outcomes"][outcome]
    est = rep["two_step"]["estimands"]
    tgt = rep["targets"]
    boot = rep["bootstrap"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    def ci(k):
        return (boot[k]["hi95"] - boot[k]["lo95"]) / 2 if k in boot else 0

    ax = axes[0]
    vals = [est["naive_selected"], est["naive_zerofill"],
            est["corrected_pop"]]
    errs = [ci("naive_selected"), ci("naive_zerofill"), ci("corrected_pop")]
    cols = [C_NAIVE, "#e67e22", C_CORR]
    ax.bar(range(3), vals, yerr=errs, color=cols, width=0.6, capsize=3)
    ax.axhline(tgt["true_pop"], color=C_TRUE, lw=2, ls="--",
               label="ground truth (all i)")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["naive\nE[R|V=1]", "naive\nzero-fill", "IMR-\ncorrected"])
    ax.set_ylabel("population mean reliance")
    ax.set_title("(a) population estimand")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    vals = [0.0, est["corrected_hidden"]]
    errs = [0.0, ci("corrected_hidden")]
    ax.bar(range(2), vals, yerr=errs, color=["#e67e22", C_CORR], width=0.5,
           capsize=3)
    ax.axhline(tgt["true_hidden"], color=C_TRUE, lw=2, ls="--",
               label="ground truth (V=0)")
    ax.set_xticks(range(2))
    ax.set_xticklabels(["naive\n(assume 0)", "IMR-\ncorrected"])
    ax.set_ylabel("hidden reliance E[R | V=0]")
    ax.set_title("(b) unverbalized reliance")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def fig_lens(df, path="paper/figs/fig3_lens.pdf"):
    """Pre-verbalization commitment: per-layer p(hinted letter), hinted vs
    unhinted prompts, split by later verbalization status."""
    d = df[df["parse_ok"] & (df["hint_type"] != "placebo")]
    if "lens_h" not in d.columns:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    hl = d["hint_letter"].map({"A": 0, "B": 1, "C": 2, "D": 3}).values

    def curves(rows, idx):
        M = np.stack([np.asarray(r, dtype=float)[:, i]
                      for r, i in zip(rows, idx)])
        return M.mean(axis=0)

    for v, color in ((1, C_NAIVE), (0, C_GREY)):
        m = d["V"].values == v
        ch = curves(d.loc[m, "lens_h"].tolist(), hl[m])
        cu = curves(d.loc[m, "lens_u"].tolist(), hl[m])
        L = np.arange(len(ch))
        ax.plot(L, ch, color=color, lw=1.8,
                label=f"hinted, V={v} (n={m.sum()})")
        ax.plot(L, cu, color=color, lw=1.2, ls=":")
    ax.axhline(0.25, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("layer (logit lens, no CoT)")
    ax.set_ylabel("p(hinted letter)")
    ax.set_title("pre-CoT commitment to the hinted answer")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def fig_sensitivity(results: dict, path="paper/figs/fig4_sensitivity.pdf",
                    outcome="R_TE"):
    """Corrected estimands across the fixed-rho grid vs ground truth."""
    rep = results["outcomes"][outcome]
    rows = rep["rho_sensitivity"]
    tgt = rep["targets"]
    rho_hat = rep["mle"]["rho"]
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    xs = [r["rho_fixed"] for r in rows]
    ax.plot(xs, [r["corrected_pop"] for r in rows], "o-", color=C_CORR,
            label="corrected population mean")
    ax.plot(xs, [r["corrected_hidden"] for r in rows], "s-", color="#8e44ad",
            label="corrected hidden mean")
    ax.axhline(tgt["true_pop"], color=C_TRUE, ls="--", lw=1.5,
               label="true population")
    ax.axhline(tgt["true_hidden"], color=C_TRUE, ls=":", lw=1.5,
               label="true hidden")
    ax.axvline(rho_hat, color=C_NAIVE, lw=1.2, alpha=0.7,
               label=f"$\\hat\\rho$ = {rho_hat:.2f}")
    ax.set_xlabel(r"fixed $\rho$")
    ax.set_ylabel("estimate")
    ax.set_title(r"$\rho$-sensitivity of the correction")
    ax.legend(frameon=False, fontsize=6.5)
    fig.savefig(path)
    plt.close(fig)


def fig_models(results_list: list[dict], labels: list[str],
               path="paper/figs/fig5_models.pdf", outcome="R_TE"):
    """rho with CI + gate outcomes across models (and per hint type for the
    primary model)."""
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ys, names = [], []
    for res, lab in zip(results_list, labels):
        rep = res["outcomes"][outcome]
        lo, hi = rep["rho_wald"]["rho_ci95"]
        ys.append((rep["mle"]["rho"], lo, hi))
        names.append(lab)
    y = np.arange(len(ys))[::-1]
    for yi, (r, lo, hi) in zip(y, ys):
        ax.plot([lo, hi], [yi, yi], color=C_CORR, lw=2)
        ax.plot([r], [yi], "o", color=C_CORR, ms=5)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(r"$\hat\rho$ (MLE, 95% CI)")
    ax.set_title("verbalization-selection correlation")
    fig.savefig(path)
    plt.close(fig)


def load_results(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
