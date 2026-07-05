"""All paper figures, regenerated from runs/ + results/ (no hand-drawn numbers).
Writes PDF+PNG into analysis-out and paper/figs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (ROOT, RUNS, RESULTS, SEEDS, SIZES_TRAIN, L_HOLDOUT,          # noqa: E402
                    L_TARGET, DX, DT_S, K_INRANGE_LO, analyze_measurement)
from floweval import fit_edmd_flows, predict_edmd_flow, R_MAX                    # noqa: E402

FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "analysis_out"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.2, "figure.dpi": 130,
                     "lines.linewidth": 1.1, "savefig.bbox": "tight"})
SIZE_COLORS = {22.0: "#c6dbef", 33.0: "#a6cee3", 44.0: "#6baed6", 66.0: "#3182bd",
               88.0: "#08519c", 176.0: "#e6550d", 1408.0: "#a63603"}


def savefig(fig, name):
    for d in (FIGS, OUT):
        fig.savefig(d / f"{name}.pdf")
        fig.savefig(d / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"fig: {name}")


def load_curves():
    return {L: analyze_measurement(RUNS / f"measure_L{L:g}.npz")
            for L in SIZES_TRAIN}


def fig1_snapshots():
    """Space-time snapshots at L=22 and L=1408 (the problem in one look)."""
    z22 = np.load(RUNS / "measure_L22.npz")
    z14 = np.load(RUNS / "measure_L1408.npz")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4),
                             gridspec_kw={"width_ratios": [1, 6.5]})
    for ax, z, L in ((axes[0], z22, 22), (axes[1], z14, 1408)):
        sn = z["snippet"][:800]
        ax.imshow(sn, aspect="auto", origin="lower", cmap="RdBu_r",
                  extent=[0, L, 0, sn.shape[0] * DT_S], vmin=-3, vmax=3,
                  interpolation="nearest")
        ax.set_xlabel("x")
        ax.set_title(f"L = {L}", fontsize=9)
    axes[0].set_ylabel("t")
    axes[1].set_yticklabels([])
    savefig(fig, "fig1_snapshots")


def fig2_dispersion(curves):
    """E0: gamma(k; L) and S(k; L) with 1/L flow insets on the common grid."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for L in SIZES_TRAIN + [L_HOLDOUT]:
        c = curves[L] if L in curves else analyze_measurement(RUNS / f"measure_L{L:g}.npz")
        g = c["gamma"].mean(axis=0)
        ge = c["gamma"].std(axis=0, ddof=1) / np.sqrt(3)
        axes[0].errorbar(c["k"], g, yerr=ge, fmt="o-", ms=2, lw=0.8,
                         color=SIZE_COLORS[L], label=f"L={L:g}")
        s = c["s_density"].mean(axis=0)
        axes[1].semilogy(c["k"], s, "o-", ms=2, lw=0.8, color=SIZE_COLORS[L])
    axes[0].set_xlabel("k"), axes[0].set_ylabel(r"$\gamma(k)$ (leading decay rate)")
    axes[0].legend(ncol=2, columnspacing=0.8)
    axes[1].set_xlabel("k"), axes[1].set_ylabel(r"$\hat S(k)$ (spectral density)")
    # inset-style panel: flow at two common k
    gate = json.load(open(RESULTS / "exp0_scaling.json"))["gate_s"]
    for qi, (q, style) in enumerate((("gamma", "o-"), ("s_density", "s--"))):
        rows = gate["per_quantity"][q]["rows"]
        for r in (rows[1], rows[4]):
            invL = 22.0 / np.asarray(SIZES_TRAIN)
            y = np.asarray(r["y"], float)
            norm = y[-1]
            axes[2].errorbar(invL, y / norm, yerr=np.asarray(r["se"]) / norm,
                             fmt=style, ms=3, lw=0.9,
                             label=f"{'γ' if q == 'gamma' else 'S'}, k={r['k']:.2f}")
            axes[2].plot(22.0 / L_HOLDOUT, r["holdout_true"] / norm, "k*", ms=7)
            axes[2].plot(22.0 / L_HOLDOUT, r["holdout_pred"] / norm, "rx", ms=6)
    axes[2].set_xlabel(r"$22/L$")
    axes[2].set_ylabel("normalized flow")
    axes[2].legend()
    axes[2].set_title("finite-size flow (★ true, × predicted)", fontsize=8)
    savefig(fig, "fig2_dispersion")


def fig3_headline(curves):
    """E2 at L=1408: predicted vs measured gamma(k), S(k), C(r)."""
    exp2 = json.load(open(RESULTS / "exp2_flow.json"))
    t = exp2["targets"]["1408"]
    k = np.asarray(t["k"])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    tg = np.asarray(t["truth_gamma"])
    tse = np.asarray(t["truth_gamma_se"])
    axes[0].fill_between(k, tg - 2 * tse, tg + 2 * tse, color="0.8", label="truth ±2se")
    for name, color in (("fitted_flow", "#d62728"), ("learned_flow", "#2ca02c")):
        axes[0].plot(k, t["methods"][name]["pred_gamma_mean"], color=color,
                     lw=1.0, label=name.replace("_", " "))
    axes[0].axvspan(0, K_INRANGE_LO, color="#fff3c9", zorder=0)
    axes[0].set_xlim(0, 2.2), axes[0].set_ylim(bottom=0)
    axes[0].set_xlabel("k"), axes[0].set_ylabel(r"$\gamma(k)$ at $L=1408$")
    axes[0].legend()
    ts = np.asarray(t["truth_s"])
    axes[1].semilogy(k, ts, color="0.4", lw=1.6, label="truth")
    axes[1].semilogy(k, t["methods"]["fitted_flow"]["pred_s_mean"], "#d62728",
                     lw=0.9, label="fitted flow")
    axes[1].axvspan(0, K_INRANGE_LO, color="#fff3c9", zorder=0)
    axes[1].set_xlim(0, 3.0)
    axes[1].set_xlabel("k"), axes[1].set_ylabel(r"$\hat S(k)$ at $L=1408$")
    axes[1].legend()
    # C(r)
    z14 = np.load(RUNS / "measure_L1408.npz")
    p_true = z14["p_mean"].mean(axis=0)
    r = np.arange(0.0, R_MAX + DX / 2, DX)
    kf = 2 * np.pi * np.arange(len(p_true)) / 1408.0
    c_true = np.cos(np.outer(r, kf)) @ p_true
    axes[2].plot(r, c_true, color="0.4", lw=1.6, label="truth")
    s_pred = np.asarray(t["methods"]["fitted_flow"]["pred_s_mean"])
    ok = np.isfinite(s_pred)
    c_pred = np.cos(np.outer(r, k[ok])) @ (s_pred[ok] * 2 * np.pi / 1408.0)
    axes[2].plot(r, c_pred, "#d62728", lw=0.9, ls="--", label="fitted flow")
    axes[2].set_xlabel("r"), axes[2].set_ylabel("C(r)")
    axes[2].legend()
    savefig(fig, "fig3_headline")


def fig4_baselines():
    """E3: flow vs nulls at L=1408 (bar chart over headline metrics)."""
    exp2 = json.load(open(RESULTS / "exp2_flow.json"))
    exp3 = json.load(open(RESULTS / "exp3_baselines.json"))
    methods = {
        "fitted flow": exp2["targets"]["1408"]["methods"]["fitted_flow"]["median"],
        "learned flow": exp2["targets"]["1408"]["methods"]["learned_flow"]["median"],
        "interp-88": exp3["targets"]["1408"]["methods"]["interp88"]["median"],
        "interp-44": exp3["targets"]["1408"]["methods"]["interp44"]["median"],
        "interp-22": exp3["targets"]["1408"]["methods"]["interp22"]["median"],
        "zero-shot": exp3["targets"]["1408"]["methods"]["zero_shot"]["median"],
        "EDMD 2000tu": exp3["targets"]["1408"]["methods"]["edmd_limited_T2000"]["median"],
    }
    sb = exp3["targets"]["1408"]["methods"].get("fitted_flow_smallbase")
    metrics = [("gamma_med_rel", r"$\gamma$ med rel err"),
               ("s_med_log10", r"$\hat S$ med $|\log_{10}|$"),
               ("c_rel_l2", "C(r) rel L2"),
               ("tau_med_rel", r"$\tau_e$ med rel err")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.0, 2.1))
    names = list(methods)
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#7fb3d5", "#c6dbef", "#9467bd",
              "#8c564b"]
    for ax, (m, lab) in zip(axes, metrics):
        vals = [methods[n].get(m, np.nan) for n in names]
        ax.barh(range(len(names))[::-1], vals, color=colors)
        ax.set_yticks(range(len(names))[::-1])
        ax.set_yticklabels(names if ax is axes[0] else [""] * len(names))
        ax.set_xlabel(lab)
    savefig(fig, "fig4_baselines")
    if sb:
        print("smallbase:", {m: round(sb["median"][m], 4) for m, _ in metrics})


def fig5_boundary():
    """E4: error vs L; odd-parity curves; band breakdown."""
    exp4 = json.load(open(RESULTS / "exp4_boundary.json"))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    ladder = exp4["ladder"]
    Ls = sorted({row["L"] for row in ladder})
    for name, color in (("fitted_flow", "#d62728"), ("learned_flow", "#2ca02c"),
                        ("interp88", "#1f77b4")):
        ys = [next(r["gamma_med_rel"] for r in ladder
                   if r["L"] == L and r["method"] == name) for L in Ls]
        axes[0].semilogx(Ls, ys, "o-", ms=3, color=color,
                         label=name.replace("_", " "))
    axes[0].set_xlabel("target L"), axes[0].set_ylabel(r"$\gamma$ med rel err")
    axes[0].legend()
    od = exp4["odd_parity"]["curves"]
    axes[1].plot(od["r"], od["c_per"], color="0.4", lw=1.5, label="periodic (true)")
    axes[1].plot(od["r"], od["c_odd"], "#1f77b4", lw=1.0, label="Dirichlet bulk")
    axes[1].plot(od["r"], od["c_flow"], "#d62728", ls="--", lw=1.0, label="flow pred.")
    axes[1].set_xlabel("r"), axes[1].set_ylabel(r"$C(r)/C(0)$"), axes[1].legend()
    bands = exp4["bands_1408"]
    x = np.arange(2)
    wid = 0.35
    for i, (name, color) in enumerate((("fitted_flow", "#d62728"),
                                       ("learned_flow", "#2ca02c"))):
        vals = [bands[name]["energy"]["gamma_med_rel"],
                bands[name]["microscale"]["gamma_med_rel"]]
        axes[2].bar(x + (i - 0.5) * wid, vals, wid, color=color,
                    label=name.replace("_", " "))
    axes[2].set_xticks(x), axes[2].set_xticklabels(["energy band", "microscale"])
    axes[2].set_ylabel(r"$\gamma$ med rel err"), axes[2].legend()
    savefig(fig, "fig5_boundary")


def fig6_lowk():
    """New-mode band at 1408: the honest extrapolation-in-k picture."""
    exp2 = json.load(open(RESULTS / "exp2_flow.json"))
    t = exp2["targets"]["1408"]
    k = np.asarray(t["k"])
    mask = k <= 4 * K_INRANGE_LO
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    tg = np.asarray(t["truth_gamma"])
    tse = np.asarray(t["truth_gamma_se"])
    ax.errorbar(k[mask], tg[mask], yerr=2 * tse[mask], fmt="ko", ms=3,
                label="truth ±2se")
    for name, color in (("fitted_flow", "#d62728"), ("learned_flow", "#2ca02c")):
        pg = np.asarray(t["methods"][name]["pred_gamma_mean"])
        ax.plot(k[mask], pg[mask], "o-", ms=2.5, color=color,
                label=name.replace("_", " "))
    ax.axvline(K_INRANGE_LO, color="0.6", ls=":")
    ax.text(K_INRANGE_LO * 1.05, ax.get_ylim()[1] * 0.9, "smallest trained k",
            fontsize=6.5, color="0.4")
    ax.set_xlabel("k"), ax.set_ylabel(r"$\gamma(k)$")
    ax.legend()
    savefig(fig, "fig6_lowk")


if __name__ == "__main__":
    curves = load_curves()
    fig1_snapshots()
    fig2_dispersion(curves)
    fig3_headline(curves)
    fig4_baselines()
    fig5_boundary()
    fig6_lowk()
