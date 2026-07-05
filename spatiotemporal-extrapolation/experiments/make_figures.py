"""All paper figures, regenerated from runs/ + results/ (no hand-drawn numbers)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (ROOT, RUNS, RESULTS, SIZES_TRAIN, L_HOLDOUT, DX, DT_S,        # noqa: E402
                    K_FULL_LO, K_INRANGE_LO, analyze_measurement)
from floweval import R_MAX                                                        # noqa: E402

FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "analysis_out"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.0, "figure.dpi": 130,
                     "lines.linewidth": 1.1, "savefig.bbox": "tight"})
SIZE_COLORS = {22.0: "#c6dbef", 44.0: "#6baed6", 66.0: "#3182bd", 88.0: "#08519c",
               176.0: "#e6550d"}
FLOW_C, INTERP_C, TRUTH_C = "#d62728", "#1f77b4", "0.35"


def savefig(fig, name):
    for d in (FIGS, OUT):
        fig.savefig(d / f"{name}.pdf")
        fig.savefig(d / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"fig: {name}")


def fig1_snapshots():
    z22, z14 = np.load(RUNS / "measure_L22.npz"), np.load(RUNS / "measure_L1408.npz")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4),
                             gridspec_kw={"width_ratios": [1, 6.5]})
    for ax, z, L in ((axes[0], z22, 22), (axes[1], z14, 1408)):
        sn = z["snippet"][:800]
        ax.imshow(sn, aspect="auto", origin="lower", cmap="RdBu_r",
                  extent=[0, L, 0, sn.shape[0] * DT_S], vmin=-3, vmax=3,
                  interpolation="nearest")
        ax.set_xlabel("x"), ax.set_title(f"L = {L}", fontsize=9)
    axes[0].set_ylabel("t"), axes[1].set_yticklabels([])
    savefig(fig, "fig1_snapshots")


def fig2_dispersion():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for L in SIZES_TRAIN + [L_HOLDOUT]:
        c = analyze_measurement(RUNS / f"measure_L{L:g}.npz")
        g, ge = c["gamma"].mean(0), c["gamma"].std(0, ddof=1) / np.sqrt(3)
        axes[0].errorbar(c["k"], g, yerr=ge, fmt="o-", ms=2, lw=0.8,
                         color=SIZE_COLORS[L], label=f"L={L:g}")
        axes[1].semilogy(c["k"], c["s_density"].mean(0), "o-", ms=2, lw=0.8,
                         color=SIZE_COLORS[L])
    axes[0].set_xlabel("k"), axes[0].set_ylabel(r"$\gamma(k)$ decay rate")
    axes[0].legend(ncol=2, columnspacing=0.8)
    axes[1].set_xlabel("k"), axes[1].set_ylabel(r"$\hat S(k)$ spectral density")
    gate = json.load(open(RESULTS / "exp0_scaling.json"))["gate_s"]
    for q, style, lab in (("gamma", "o-", r"$\gamma$"), ("s_density", "s--", r"$\hat S$")):
        for r in (gate["per_quantity"][q]["rows"][1], gate["per_quantity"][q]["rows"][4]):
            invL = 22.0 / np.asarray(SIZES_TRAIN)
            y = np.asarray(r["y"], float); norm = y[-1]
            axes[2].errorbar(invL, y / norm, yerr=np.asarray(r["se"]) / norm,
                             fmt=style, ms=3, lw=0.9, label=f"{lab}, k={r['k']:.2f}")
            axes[2].plot(22.0 / L_HOLDOUT, r["holdout_true"] / norm, "k*", ms=7)
            axes[2].plot(22.0 / L_HOLDOUT, r["holdout_pred"] / norm, "rx", ms=6)
    axes[2].set_xlabel(r"$22/L$"), axes[2].set_ylabel("normalized flow")
    axes[2].legend(), axes[2].set_title(r"$1/L$ flow ($\star$ true, $\times$ pred)", fontsize=8)
    savefig(fig, "fig2_dispersion")


def fig3_headline():
    """E2 at L=1408: flow AND interp-88 vs measured gamma(k), S(k), C(r)."""
    exp2 = json.load(open(RESULTS / "exp2_flow.json"))
    exp3 = json.load(open(RESULTS / "exp3_baselines.json"))
    t = exp2["targets"]["1408"]
    k = np.asarray(t["k"])
    curves = {L: analyze_measurement(RUNS / f"measure_L{L:g}.npz") for L in SIZES_TRAIN}
    sys.path.insert(0, str(Path(__file__).parent))
    from floweval import predict_interp_null
    interp = predict_interp_null(curves[88.0], 0, k)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    tg, tse = np.asarray(t["truth_gamma"]), np.asarray(t["truth_gamma_se"])
    axes[0].fill_between(k, tg - 2 * tse, tg + 2 * tse, color="0.85", label="truth $\\pm2$se")
    axes[0].plot(k, t["methods"]["fitted_flow"]["pred_gamma_mean"], color=FLOW_C,
                 lw=1.0, label="FSS flow")
    axes[0].plot(k, interp["gamma"], color=INTERP_C, lw=1.0, ls="--", label="interp-88")
    axes[0].axvspan(0, K_FULL_LO, color="#fff3c9", zorder=0)
    axes[0].set_xlim(0, 2.2), axes[0].set_ylim(0, 0.3)
    axes[0].set_xlabel("k"), axes[0].set_ylabel(r"$\gamma(k)$ at $L=1408$")
    axes[0].legend(loc="lower right")
    axes[1].semilogy(k, t["truth_s"], color=TRUTH_C, lw=1.6, label="truth")
    axes[1].semilogy(k, t["methods"]["fitted_flow"]["pred_s_mean"], FLOW_C, lw=0.9,
                     ls="--", label="FSS flow")
    axes[1].axvspan(0, K_FULL_LO, color="#fff3c9", zorder=0)
    axes[1].set_xlim(0, 3.0), axes[1].set_ylim(1e-4, 1e1)
    axes[1].set_xlabel("k"), axes[1].set_ylabel(r"$\hat S(k)$ at $L=1408$"), axes[1].legend()
    z14 = np.load(RUNS / "measure_L1408.npz")
    p_true = z14["p_mean"].mean(0)
    r = np.arange(0.0, R_MAX + DX / 2, DX)
    kf = 2 * np.pi * np.arange(len(p_true)) / 1408.0
    axes[2].plot(r, np.cos(np.outer(r, kf)) @ p_true, color=TRUTH_C, lw=1.6, label="truth")
    s_pred = np.asarray(t["methods"]["fitted_flow"]["pred_s_mean"])
    ok = np.isfinite(s_pred) & (k >= K_FULL_LO)
    axes[2].plot(r, np.cos(np.outer(r, k[ok])) @ (s_pred[ok] * 2 * np.pi / 1408.0),
                 FLOW_C, lw=0.9, ls="--", label="FSS flow (bulk)")
    axes[2].set_xlabel("r"), axes[2].set_ylabel("C(r)"), axes[2].legend()
    fig.subplots_adjust(wspace=0.32)
    savefig(fig, "fig3_headline")


def fig4_convergence_baselines():
    """The mechanism (convergence to limit) + the K2 verdict (flow vs nulls)."""
    exp3 = json.load(open(RESULTS / "exp3_baselines.json"))
    exp4 = json.load(open(RESULTS / "exp4_boundary.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    conv = exp4["convergence"]
    Ls = [r["L"] for r in conv]
    axes[0].loglog(Ls, [r["gamma_rel_to_limit"] for r in conv], "o-", color=FLOW_C,
                   label=r"$\gamma$")
    axes[0].loglog(Ls, [r["s_rel_to_limit"] for r in conv], "s--", color=INTERP_C,
                   label=r"$\hat S$")
    axes[0].axvline(88, color="0.6", ls=":")
    axes[0].set_xlabel("L"), axes[0].set_ylabel("median dist. to $L{=}1408$ limit")
    axes[0].set_title("why the null wins: fast convergence", fontsize=8), axes[0].legend()
    t3 = exp3["targets"]["1408"]["methods"]
    order = [("fitted_flow", "FSS flow"), ("interp88", "interp-88"),
             ("interp44", "interp-44"), ("interp22", "interp-22"),
             ("edmd_limited_T10000", "EDMD $T{=}10^4$")]
    metrics = [("gamma_med_rel", r"$\gamma$"), ("s_med_log10", r"$\hat S\,|\log_{10}|$"),
               ("c_rel_l2", "C(r)"), ("tau_med_rel", r"$\tau_e$")]
    x = np.arange(len(order))
    w = 0.2
    for j, (m, lab) in enumerate(metrics):
        vals = [t3[n]["median"].get(m, np.nan) for n, _ in order]
        axes[1].bar(x + (j - 1.5) * w, vals, w, label=lab)
    axes[1].set_xticks(x), axes[1].set_xticklabels([lab for _, lab in order],
                                                   rotation=30, ha="right", fontsize=6.5)
    axes[1].set_ylabel("median error (rel / $|\\log_{10}|$)")
    axes[1].set_title("L=1408: null beats flow (K2 fires)", fontsize=8), axes[1].legend(ncol=2)
    savefig(fig, "fig4_convergence_baselines")


def fig5_boundary():
    exp4 = json.load(open(RESULTS / "exp4_boundary.json"))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    lad = exp4["ladder"]
    Ls = sorted({r["L"] for r in lad})
    for name, color, lab in (("fitted_flow", FLOW_C, "FSS flow"),
                             ("interp88", INTERP_C, "interp-88")):
        ys = [next(r["gamma_med_rel"] for r in lad if r["L"] == L and r["method"] == name)
              for L in Ls]
        axes[0].semilogx(Ls, ys, "o-", ms=3, color=color, label=lab)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("target L"), axes[0].set_ylabel(r"$\gamma$ med rel err"), axes[0].legend()
    axes[0].set_ylim(bottom=0)
    od = exp4["odd_parity"]["curves"]
    axes[1].plot(od["r"], od["c_per"], color=TRUTH_C, lw=1.5, label="periodic")
    axes[1].plot(od["r"], od["c_odd"], INTERP_C, lw=1.0, label="Dirichlet bulk")
    axes[1].plot(od["r"], od["c_flow"], FLOW_C, ls="--", lw=1.0, label="flow pred.")
    axes[1].set_xlabel("r"), axes[1].set_ylabel(r"$C(r)/C(0)$")
    axes[1].set_title("boundary conditions", fontsize=8), axes[1].legend()
    b = exp4["bands_1408"]["fitted_flow"]
    axes[2].bar([0, 1], [b["energy"]["gamma_med_rel"], b["microscale"]["gamma_med_rel"]],
                color=[FLOW_C, "#8c564b"], width=0.6)
    axes[2].set_xticks([0, 1]), axes[2].set_xticklabels(["energy\nband", "microscale"])
    axes[2].set_ylabel(r"$\gamma$ med rel err"), axes[2].set_title("scale content", fontsize=8)
    savefig(fig, "fig5_boundary")


def fig6_lowk():
    """New-mode band at 1408: statics extrapolate in k, dynamics do not."""
    exp2 = json.load(open(RESULTS / "exp2_flow.json"))
    t = exp2["targets"]["1408"]
    k = np.asarray(t["k"])
    mask = k <= 6 * K_INRANGE_LO
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    tg, tse = np.asarray(t["truth_gamma"]), np.asarray(t["truth_gamma_se"])
    axes[0].errorbar(k[mask], tg[mask], yerr=2 * tse[mask], fmt="ko", ms=2.5, label="truth")
    axes[0].plot(k[mask], np.asarray(t["methods"]["fitted_flow"]["pred_gamma_mean"])[mask],
                 "o-", ms=2.5, color=FLOW_C, label="FSS flow")
    for kv, lab in ((K_INRANGE_LO, "$2\\pi/88$"), (K_FULL_LO, "$2\\pi/22$")):
        axes[0].axvline(kv, color="0.6", ls=":")
    axes[0].set_xlabel("k"), axes[0].set_ylabel(r"$\gamma(k)$")
    axes[0].set_title("dynamics: new modes NOT extrapolated", fontsize=8), axes[0].legend()
    axes[1].semilogy(k[mask], np.asarray(t["truth_s"])[mask], "ko", ms=2.5, label="truth")
    axes[1].semilogy(k[mask], np.asarray(t["methods"]["fitted_flow"]["pred_s_mean"])[mask],
                     "o-", ms=2.5, color=FLOW_C, label="FSS flow")
    for kv in (K_INRANGE_LO, K_FULL_LO):
        axes[1].axvline(kv, color="0.6", ls=":")
    axes[1].set_xlabel("k"), axes[1].set_ylabel(r"$\hat S(k)$")
    axes[1].set_title("statics: new modes extrapolated", fontsize=8), axes[1].legend()
    savefig(fig, "fig6_lowk")


if __name__ == "__main__":
    fig1_snapshots()
    fig2_dispersion()
    fig3_headline()
    fig4_convergence_baselines()
    fig5_boundary()
    fig6_lowk()
