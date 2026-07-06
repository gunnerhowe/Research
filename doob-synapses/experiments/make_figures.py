"""Generate all paper figures from results/*.json. Saves PDF+PNG to paper/figs/.
Robust to missing JSONs (skips figures whose inputs are absent)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "legend.fontsize": 8, "figure.dpi": 130, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 1.8,
})
C = {"doob": "#c1121f", "ou": "#0077b6", "ewc": "#588157", "mesu": "#7209b7",
     "none": "#8d99ae", "bss2": "#e07a00"}


def load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{stem}.{ext}")
    plt.close(fig)
    print(f"[fig] {stem}")


def _band(ax, x, m, sd, color, label, **kw):
    m, sd = np.array(m), np.array(sd)
    ax.plot(x, m, color=color, label=label, marker="o", ms=3.5, **kw)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15, lw=0)


# --------------------------------------------------------------------------- #
def fig1_gate_f():
    e0 = load("exp0_falsifier.json")
    if not e0:
        return
    sig = e0["config"]["sigmas"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m in ("none", "ou", "ewc", "mesu", "doob"):
        d = e0["methods"][m]
        lw = 2.6 if m == "doob" else 1.4
        _band(ax, sig, d["retention_mean"], d["retention_sd"], C[m],
              m.upper() if m != "doob" else "Doob (ours)", lw=lw,
              zorder=5 if m == "doob" else 2)
    iu = e0["methods"]["doob"]["inverted_u"]
    sstar = iu["sigma_star"]
    ax.axvline(sstar, color=C["doob"], ls=":", lw=1, alpha=0.7)
    ax.annotate(f"$\\sigma^*={sstar:g}$\n+{100*iu['lift_over_zero']:.1f} pts",
                xy=(sstar, iu["ret_at_peak"]),
                xytext=(sstar + 0.06, iu["ret_at_peak"] + 0.005),
                fontsize=8, color=C["doob"])
    ax.set_xlabel("intrinsic-noise amplitude $\\sigma$")
    ax.set_ylabel("retention (mean past-task acc.)")
    ax.set_title("GATE F: barrier-conditioning yields a retention inverted-U\n"
                 "matched anchored-drift controls are monotone")
    ax.legend(loc="upper right", ncol=2)
    save(fig, "fig1_gate_f")


def fig2_isolation():
    e1 = load("exp1_isolation.json")
    if not e1:
        return
    sig = e1["config"]["sigmas"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    # (a) kappa scan
    ax = axes[0]
    kaps = e1["config"]["kappas"]
    cols = plt.cm.viridis(np.linspace(0.15, 0.9, len(kaps)))
    for k, col in zip(kaps, cols):
        d = e1["kappa_scan"][f"{k}"]
        ax.plot(sig, d["retention_mean"], color=col, marker="o", ms=3,
                label=f"$\\kappa$={k}")
    ax.set_xlabel("$\\sigma$"); ax.set_ylabel("retention")
    ax.set_title("(a) steering strength $\\kappa$: 0 = unconditioned OU")
    ax.legend(title="conditioning")
    # (b) barrier scan
    ax = axes[1]
    barrs = e1["config"]["barriers"]
    cols = plt.cm.plasma(np.linspace(0.15, 0.85, len(barrs)))
    for b, col in zip(barrs, cols):
        d = e1["barrier_scan"][f"{b}"]
        iu = d["inverted_u"]
        ax.plot(sig, d["retention_mean"], color=col, marker="o", ms=3,
                label=f"$b_0$={b} ($\\sigma^*$={iu['sigma_star']:g})")
        ax.axvline(iu["sigma_star"], color=col, ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("$\\sigma$"); ax.set_ylabel("retention")
    ax.set_title("(b) barrier scale $b_0$: optimum tracks the barrier")
    ax.legend(title="barrier")
    save(fig, "fig2_isolation")


def fig3_bss2():
    e2, e0 = load("exp2_bss2.json"), load("exp0_falsifier.json")
    if not e2:
        return
    sig = e2["config"]["sigmas"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    # (a) device-faithful vs white
    ax = axes[0]
    df = e2["device_faithful"]
    _band(ax, sig, df["retention_mean"], df["retention_sd"], C["bss2"],
          "BSS-2 noise (emul.)")
    if e0:
        dd = e0["methods"]["doob"]
        _band(ax, sig, dd["retention_mean"], dd["retention_sd"], C["doob"],
              "white noise (sim.)")
    ax.set_xlabel("effective noise $\\sigma_{\\mathrm{eff}}$")
    ax.set_ylabel("retention")
    ax.set_title("(a) inverted-U survives device-faithful\n"
                 "BSS-2 noise (colored+6-bit+FP) — EMULATION")
    ax.legend()
    # (b) color scan
    ax = axes[1]
    colors = e2["config"]["colors"]
    lifts = [e2["color_scan"][f"{c}"]["inverted_u"]["lift_over_zero"] for c in colors]
    us = [e2["color_scan"][f"{c}"]["inverted_u"]["inverted_u"] for c in colors]
    bars = ax.bar([str(c) for c in colors], [100 * l for l in lifts],
                  color=[C["bss2"] if u else "#bbbbbb" for u in us])
    ax.set_xlabel("temporal color of device noise (AR(1) $\\rho$)")
    ax.set_ylabel("retention lift at $\\sigma^*$ (pts)")
    ax.set_title("(b) where the mechanism holds vs noise color")
    for b, u in zip(bars, us):
        ax.annotate("U" if u else "flat", (b.get_x() + b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
    save(fig, "fig3_bss2")


def fig4_baselines():
    e3 = load("exp3_baselines.json")
    if not e3:
        return
    order = ["doob*", "mesu", "ewc_best", "ou", "benna_fusi", "replay", "none"]
    labels = {"doob*": "Doob*", "mesu": "MESU", "ewc_best": "EWC", "ou": "OU/OUA",
              "benna_fusi": "Benna-Fusi", "replay": "replay", "none": "naive"}
    M = e3["methods"]
    order = [o for o in order if o in M]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    # (a) retention bars
    ax = axes[0]
    means, sds = [], []
    for o in order:
        r = M[o].get("retention", None)
        means.append(M[o]["retention_mean"])
        sds.append(np.std(r, ddof=1) if isinstance(r, list) and len(r) > 1 else 0)
    cols = [C["doob"] if o == "doob*" else "#8d99ae" for o in order]
    ax.bar([labels[o] for o in order], means, yerr=sds, color=cols, capsize=3)
    ax.set_ylabel("retention"); ax.set_ylim(0.45, max(means) + 0.06)
    ax.set_title("(a) matched-budget retention (8 seeds)")
    ax.tick_params(axis="x", rotation=30)
    # (b) retention vs energy
    ax = axes[1]
    for o in order:
        e = M[o].get("energy_pj_gpu", None)
        if e is None:
            continue
        ax.scatter(e, M[o]["retention_mean"], color=C["doob"] if o == "doob*" else "#0077b6",
                   s=40, zorder=3)
        ax.annotate(labels[o], (e, M[o]["retention_mean"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    if "energy_pj_bss2" in M["doob*"]:
        ax.scatter(M["doob*"]["energy_pj_bss2"], M["doob*"]["retention_mean"],
                   color=C["bss2"], marker="*", s=160, zorder=4)
        ax.annotate("Doob on BSS-2\n(emul., noise free)",
                    (M["doob*"]["energy_pj_bss2"], M["doob*"]["retention_mean"]),
                    fontsize=7, xytext=(4, -14), textcoords="offset points", color=C["bss2"])
    ax.set_xscale("log"); ax.set_xlabel("energy / consolidation step (pJ, model)")
    ax.set_ylabel("retention")
    ax.set_title("(b) retention vs compute-energy")
    save(fig, "fig4_baselines")


def fig5_modality():
    e4 = load("exp4_modality.json")
    if not e4:
        return
    sig = e4["config"]["sigmas"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    # (a) Yin-Yang inverted-U
    ax = axes[0]
    yy = e4["yin_yang"]
    _band(ax, sig, yy["doob"]["retention_mean"], yy["doob"]["retention_sd"],
          C["doob"], "Doob (ours)", lw=2.4)
    _band(ax, sig, yy["ou"]["retention_mean"], yy["ou"]["retention_sd"],
          C["ou"], "OU (control)")
    ax.set_xlabel("$\\sigma$"); ax.set_ylabel("retention")
    ax.set_title("(a) continual Yin-Yang (2nd modality)")
    ax.legend()
    # (b) sigma* / lift vs task similarity
    ax = axes[1]
    keys = sorted(e4["similarity_scan"], key=float)
    degs = [e4["similarity_scan"][k]["max_rot_deg"] for k in keys]
    sstar = [e4["similarity_scan"][k]["inverted_u"]["sigma_star"] for k in keys]
    lift = [100 * e4["similarity_scan"][k]["inverted_u"]["lift_over_zero"] for k in keys]
    ax.plot(degs, sstar, "o-", color=C["doob"], label="$\\sigma^*$")
    ax.set_xlabel("per-task rotation (deg) = dissimilarity")
    ax.set_ylabel("$\\sigma^*$", color=C["doob"])
    ax2 = ax.twinx(); ax2.plot(degs, lift, "s--", color=C["mesu"], label="lift")
    ax2.set_ylabel("retention lift (pts)", color=C["mesu"]); ax2.grid(False)
    ax.set_title("(b) noise-optimum vs task similarity")
    save(fig, "fig5_modality")


def fig6_mechanism():
    e0 = load("exp0_falsifier.json")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    # (a) Doob restoring force ~ sigma^2 score, illustrative
    ax = axes[0]
    b = 0.2
    z = np.linspace(-0.98, 0.98, 400) * b
    score = -(math.pi / (2 * b)) * np.tan(np.pi * z / (2 * b))
    for s in (0.02, 0.05, 0.1, 0.2):
        ax.plot(z, s**2 * score, label=f"$\\sigma$={s}")
    ax.set_ylim(-0.6, 0.6)
    ax.axvline(b, color="k", ls=":", lw=0.8); ax.axvline(-b, color="k", ls=":", lw=0.8)
    ax.set_xlabel("weight displacement $w-\\mu$")
    ax.set_ylabel("Doob steering drift $\\sigma^2\\,\\partial_w\\log h$")
    ax.set_title("(a) noise-amplified barrier steering ($\\propto\\sigma^2$)")
    ax.legend()
    # (b) decomposition: retention (U) = protection up, plasticity down
    ax = axes[1]
    if e0:
        sig = e0["config"]["sigmas"]
        d = e0["methods"]["doob"]
        ax.plot(sig, d["retention_mean"], "-o", ms=3, color=C["doob"], label="retention")
        ax.plot(sig, d["plasticity_mean"], "-s", ms=3, color="#0077b6", label="plasticity")
        ax.plot(sig, 1 - np.array(d["forgetting_mean"]), "-^", ms=3, color="#588157",
                label="1 - forgetting")
        istar = d["inverted_u"]["i_peak"]
        ax.axvline(sig[istar], color="k", ls=":", lw=0.8, alpha=0.6)
        ax.set_xlabel("$\\sigma$"); ax.set_ylabel("metric")
        ax.set_title("(b) retention tracks a forgetting minimum;\nplasticity ~ noise-insensitive")
        ax.legend()
    save(fig, "fig6_mechanism")


def main():
    fig1_gate_f(); fig2_isolation(); fig3_bss2()
    fig4_baselines(); fig5_modality(); fig6_mechanism()


if __name__ == "__main__":
    main()
