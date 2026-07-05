"""Generate all figures for both papers from committed results/*.json.

Paper A -> paper/figs/   Paper B -> paper2/figs/
Skips figures whose inputs are missing (prints a note).

Run: python experiments/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RESULTS

FIGA = ROOT / "paper" / "figs"
FIGB = ROOT / "paper2" / "figs"
FIGA.mkdir(parents=True, exist_ok=True)
FIGB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})

C = {"deep_ensemble": "#4477AA", "mc_dropout": "#66CCEE", "gp": "#228833",
     "iw_oracle_ensemble": "#CCBB44", "blind_two_head": "#AA3377",
     "heckman_ens": "#EE6677", "heckman_2s_ens": "#B2182B",
     "oracle": "#777777",
     "last_value": "#4477AA", "naive_pow3": "#CCBB44",
     "naive_pow3_vpen": "#228833", "eb_replay": "#EE6677"}
LBL = {"deep_ensemble": "Deep ensemble", "mc_dropout": "MC dropout",
       "gp": "GP", "iw_oracle_ensemble": "IW (oracle)",
       "blind_two_head": "Two-head (blind)", "heckman_ens": "Heckman (MLE)",
       "heckman_2s_ens": "Heckman (2-step)", "oracle": "Oracle",
       "skyline_ensemble": "Skyline (no sel.)",
       "last_value": "last value", "naive_pow3": "naive pow3",
       "naive_pow3_vpen": "pow3 $-\\kappa\\sigma$",
       "eb_replay": "EB corrected"}


def load(name):
    p = RESULTS / name
    if not p.exists():
        print(f"[skip] missing {name}")
        return None
    return json.loads(p.read_text())


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    print(f"[fig] {outdir / name}.pdf")


def agg(rows, method, rho, alpha, key):
    v = [r[key] for r in rows if r["method"] == method
         and r["rho"] == rho and r["alpha"] == alpha and key in r]
    return (np.mean(v), np.std(v)) if v else (np.nan, np.nan)


# ------------------------------------------------------------- Paper A


def figA1():
    ill = load("expA_fig1_illustration.json")
    e1 = load("expA_e1.json")
    if ill is None or e1 is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax = axes[0]
    g = np.array(ill["grid"])
    ax.scatter(ill["x_uns"], ill["y_uns"], s=2.5, c="#cccccc",
               label="unobserved $y$", rasterized=True)
    ax.scatter(ill["x_sel"], ill["y_sel"], s=2.5, c="#4477AA",
               label="selected sample", rasterized=True)
    ax.plot(g, ill["f0"], "k-", lw=1.4, label="$f_0(x)$")
    mu, sd = np.array(ill["ens_mu"]), np.array(ill["ens_sd"])
    ax.plot(g, mu, color=C["deep_ensemble"], lw=1.2)
    ax.fill_between(g, mu - 1.645 * sd, mu + 1.645 * sd, alpha=0.25,
                    color=C["deep_ensemble"], label="deep ensemble 90%")
    mu, sd = np.array(ill["heck_mu"]), np.array(ill["heck_sd"])
    ax.plot(g, mu, color=C["heckman_ens"], lw=1.2)
    ax.fill_between(g, mu - 1.645 * sd, mu + 1.645 * sd, alpha=0.25,
                    color=C["heckman_ens"], label="Heckman ens.\\ 90%")
    ax.axvspan(g[np.array(ill["prop_x_grid"]) <= 0.3].min(), 3.0,
               color="#EE6677", alpha=0.06)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(f"(a) $\\rho={ill['rho']}$, instrument present")
    ax.legend(loc="upper left", ncol=2, frameon=False, handletextpad=0.4)

    ax = axes[1]
    rows = e1
    rhos = sorted({r["rho"] for r in rows if r["method"] != "_meta"})
    for m in ["deep_ensemble", "gp", "iw_oracle_ensemble",
              "heckman_2s_ens", "heckman_ens"]:
        y = [agg(rows, m, r, 1.0, "picp90_against")[0] for r in rhos]
        e = [agg(rows, m, r, 1.0, "picp90_against")[1] for r in rhos]
        ax.errorbar(rhos, y, yerr=e, color=C[m], label=LBL[m], lw=1.3,
                    marker="o", ms=3, capsize=2)
    ax.axhline(0.90, color="k", ls=":", lw=0.9)
    ax.set_xlabel("selection-outcome correlation $\\rho$")
    ax.set_ylabel("coverage@90, selected-against")
    ax.set_title("(b) coverage vs $\\rho$ (instrument present)")
    ax.legend(frameon=False, loc="lower left")
    save(fig, FIGA, "fig1_setup")


def figA2():
    rows = load("expA_e1.json")
    if rows is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharex=True)
    rhos = sorted({r["rho"] for r in rows if r["method"] != "_meta"})
    methods = ["deep_ensemble", "iw_oracle_ensemble", "heckman_2s_ens",
               "heckman_ens"]
    for ax, alpha, ttl in ((axes[0], 1.0, "(a) instrument present"),
                           (axes[1], 0.0, "(b) instrument absent")):
        for m in methods:
            y = [agg(rows, m, r, alpha, "picp90_against")[0] for r in rhos]
            e = [agg(rows, m, r, alpha, "picp90_against")[1] for r in rhos]
            ax.errorbar(rhos, y, yerr=e, color=C[m], label=LBL[m], lw=1.3,
                        marker="o", ms=3, capsize=2)
        ax.axhline(0.90, color="k", ls=":", lw=0.9)
        ax.set_title(ttl)
        ax.set_xlabel("$\\rho$")
    axes[0].set_ylabel("coverage@90, selected-against")
    ax = axes[2]
    for alpha, ls in ((1.0, "-"), (0.0, "--")):
        for m in ["heckman_ens", "heckman_2s_ens"]:
            y = [agg(rows, m, r, alpha, "bias_f0_against")[0] for r in rhos]
            ax.plot(rhos, y, ls=ls, color=C[m], marker="o", ms=3, lw=1.3,
                    label=f"{LBL[m]}, {'w/' if alpha else 'no'} instr.")
        y = [agg(rows, "deep_ensemble", r, alpha, "bias_f0_against")[0]
             for r in rhos]
        ax.plot(rhos, y, ls=ls, color=C["deep_ensemble"], marker="o",
                ms=3, lw=1.3,
                label=f"{LBL['deep_ensemble']}, "
                      f"{'w/' if alpha else 'no'} instr.")
    ax.axhline(0.0, color="k", ls=":", lw=0.9)
    ax.set_title("(c) bias vs $f_0$, selected-against")
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel("bias")
    ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    axes[0].legend(frameon=False, loc="lower left")
    save(fig, FIGA, "fig2_honesty")


def figA3():
    rows = load("expA_e2.json")
    if rows is None:
        return
    datasets = sorted({r["dataset"] for r in rows})
    methods = ["deep_ensemble", "mc_dropout", "gp", "iw_oracle_ensemble",
               "blind_two_head", "heckman_2s_ens", "skyline_ensemble"]

    def mean_sd(ds, m, key):
        v = [r[key] for r in rows if r["dataset"] == ds
             and r["method"] == m and r["rho"] == 0.8 and key in r]
        return (np.mean(v), np.std(v)) if v else (np.nan, 0)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    width = 0.38
    xs = np.arange(len(methods))
    panels = [("ece_against", "(a) region-ECE, selected-against\n"
               "(lower better)", None),
              ("picp90_against", "(b) coverage@90,\nselected-against", 0.90),
              ("picp90_well", "(c) coverage@90, well-sampled\n"
               "(GP/dropout over-widen)", 0.90)]
    for ax, (key, title, hline) in zip(axes, panels):
        for j, ds in enumerate(datasets):
            vals = [mean_sd(ds, m, key)[0] for m in methods]
            errs = [mean_sd(ds, m, key)[1] for m in methods]
            ax.bar(xs + (j - 0.5) * width, vals, width=width, yerr=errs,
                   capsize=1.5, label=ds,
                   color=["#4477AA", "#EE6677"][j], alpha=0.9)
        if hline:
            ax.axhline(hline, color="k", ls=":", lw=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels([LBL.get(m, m) for m in methods], rotation=55,
                           ha="right", fontsize=6.0)
        ax.set_title(title)
    axes[0].legend(frameon=False, fontsize=6.5)
    save(fig, FIGA, "fig3_real")


def figA4():
    d = load("expA_e4.json")
    if d is None:
        return
    rows = []
    for task, res in d.items():
        if "pairwise" not in res:
            continue
        for p in res["pairwise"]:
            if "rho_mle" in p:
                rows.append((f"{p['anchor'][:14]}$\\to$"
                             f"{p['target'][:20]}", p["rho_mle"],
                             p.get("rho_mle_se"), p["boundary"], task))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.6, 0.32 * len(rows) + 0.9))
    ys = np.arange(len(rows))[::-1]
    for y, (lbl, rho, se, bnd, task) in zip(ys, rows):
        col = "#BBBBBB" if bnd else "#EE6677"
        if se and not bnd:
            ax.errorbar([rho], [y], xerr=[1.96 * se], color=col, capsize=2,
                        marker="o", ms=4)
        else:
            ax.plot([rho], [y], marker="s" if bnd else "o", color=col,
                    ms=4)
    ax.axvline(0, color="k", ls=":", lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax.set_xlabel("reporting-selection correlation $\\hat\\rho$ "
                  "(squares = boundary $|\\hat\\rho|=1$)")
    ax.set_xlim(-1.1, 1.1)
    save(fig, FIGA, "fig4_vignette")


# ------------------------------------------------------------- Paper B


def figB1():
    ill = load("expB_fig1_illustration.json")
    e0 = load("expB_e0.json")
    if ill is None or e0 is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))

    ax = axes[0]
    y_obs = np.array(ill["y_obs"])
    y_true = np.array(ill["y_true"])
    alive1 = np.array(ill["alive"])[:, 1]
    t = np.arange(1, y_obs.shape[1] + 1)
    for i in range(len(y_obs)):
        ax.plot(t, y_obs[i], color="#4477AA" if alive1[i] else "#cccccc",
                lw=0.5, alpha=0.6)
    for i, f in list(ill["survivor_fits"].items())[:5]:
        ax.plot(t, f["extrap"], color="#EE6677", lw=0.9, ls="--")
        ax.plot(t, y_true[int(i)], color="k", lw=0.8)
    for r in ill["rungs"]:
        ax.axvline(r, color="k", lw=0.5, ls=":")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val.\\ accuracy")
    ax.set_title("(a) SH survivors, naive fits (--)\nvs true curves (—)")

    rows = e0["rows"]
    sig_all = sorted({r["sigma"] for r in rows})
    ax = axes[1]
    m = [np.mean([r["rung0"]["selection_bias"] for r in rows
                  if r["sigma"] == s and r["eta"] == 3.0])
         for s in sig_all]
    sd = [np.std([r["rung0"]["selection_bias"] for r in rows
                  if r["sigma"] == s and r["eta"] == 3.0])
          for s in sig_all]
    ax.errorbar(sig_all, m, yerr=sd, color="#EE6677", marker="o", ms=3.5,
                capsize=2, lw=1.3)
    if e0.get("calibrated_sigma"):
        for k, v in e0["calibrated_sigma"].items():
            if v > 1e-6:
                ax.axvline(v, color="#228833", lw=0.9, ls="--")
                ax.text(v, ax.get_ylim()[1] * 0.92, " LCBench",
                        fontsize=6, color="#228833", rotation=90,
                        va="top")
    ax.axhline(0, color="k", ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("observation noise $\\sigma_{obs}$")
    ax.set_ylabel("survivor selection bias")
    ax.set_title("(b) bias vs noise (rung 1, $\\eta=3$)")

    ax = axes[2]
    etas = sorted({r["eta"] for r in rows})
    for k, rung in enumerate(["rung0", "rung1", "rung2"]):
        m = [np.mean([r[rung]["selection_bias"] for r in rows
                      if r["sigma"] == 0.02 and r["eta"] == e])
             for e in etas]
        ax.plot(etas, m, marker="o", ms=3.5, lw=1.3,
                label=f"rung {k + 1}",
                color=["#EE6677", "#CCBB44", "#4477AA"][k])
    ax.axhline(0, color="k", ls=":", lw=0.9)
    ax.set_xlabel("selection pressure $\\eta$")
    ax.set_title("(c) bias vs $\\eta$ ($\\sigma=0.02$)")
    ax.legend(frameon=False)
    save(fig, FIGB, "fig1_mechanism")


def figB2():
    d = load("expB_e1.json")
    if d is None:
        return
    rows = d["rows"]
    sigmas = sorted({r["sigma"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))

    ax = axes[0]
    ests = ["naive_mu_a", "eb_surv_naive_mu_a", "eb_surv_heck_1b_mu_a",
            "eb_surv_heck_mb_mu_a", "eb_all_mu_a"]
    lbl = ["naive\n(survivor mean)", "EB survivor\n(no corr.)",
           "+Heckman\n1 bracket", "+Heckman\n3 brackets",
           "EB all\n(MAR ref.)"]
    cols = ["#CCBB44", "#AA3377", "#66CCEE", "#EE6677", "#228833"]
    xs = np.arange(len(ests))
    sig0 = 0.01
    for i, k in enumerate(ests):
        v = [r["pop"][k] for r in rows if r["sigma"] == sig0]
        ax.errorbar(i, np.mean(v), yerr=np.std(v), marker="o", ms=4,
                    capsize=3, color=cols[i])
    truth = rows[0]["pop"]["true_mu_a"]
    ax.axhline(truth, color="k", ls=":", lw=1.0)
    ax.text(len(ests) - 0.6, truth + 0.003, "truth", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(lbl, fontsize=6.4)
    ax.set_ylabel("estimated population mean $\\mu_a$")
    ax.set_title(f"(a) population recovery ($\\sigma={sig0}$)")

    ax = axes[1]
    methods = ["naive_ls", "tobit", "eb_surv_naive", "eb_surv_heck_mb",
               "eb_all"]
    lbl2 = ["naive LS", "Tobit", "EB surv.", "EB+Heck (3br)", "EB all"]
    cols2 = ["#CCBB44", "#66CCEE", "#AA3377", "#EE6677", "#228833"]
    width = 0.15
    for i, m in enumerate(methods):
        for k in range(3):
            v = [r["extrap"][f"rung{k}"][m]["bias"] for r in rows
                 if r["sigma"] == sig0]
            ax.bar(k + (i - 2) * width, np.mean(v), width=width,
                   yerr=np.std(v), capsize=1.5, color=cols2[i],
                   label=lbl2[i] if k == 0 else None)
    ax.axhline(0, color="k", ls=":", lw=0.9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"rung {k+1}\n(t={t})" for k, t in
                        enumerate([4, 12, 36])], fontsize=7)
    ax.set_ylabel("survivor extrapolation bias")
    ax.set_title(f"(b) predicted final $-$ true ($\\sigma={sig0}$)")
    ax.legend(frameon=False, fontsize=6.2, ncol=2)
    save(fig, FIGB, "fig2_correction")


def figB3():
    d = load("expB_e2.json")
    if d is None:
        return
    rows = d["rows"]
    datasets = sorted({r["dataset"] for r in rows})
    methods = ["last_value", "naive_pow3", "naive_pow3_vpen", "eb_replay"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

    ax = axes[0]
    xs = np.arange(len(datasets))
    width = 0.19
    for i, m in enumerate(methods):
        v = [np.mean([r["pred@rung0"][m]["spearman"] for r in rows
                      if r["dataset"] == ds]) for ds in datasets]
        ax.bar(xs + (i - 1.5) * width, v, width=width, color=C[m],
               label=LBL[m])
    ax.set_xticks(xs)
    ax.set_xticklabels(datasets, rotation=40, ha="right", fontsize=6.3)
    ax.set_ylabel("Spearman (pred.\\ vs true final)")
    ax.set_title("(a) rank fidelity at rung 1")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[1]
    for i, m in enumerate(methods):
        v = [np.mean([r["pred@rung0"][m]["bias_top_decile"] for r in rows
                      if r["dataset"] == ds]) for ds in datasets]
        ax.bar(xs + (i - 1.5) * width, v, width=width, color=C[m])
    ax.axhline(0, color="k", ls=":", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(datasets, rotation=40, ha="right", fontsize=6.3)
    ax.set_ylabel("bias, top decile of predictions")
    ax.set_title("(b) winner's curse at rung 1")

    ax = axes[2]
    for i, m in enumerate(methods):
        for j, key in enumerate(["regret_completed", "regret_allpick"]):
            v = [r[m][key] for r in rows if r["n_pool"] == 200] or \
                [r[m][key] for r in rows]
            ax.bar(j + (i - 1.5) * width, np.mean(v), width=width,
                   color=C[m], yerr=np.std(v) / np.sqrt(len(v)),
                   capsize=1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["completed pick", "predicted pick\n(all configs)"],
                       fontsize=7)
    ax.set_ylabel("final regret (accuracy)")
    ax.set_title("(c) decision-level regret")
    save(fig, FIGB, "fig3_lcbench")


if __name__ == "__main__":
    figA1()
    figA2()
    figA3()
    figA4()
    figB1()
    figB2()
    figB3()
