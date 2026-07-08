"""Generate all paper figures from results/*.json. Never hand-edited numbers."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 150,
                     "savefig.bbox": "tight"})

E0 = json.loads((RES / "exp0_existence.json").read_text())
PAIR_ORDER = ["null", "seed", "noise", "distill", "prune", "difftask"]
PAIR_LABEL = {"null": "recoded\n(null)", "seed": "indep.\nseed",
              "noise": "noise\ntwin", "distill": "distilled\ntwin",
              "prune": "pruned\ntwin", "difftask": "diff.\ntask"}
C = {"cka": "#4878cf", "dsa": "#d65f5f", "dbar": "#2f7f4f", "belief": "#8172b2",
     "floor": "0.55"}


def seeds_of(pair, key, sub=None):
    rows = E0["results"][pair]
    if sub is None:
        return np.array([r[key] for r in rows])
    return np.array([r[key][sub] for r in rows])


def _bar_with_points(ax, xs, vals_list, color):
    means = [np.mean(v) for v in vals_list]
    ax.bar(xs, means, 0.62, color=color, alpha=0.75, zorder=2)
    for x, v in zip(xs, vals_list):
        ax.scatter(np.full(len(v), x) + np.linspace(-0.13, 0.13, len(v)), v,
                   s=8, color="k", zorder=3, alpha=0.7)


# ---------------------------------------------------------------- fig 1: E0 summary
def fig_e0():
    fig, axes = plt.subplots(3, 1, figsize=(5.4, 5.6), sharex=True)
    xs = np.arange(len(PAIR_ORDER))

    _bar_with_points(axes[0], xs, [seeds_of(p, "cka") for p in PAIR_ORDER], C["cka"])
    axes[0].set_ylabel("CKA")
    axes[0].axhline(0.90, color="k", lw=0.7, ls="--")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("static geometry (CKA): higher = more similar")

    _bar_with_points(axes[1], xs, [seeds_of(p, "dsa") for p in PAIR_ORDER], C["dsa"])
    nu, kap = E0["gate"]["nu_dsa"], E0["gate"]["kappa_dsa"]
    axes[1].axhline(nu + 0.25 * (kap - nu), color="k", lw=0.7, ls="--")
    axes[1].set_ylabel("DSA distance")
    axes[1].set_title("deterministic-conjugacy dynamics (DSA): lower = more similar")

    def amended_ratio(r, curve="dbar_curve", wall="k_wall"):
        ok = [row for row in r[curve]
              if 2 <= row["n"] <= r[wall] and row["dbar"] >= 2 * row["floor"]]
        if not ok:
            return 1.0          # no eligible row: at the same-process floor
        w = max(ok, key=lambda x: x["delta"])
        return w["dbar"] / max(w["floor"], 1e-12)

    dbar_ratio = [[amended_ratio(r) for r in E0["results"][p]] for p in PAIR_ORDER]
    _bar_with_points(axes[2], xs, dbar_ratio, C["dbar"])
    for x, p in zip(xs, PAIR_ORDER):
        b = [amended_ratio(r, "belief_curve", "belief_k_wall")
             for r in E0["results"][p] if "belief_curve" in r]
        if b:
            axes[2].scatter(np.full(len(b), x) + 0.28, b, s=16, marker="D",
                            color=C["belief"], zorder=3,
                            label="belief readout" if p == PAIR_ORDER[0] else None)
    axes[2].axhline(2.0, color="k", lw=0.7, ls="--")
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"$\bar d_{n^*}$ / floor")
    axes[2].set_title(r"process distance ($\bar d$): 1 = same-process floor")
    axes[2].set_xticks(xs, [PAIR_LABEL[p] for p in PAIR_ORDER])
    axes[2].legend(frameon=False, loc="upper left")
    fig.savefig(FIGS / "fig_e0_summary.pdf")
    plt.close(fig)


# --------------------------------------------------------- fig 2: dbar_n curves
def _curve_panel(ax, pair, key, title, wall_key="k_wall"):
    rows_seeds = [r[key] for r in E0["results"][pair]]
    ns = [row["n"] for row in rows_seeds[0]]
    vals = np.array([[row["dbar"] for row in rows] for rows in rows_seeds])
    floors = np.array([[row["floor"] for row in rows] for rows in rows_seeds])
    ax.errorbar(ns, vals.mean(0), yerr=vals.std(0), marker="o", ms=3,
                color=C["dbar"], label=r"$\bar d_n$(pair)", lw=1.2, capsize=2)
    ax.errorbar(ns, floors.mean(0), yerr=floors.std(0), marker="s", ms=3,
                color=C["floor"], label="same-process floor", lw=1, capsize=2)
    wall = np.mean([r[wall_key] for r in E0["results"][pair]])
    ax.axvline(wall, color="#b5442d", lw=0.9, ls=":",
               label=r"wall $\log_2 N/\hat h$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("block length n")
    ax.set_title(title)
    ax.legend(frameon=False)


def fig_curves():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
    _curve_panel(axes[0, 0], "noise", "dbar_curve",
                 "noise twin — emitted-token readout")
    _curve_panel(axes[0, 1], "noise", "belief_curve",
                 "noise twin — quantized-belief readout", wall_key="belief_k_wall")
    _curve_panel(axes[1, 0], "null", "dbar_curve",
                 "recoded null — emitted-token readout")
    _curve_panel(axes[1, 1], "difftask", "dbar_curve",
                 "different task — emitted-token readout")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\bar d_n$")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_curves.pdf")
    plt.close(fig)


# ------------------------------------------------------- fig 3: convergence (V3)
def fig_convergence():
    conv = E0["convergence"]["curves"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    Ns = [c["N"] for c in conv]
    show_ns = (2, 4, 8, 16, 32)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(show_ns)))
    for color, n in zip(cmap, show_ns):
        y = [next((r["delta"] for r in c["rows"] if r["n"] == n), np.nan)
             for c in conv]
        ax.plot(Ns, y, marker="o", ms=3, color=color, label=f"n={n}")
    ax.set_xscale("log")
    ax.set_xlabel("sample budget N (symbols)")
    ax.set_ylabel(r"$\bar d_n$ $-$ floor$_n$")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(frameon=False, ncol=2,
              title=f"pair: {E0['convergence']['pair']}")
    ax.set_title("separation vs. budget: real signal grows out of the floor")
    fig.savefig(FIGS / "fig_convergence.pdf")
    plt.close(fig)


# ------------------------------------------------------------- fig 4: E1 sweep
def fig_e1():
    E1 = json.loads((RES / "exp1_generality.json").read_text())
    fig, axes = plt.subplots(3, 2, figsize=(6.8, 5.8), sharex="col")

    for col, (rows, knob, xlabel) in enumerate(
            [(E1["noise"], "sigma", r"hidden-state noise $\sigma$"),
             (E1["prune"], "frac", "prune fraction")]):
        knobs = sorted({r[knob] for r in rows})
        def am_ratio(r, curve="dbar_curve", wall="k_wall"):
            ok = [row for row in r[curve]
                  if 2 <= row["n"] <= r[wall] and row["dbar"] >= 2 * row["floor"]]
            if not ok:
                return 1.0
            w = max(ok, key=lambda x: x["delta"])
            return w["dbar"] / max(w["floor"], 1e-12)

        panels = [("cka", "CKA", C["cka"], None),
                  ("dsa", "DSA distance", C["dsa"], None),
                  ("dbar", r"$\bar d_{n^*}$ / floor", C["dbar"], "ratio")]
        for rowi, (key, ylabel, color, mode) in enumerate(panels):
            ax = axes[rowi, col]
            if mode == "ratio":
                vals = np.array([[am_ratio(r) for r in rows if r[knob] == k]
                                 for k in knobs])
                bel = np.array([[am_ratio(r, "belief_curve", "belief_k_wall")
                                 for r in rows if r[knob] == k] for k in knobs])
                ax.errorbar(knobs, bel.mean(1), yerr=bel.std(1), marker="D",
                            ms=3, color=C["belief"], lw=1.1, capsize=2, ls="--",
                            label="belief readout")
                ax.set_yscale("log")
                ax.axhline(2.0, color="k", lw=0.6, ls="--")
                if col == 0:
                    ax.legend(frameon=False, fontsize=7)
            else:
                vals = np.array([[r[key] for r in rows if r[knob] == k]
                                 for k in knobs])
            m, s = vals.mean(1), vals.std(1)
            ax.errorbar(knobs, m, yerr=s, marker="o", ms=3.5, color=color,
                        lw=1.3, capsize=2,
                        label="emitted readout" if mode == "ratio" else None)
            if rowi == 0:
                ax.set_ylim(0, 1.05)
                ax.axhline(0.90, color="k", lw=0.6, ls="--")
            ax.set_ylabel(ylabel if col == 0 else "")
            if rowi == 2:
                ax.set_xlabel(xlabel)
    axes[0, 0].set_title("noise sweep")
    axes[0, 1].set_title("prune sweep")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_e1_sweep.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ fig 5: E2
def fig_e2():
    E2 = json.loads((RES / "exp2_transformer.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    groups = [("gm", "noise"), ("gm", "distill"), ("mess3", "noise"),
              ("mess3", "distill")]
    labels = ["GM\nnoise", "GM\ndistill", "MESS3\nnoise", "MESS3\ndistill"]
    xs = np.arange(len(groups))

    def vals(key, sub=None):
        out = []
        for t, p in groups:
            rows = E2[t][p]
            out.append([r[key] if sub is None else r[key][sub] for r in rows])
        return out

    _bar_with_points(axes[0], xs, vals("cka"), C["cka"])
    axes[0].set_ylabel("CKA")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(np.mean([c["cka"] for c in E2["cross"]]), color="k",
                    lw=0.8, ls=":", label="GM vs MESS3")
    _bar_with_points(axes[1], xs, vals("dsa"), C["dsa"])
    axes[1].set_ylabel("DSA distance")
    axes[1].axhline(np.mean([c["dsa"] for c in E2["cross"]]), color="k",
                    lw=0.8, ls=":", label="GM vs MESS3")
    ratio = [[r["plateau"]["dbar"] / max(r["plateau"]["floor"], 1e-12)
              for r in E2[t][p]] for t, p in groups]
    _bar_with_points(axes[2], xs, ratio, C["dbar"])
    axes[2].set_yscale("log")
    axes[2].axhline(2.0, color="k", lw=0.6, ls="--")
    axes[2].set_ylabel(r"$\bar d_{n^*}$ / floor")
    for ax in axes:
        ax.set_xticks(xs, labels)
    axes[0].legend(frameon=False)
    fig.suptitle("E2: transformer readout", y=1.04)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_e2.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_e0()
    fig_curves()
    fig_convergence()
    try:
        fig_e1()
    except FileNotFoundError:
        print("exp1 results missing, skipping fig_e1")
    try:
        fig_e2()
    except FileNotFoundError:
        print("exp2 results missing, skipping fig_e2")
    print("figures written to", FIGS)
