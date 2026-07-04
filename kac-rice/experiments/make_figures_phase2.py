"""Figures + tables for the Phase 2 (PINN crossing-budget) paper.

Reads results/phase2/*.json and the cached reference; writes paper/figs/ and
paper/tables/ with a p2_ prefix. Run after run_phase2.py.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT

RES = ROOT / "results" / "phase2"
FIGS = ROOT / "paper2" / "figs"
TABS = ROOT / "paper2" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "lines.linewidth": 1.4,
})

LABELS = {
    "vanilla": "vanilla PINN",
    "budget_ref": "+ budget (reference sim)",
    "budget_phys": "+ budget (physics prior)",
    "budget_scalar": "+ budget (scalar cap)",
    "grad_damp": "+ gradient damping",
    "curv_damp": "+ curvature damping",
    "fourier_pinn": "Fourier-feature PINN",
    "curriculum": "time curriculum",
    "causal": "causal weighting",
    "fourier_budget_ref": "Fourier PINN + budget",
}
COLORS = {
    "vanilla": "#555555",
    "budget_ref": "#d62728", "budget_phys": "#ff7f0e", "budget_scalar": "#e377c2",
    "grad_damp": "#2ca02c", "curv_damp": "#98df8a",
    "fourier_pinn": "#9467bd", "curriculum": "#1f77b4", "causal": "#17becf",
    "fourier_budget_ref": "#8c564b",
}


def load(suite):
    recs, seen = [], set()
    for p in sorted(RES.glob(f"{suite}*.json")):
        for r in json.loads(p.read_text()):
            key = (r.get("config"), r.get("seed"), r.get("iters"))
            if key in seen:
                continue
            seen.add(key)
            recs.append(r)
    return recs


def base_config(cfg):
    """strip weight suffixes: grad_damp_w0.3 -> grad_damp"""
    for stem in ("grad_damp", "curv_damp", "budget_ref_w"):
        if cfg.startswith(stem.rstrip("_w")) and "_w" in cfg:
            return cfg.split("_w")[0]
    return cfg


def all_records():
    recs = []
    for s in ("exp3", "exp4", "exp5", "exp6", "exp7", "abl"):
        recs += load(s)
    return recs


def best_damper_records(recs, stem):
    """For weight-swept dampers, pick the weight with best mean rel_l2
    (most favorable treatment of the null control)."""
    by_w = defaultdict(list)
    for r in recs:
        if r["config"].startswith(stem) and "_w" in r["config"]:
            by_w[r["config"]].append(r)
    if not by_w:
        return [], None
    best = min(by_w, key=lambda k: np.mean([x["rel_l2"] for x in by_w[k]]))
    return by_w[best], best


def fig_heatmaps():
    """|A(x,t)| space-time maps: reference, vanilla, budget-rescued."""
    z = np.load(RES / "reference.npz")
    x, t, A_ref = z["x"], z["t"], z["A"]
    panels = [("reference (spectral sim)", np.abs(A_ref))]
    for cfg, name in (("vanilla", "vanilla PINN"),
                      ("budget_ref", "+ crossing budget (ours)")):
        p = RES / f"pred_exp3_{cfg}.npy"
        if not p.exists():
            p = RES / f"pred_exp4_{cfg}.npy"
        if p.exists():
            panels.append((name, np.abs(np.load(p))))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 2.6),
                             sharey=True)
    for ax, (name, F) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(F, aspect="auto", origin="lower", cmap="viridis",
                       extent=[x[0], x[-1], t[0], t[-1]], vmin=0, vmax=1.3)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("$x$")
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("$t$")
    fig.colorbar(im, ax=axes, shrink=0.85, label="$|A|$")
    fig.savefig(FIGS / "p2_heatmaps.pdf")
    plt.close(fig)


def fig_traces():
    recs = all_records()
    show = ["vanilla", "budget_ref", "budget_phys", "budget_scalar"]
    gd, gd_name = best_damper_records(recs, "grad_damp")
    cd, cd_name = best_damper_records(recs, "curv_damp")
    groups = defaultdict(list)
    for r in recs:
        if base_config(r["config"]) in show:
            groups[base_config(r["config"])].append(r)
    if gd:
        groups["grad_damp"] = gd
    if cd:
        groups["curv_damp"] = cd

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for cfg, rs in groups.items():
        its = [h["iter"] for h in rs[0]["history"]]
        m = min(len(r["history"]) for r in rs)
        vals = np.array([[h["rel_l2"] for h in r["history"][:m]] for r in rs])
        vals = np.clip(vals, None, 10)
        lab = LABELS.get(cfg, cfg)
        if cfg == "grad_damp" and gd_name:
            lab += f" (best $w$)"
        if cfg == "curv_damp" and cd_name:
            lab += f" (best $w$)"
        ax.plot(its[:m], vals.mean(0), label=lab, color=COLORS.get(cfg))
        ax.fill_between(its[:m], vals.min(0), vals.max(0), alpha=0.15,
                        color=COLORS.get(cfg), lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative $L_2$ error")
    ax.legend(fontsize=6.5)
    fig.savefig(FIGS / "p2_traces.pdf")
    plt.close(fig)


def fig_crossings():
    """Crossing-density profile: budget vs vanilla vs rescued final fields."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from kacrice.cgle import budgets_from_reference

    z = np.load(RES / "reference.npz")
    x, A_ref = z["x"], z["A"]
    levels = np.linspace(-1.05, 1.05, 16)
    b = budgets_from_reference(A_ref, x, levels, eps=0.08)["re"]

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(levels, b, "k--", lw=1.2, label="budget $b_j$ (ref., x1.5 slack)")
    recs = all_records()
    for cfg in ("vanilla", "budget_ref"):
        rs = [r for r in recs if r["config"] == cfg and "crossings_re_mean" in r]
        if rs:
            arr = np.array([r["crossings_re_mean"] for r in rs])
            ax.plot(levels, arr.mean(0), color=COLORS.get(cfg),
                    label=f"{LABELS.get(cfg, cfg)} (final)")
            ax.fill_between(levels, arr.min(0), arr.max(0), alpha=0.15,
                            color=COLORS.get(cfg), lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("level $u$")
    ax.set_ylabel(r"spatial crossing density of $\mathrm{Re}\,A$")
    ax.legend(fontsize=6.5)
    fig.savefig(FIGS / "p2_crossings.pdf")
    plt.close(fig)


def tab_main():
    """Two column groups: front benchmark (exp3/exp4/exp5) and chaotic testbed
    (chaos suite). Rows: vanilla, three budget sources, gradient damping."""
    front = load("exp3") + load("exp4") + load("exp5")
    chaos = load("chaos")
    order = ["vanilla", "budget_ref", "budget_phys", "budget_scalar",
             "grad_damp"]

    def cell(recs, cfg):
        rs = [r for r in recs if base_config(r["config"]) == cfg]
        if cfg == "grad_damp" and rs:
            by_w = defaultdict(list)
            for r in rs:
                by_w[r["config"]].append(r)
            bestk = min(by_w, key=lambda k: np.mean(
                [x["rel_l2"] for x in by_w[k]]))
            rs = by_w[bestk]
        if not rs:
            return "--", ""
        rl = np.array([min(r["rel_l2"], 99.0) for r in rs])
        n = len(rs)
        val = (f"{rl.mean():.3f} $\\pm$ {rl.std():.3f}" if n > 1
               else f"{rl.mean():.3f}")
        return val, f" [{n}]"

    lines = ["\\begin{tabular}{lcc}", "\\toprule",
             " & front benchmark & BF-unstable testbed \\\\",
             "Config & rel.\\ $L_2$ [seeds] & rel.\\ $L_2$ [seeds] \\\\",
             "\\midrule"]
    for cfg in order:
        f_val, f_n = cell(front, cfg)
        c_val, c_n = cell(chaos, cfg)
        if f_val == "--" and c_val == "--":
            continue
        note = " (best $w$)" if cfg == "grad_damp" else ""
        lines.append(f"{LABELS.get(cfg, cfg)}{note} & {f_val}{f_n} & "
                     f"{c_val}{c_n} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABS / "p2_main.tex").write_text("\n".join(lines))


def fig_taxonomy():
    """The diagnostic plot: measured crossing profiles vs physical budgets on
    both testbeds. Every vanilla failure sits BELOW budget."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from kacrice.cgle import budgets_from_reference

    levels = np.linspace(-1.05, 1.05, 16)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    for ax, (ref_file, suites, title) in zip(axes, [
        ("reference.npz", ("exp3",), "front benchmark (ASPEN setting)"),
        ("reference_chaotic.npz", ("chaos",), "BF-unstable testbed"),
    ]):
        z = np.load(RES / ref_file)
        b = budgets_from_reference(z["A"], z["x"], levels, eps=0.08)["re"]
        ax.plot(levels, b, "k--", lw=1.3, label="physical budget $b_j$")
        prof_ref = None
        rows = []
        for s in suites:
            for r in load(s):
                if r["config"].startswith("vanilla"):
                    rows.append(r["crossings_re_mean"])
        if rows:
            arr = np.array(rows)
            ax.plot(levels, arr.mean(0), color="#d62728",
                    label="vanilla PINN, final field")
            ax.fill_between(levels, arr.min(0), arr.max(0), alpha=0.2,
                            color="#d62728", lw=0)
        ax.set_xlabel("level $u$")
        ax.set_ylabel("crossing density (per unit $x$)")
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "p2_taxonomy.pdf")
    plt.close(fig)


def fig_invitro():
    recs = load("invitro")
    lv = np.linspace(-1.2, 1.2, 16)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))

    # left: reconstructions (seed 0 stores pred)
    xg = np.linspace(-1, 1, 2048)
    gt = np.tanh(3 * xg) + 0.3 * np.sin(2 * np.pi * xg)
    ax = axes[0]
    ax.plot(xg, gt, color="#333", lw=1.0, label="target")
    for cfg, col, lab in (("mse", "#555555", "MSE only"),
                          ("mse_budget", "#d62728", "+ crossing budget")):
        for r in recs:
            if r["config"] == cfg and r.get("pred"):
                ax.plot(xg, r["pred"], color=col, lw=0.9, label=lab)
                break
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title("sparse-data SIREN interpolation (seed 0)", fontsize=8)
    ax.legend(fontsize=7)

    # right: crossing profiles vs budget, all seeds
    ax = axes[1]
    bud = np.array(recs[0]["budget"])
    ax.plot(lv, bud, "k--", lw=1.3, label="budget ($m{=}6$)")
    for cfg, col, lab in (("mse", "#555555", "MSE only"),
                          ("mse_budget", "#d62728", "+ budget")):
        arr = np.array([r["crossings"] for r in recs if r["config"] == cfg])
        ax.plot(lv, arr.mean(0), color=col, label=lab)
        ax.fill_between(lv, arr.min(0), arr.max(0), alpha=0.2, color=col, lw=0)
    ax.set_xlabel("level $u$")
    ax.set_ylabel("crossing density")
    ax.set_title("profiles: clamped to specification", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "p2_invitro.pdf")
    plt.close(fig)


def main():
    done, skipped = [], []
    for name, fn in [("fig_heatmaps", fig_heatmaps), ("fig_traces", fig_traces),
                     ("fig_crossings", fig_crossings), ("tab_main", tab_main),
                     ("fig_taxonomy", fig_taxonomy), ("fig_invitro", fig_invitro)]:
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
