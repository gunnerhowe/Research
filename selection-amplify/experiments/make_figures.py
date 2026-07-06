"""Generate paper figures from results/*.json. Saves PDF+PNG into paper/figs/.

Run:  python experiments/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C

FIGS = C.ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
BETAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def load(name):
    p = C.RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=130)
    plt.close(fig)
    print(f"[fig] {name}")


def _agg(rows, beta, method, key):
    v = [r[key] for r in rows if r["beta"] == beta and r["method"] == method
         and np.isfinite(r.get(key, float("nan")))]
    return (np.mean(v), np.std(v)) if v else (np.nan, np.nan)


def fig1_curve():
    e2 = load("exp2_curve.json")
    if not e2:
        return
    rows = e2["rows"]
    iox = [np.mean([r["iox"] for r in rows if r["beta"] == b
                    and r["method"] == "_diag"]) for b in BETAS]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    styles = {"B0_obs": ("B0 obs-only", "0.6", "o"),
              "B3_reweight": ("B3 $1/\\hat s$ reweight", "tab:green", "s"),
              "decoy_rotate": ("misdirected decoy", "tab:orange", "^"),
              "method": ("method (ours)", "tab:red", "o"),
              "B4_oracle": ("B4 oracle", "0.2", "*")}
    for meth, (lab, col, mk) in styles.items():
        mu = [_agg(rows, b, meth, "acc_slice")[0] for b in BETAS]
        sd = [_agg(rows, b, meth, "acc_slice")[1] for b in BETAS]
        ax[0].errorbar(iox, mu, yerr=sd, label=lab, color=col, marker=mk, ms=4,
                       capsize=2, lw=1.5)
    ax[0].set_xlabel("selection entropy $I(O;X)$ (bits)")
    ax[0].set_ylabel("censored-slice accuracy")
    ax[0].set_title("(a) slice accuracy")
    ax[0].legend(fontsize=6.5, loc="lower left")

    # (b) THE discriminating panel: method - decoy gap on the slice
    gmu, glo, ghi = [], [], []
    from selamp.stats import bootstrap_ci
    for b in BETAS:
        m = {r["seed"]: r["acc_slice"] for r in rows if r["beta"] == b and r["method"] == "method"}
        d = {r["seed"]: r["acc_slice"] for r in rows if r["beta"] == b and r["method"] == "decoy_rotate"}
        ss = sorted(set(m) & set(d))
        gap = np.array([m[s] - d[s] for s in ss])
        gmu.append(gap.mean() * 100)
        lo, hi = bootstrap_ci(gap)
        glo.append(lo * 100); ghi.append(hi * 100)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].plot(iox, gmu, "-o", color="tab:red", ms=4)
    ax[1].fill_between(iox, glo, ghi, color="tab:red", alpha=0.2)
    ax[1].set_xlabel("selection entropy $I(O;X)$ (bits)")
    ax[1].set_ylabel("method $-$ decoy slice acc (pts)")
    ax[1].set_title("(b) selection-targeting gap (discriminator)")

    # (c) full-test accuracy
    for meth, (lab, col, mk) in styles.items():
        mu = [_agg(rows, b, meth, "acc_full")[0] for b in BETAS]
        ax[2].plot(iox, mu, marker=mk, color=col, ms=4, lw=1.5, label=lab)
    ax[2].set_xlabel("selection entropy $I(O;X)$ (bits)")
    ax[2].set_ylabel("full-test accuracy")
    ax[2].set_title("(c) full-test accuracy")
    save(fig, "fig1_curve")


def fig2_e0():
    e0 = load("exp0_sanity.json")
    if not e0:
        return
    rows = e0["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for tb, col in [("two_moons", "tab:blue"), ("eight_gaussians", "tab:green"),
                    ("pinwheel", "tab:orange")]:
        g = [np.nanmean([r["spearman_global"] for r in rows if r["testbed"] == tb and r["beta"] == b]) for b in BETAS]
        c = [np.nanmean([r["spearman_complement"] for r in rows if r["testbed"] == tb and r["beta"] == b]) for b in BETAS]
        ax[0].plot(BETAS, g, "-o", color=col, ms=3, label=f"{tb} global")
        ax[0].plot(BETAS, c, "--^", color=col, ms=3, label=f"{tb} complement")
    ax[0].axhline(0.7, color="k", ls=":", lw=0.8)
    ax[0].set_xlabel("$\\beta$"); ax[0].set_ylabel("Spearman($\\hat s$, $s_\\beta$)")
    ax[0].set_title("(a) selector recovery"); ax[0].legend(fontsize=5.5)
    for tb, col in [("two_moons", "tab:blue"), ("eight_gaussians", "tab:green"),
                    ("pinwheel", "tab:orange")]:
        ia = [np.nanmean([r["IOX_analytic"] for r in rows if r["testbed"] == tb and r["beta"] == b]) for b in BETAS]
        ip = [np.nanmean([r["IOX_plugin"] for r in rows if r["testbed"] == tb and r["beta"] == b]) for b in BETAS]
        ax[1].plot(ia, ip, "-o", color=col, ms=3, label=tb)
    lim = [0, max(0.8, ax[1].get_xlim()[1])]
    ax[1].plot(lim, lim, "k:", lw=0.8)
    ax[1].set_xlabel("$I(O;X)$ analytic"); ax[1].set_ylabel("$I(O;X)$ from $\\hat s$")
    ax[1].set_title("(b) entropy axis recovery"); ax[1].legend(fontsize=6)
    save(fig, "fig2_e0")


def fig3_collar():
    """Showcase: rebuild one stack and visualize the collar targeting."""
    from selamp import data
    from selamp.bridge import generate_labeled
    st = C.Stack(C.PRIMARY, 4.0, 0)
    Xm, ym, _ = generate_labeled(st.dm, st.sel, st.gate, 800, st.cfg, seed=0)
    Xd, yd, _ = generate_labeled(st.dm, st.sel, st.gate, 800, st.cfg,
                                 decoy="rotate", seed=0)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    Xt = st.c.X_test
    s_true = data.selection_prob(Xt, 4.0, C.PRIMARY)
    for a in ax:
        a.set_xlim(Xt[:, 0].min() - .3, Xt[:, 0].max() + .3)
        a.set_ylim(Xt[:, 1].min() - .3, Xt[:, 1].max() + .3)
        a.set_xticks([]); a.set_yticks([])
    sc = ax[0].scatter(Xt[:, 0], Xt[:, 1], c=s_true, s=4, cmap="viridis")
    ax[0].scatter(st.c.X_obs[:, 0], st.c.X_obs[:, 1], s=1, c="k", alpha=0.15)
    ax[0].set_title("(a) population (color=$s_\\beta$), corpus=black")
    plt.colorbar(sc, ax=ax[0], fraction=0.046)
    ax[1].scatter(Xt[:, 0], Xt[:, 1], s=3, c="0.8")
    ax[1].scatter(Xm[:, 0], Xm[:, 1], s=5, c="tab:red", alpha=0.6, label="method synth")
    ax[1].set_title("(b) method: fills the censored collar"); ax[1].legend(fontsize=7)
    ax[2].scatter(Xt[:, 0], Xt[:, 1], s=3, c="0.8")
    ax[2].scatter(Xd[:, 0], Xd[:, 1], s=5, c="tab:orange", alpha=0.6, label="decoy synth")
    ax[2].set_title("(c) misdirected decoy: wrong region"); ax[2].legend(fontsize=7)
    save(fig, "fig3_collar")


def fig4_squeeze():
    e4 = load("exp4_squeeze.json")
    if not e4:
        return
    sq = e4["squeeze"]
    prox = np.concatenate([np.array(r["prox"]) for r in sq])
    unc = np.concatenate([np.array(r["unc"]) for r in sq])
    adv = np.concatenate([np.array(r["correct_method"]) - np.array(r["correct_b3"])
                          for r in sq])
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    for x, a, lab in [(prox, ax[0], "distance from observed support"),
                      (unc, ax[1], "$\\hat s$ epistemic uncertainty")]:
        qs = np.quantile(x, np.linspace(0, 1, 9))
        cen, mu, se = [], [], []
        for i in range(len(qs) - 1):
            m = (x >= qs[i]) & (x < qs[i + 1])
            if m.sum() > 20:
                cen.append(0.5 * (qs[i] + qs[i + 1]))
                mu.append(adv[m].mean() * 100)
                se.append(adv[m].std() / np.sqrt(m.sum()) * 100)
        a.axhline(0, color="k", lw=0.8)
        a.errorbar(cen, mu, yerr=se, marker="o", ms=4, color="tab:red", capsize=2)
        a.set_xlabel(lab); a.set_ylabel("method $-$ B3 accuracy (pts)")
    ax[0].set_title("(a) advantage vs support distance")
    ax[1].set_title("(b) advantage vs $\\hat s$ uncertainty")
    save(fig, "fig4_squeeze")


def fig5_robust():
    e4 = load("exp4_squeeze.json")
    if not e4:
        return
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    f = e4["foreign"]
    fg = np.mean([x["foreign_acc_method"] - x["foreign_acc_b0"] for x in f]) * 100
    cg = np.mean([x["collar_acc_method"] - x["collar_acc_b0"] for x in f]) * 100
    fge = np.std([x["foreign_acc_method"] - x["foreign_acc_b0"] for x in f]) * 100
    cge = np.std([x["collar_acc_method"] - x["collar_acc_b0"] for x in f]) * 100
    ax[0].bar(["collar\n(recombinable)", "foreign\n(non-recombinable)"], [cg, fg],
              yerr=[cge, fge], color=["tab:red", "0.6"], capsize=4)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_ylabel("method gain over B0 (pts)")
    ax[0].set_title("(a) K4 scope: foreign gain $\\approx$ 0")
    rb = e4["robust"]
    etas = [n["eta"] for n in rb[0]["noise"]]
    gains = [np.mean([[n["slice_gain"] for n in r["noise"] if n["eta"] == e][0]
                      for r in rb]) * 100 for e in etas]
    ax[1].plot(etas, gains, "-o", color="tab:red", ms=4)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("noise injected into $\\hat s$")
    ax[1].set_ylabel("method slice gain (pts)")
    ax[1].set_title("(b) graceful degradation")
    save(fig, "fig5_robust")


def fig6_mnist():
    e3 = load("exp3_mnist.json")
    if not e3:
        return
    rows = e3["rows"]
    betas = sorted(set(r["beta"] for r in rows))
    iox = [np.mean([r["iox"] for r in rows if r["beta"] == b]) for b in betas]
    fig = plt.figure(figsize=(11, 3.3))
    ax0 = fig.add_subplot(1, 3, 1)
    for meth, col in [("B0_obs", "0.6"), ("B3_reweight", "tab:green"),
                      ("decoy_rotate", "tab:orange"), ("method", "tab:red")]:
        mu = [np.mean([r["acc_slice"][meth] for r in rows if r["beta"] == b]) for b in betas]
        ax0.plot(iox, mu, "-o", color=col, ms=4, label=meth)
    ax0.set_xlabel("$I(O;X)$ (bits)"); ax0.set_ylabel("thick-slice accuracy")
    ax0.set_title("(a) MNIST slice accuracy"); ax0.legend(fontsize=6)
    ax1 = fig.add_subplot(1, 3, 2)
    gap = [np.mean([r["acc_slice"]["method"] - r["acc_slice"]["decoy_rotate"]
                    for r in rows if r["beta"] == b]) * 100 for b in betas]
    ax1.axhline(0, color="k", lw=0.8)
    ax1.plot(iox, gap, "-o", color="tab:red", ms=4)
    ax1.set_xlabel("$I(O;X)$ (bits)"); ax1.set_ylabel("method $-$ decoy (pts)")
    ax1.set_title("(b) selection-targeting gap")
    grid_p = C.RESULTS / "exp3_sample_grid.npy"
    ax2 = fig.add_subplot(1, 3, 3)
    if grid_p.exists():
        g = np.load(grid_p)[:36]
        canvas = np.ones((6 * 28, 6 * 28))
        for i in range(min(36, len(g))):
            r, c = divmod(i, 6)
            canvas[r*28:(r+1)*28, c*28:(c+1)*28] = np.clip(g[i], 0, 1)
        ax2.imshow(canvas, cmap="gray_r"); ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("(c) synthesized thick-stroke digits")
    save(fig, "fig6_mnist")


def run():
    fig2_e0()
    fig1_curve()
    fig4_squeeze()
    fig5_robust()
    fig6_mnist()
    fig3_collar()          # rebuilds a stack; do last
    print("figures done")


if __name__ == "__main__":
    run()
