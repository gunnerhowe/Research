"""Build publication figures (vector PDF) for the paper from results JSONs.
Concept and validation figures regenerate their (cheap) data directly."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"
FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)

from ornstein.dbar import dbar_curve  # noqa: E402
from ornstein.surrogates import iaaft  # noqa: E402
from ornstein.systems import lorenz_trajectory  # noqa: E402

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 1.3, "figure.dpi": 150, "pdf.fonttype": 42,
})
C = {"blue": "#0173B2", "orange": "#DE8F05", "green": "#029E73",
     "red": "#D55E00", "purple": "#CC78BC", "grey": "#949494"}

# ---------------- fig_concept ----------------------------------------------------
truth = lorenz_trajectory(30_000, tau=0.1, seed=1)
speed2 = lorenz_trajectory(30_000, tau=0.1, seed=2, speed=2.0)
ia = iaaft(truth[:, 0], n_iter=200, seed=3)

fig, axes = plt.subplots(2, 3, figsize=(9.5, 4.2))
seg = slice(0, 400)
t = 0.1 * np.arange(400)
for ax, (series, name, c) in zip(axes[0], [
        (truth[:, 0], "truth", C["blue"]),
        (ia, "IAAFT surrogate", C["orange"]),
        (speed2[:, 0], r"speed$\times 2$", C["green"])]):
    ax.plot(t, series[seg], color=c, lw=0.9)
    ax.set_title(name)
    ax.set_xlabel("time")
    ax.set_ylabel("$x(t)$")
    ax.set_ylim(-22, 22)
axes[1, 0].plot(truth[:, 0], truth[:, 2], ",", color=C["blue"], alpha=0.25)
axes[1, 0].set_title("truth: attractor $(x,z)$")
axes[1, 2].plot(speed2[:, 0], speed2[:, 2], ",", color=C["green"], alpha=0.25)
axes[1, 2].set_title(r"speed$\times2$: attractor $(x,z)$ — identical")
bins = np.linspace(-22, 22, 80)
axes[1, 1].hist(truth[:, 0], bins=bins, density=True, histtype="step",
                color=C["blue"], label="truth")
axes[1, 1].hist(ia, bins=bins, density=True, histtype="step",
                color=C["orange"], ls="--", label="IAAFT")
axes[1, 1].set_title("marginal of $x$ — exactly equal")
axes[1, 1].legend()
for ax in axes[1]:
    ax.set_xlabel("$x$")
axes[1, 0].set_ylabel("$z$")
fig.tight_layout()
fig.savefig(FIGS / "fig_concept.pdf")
fig.savefig(FIGS / "fig_concept.png", dpi=140)
plt.close(fig)
print("fig_concept done")

# ---------------- fig_validation --------------------------------------------------
rng = np.random.default_rng(42)
N = 1_000_000
u = (rng.random(N) < 0.5).astype(np.int8)
alt = (np.arange(N) % 2).astype(np.int8)
a = (rng.random(N) < 0.3).astype(np.int8)
b = (rng.random(N) < 0.5).astype(np.int8)
NSV = (1, 2, 4, 8, 16, 32)
rows_alt = dbar_curve(u, alt, 2, ns=NSV, n_blocks=4000, repeats=3, seed=3)
rows_ber = dbar_curve(a, b, 2, ns=NSV, n_blocks=4000, repeats=3, seed=1)


def alt_exact(n):
    k = np.arange(n + 1)
    return float(np.sum(binom.pmf(k, n, 0.5) * np.minimum(k, n - k)) / n)


fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharex=True)
ns = [r["n"] for r in rows_alt]
axes[0].plot(ns, [alt_exact(n) for n in ns], "-", color="k", lw=1,
             label=r"exact $\bar d_n$")
axes[0].errorbar(ns, [r["dbar"] for r in rows_alt],
                 yerr=[r["dbar_std"] for r in rows_alt], fmt="o", ms=4,
                 color=C["blue"], label="estimate")
axes[0].axhline(0.5, color=C["grey"], ls=":", lw=1)
axes[0].text(1.05, 0.455, r"$\bar d = \frac{1}{2}$ (Fano-forced)", fontsize=7,
             color=C["grey"])
axes[0].set_ylim(-0.02, 0.56)
axes[0].set_title(r"iid($\frac{1}{2}$) vs. period-2 alternation")
axes[0].set_ylabel(r"$\bar d_n$")
axes[1].plot(ns, [0.2] * len(ns), "-", color="k", lw=1, label=r"exact $\bar d_n = 0.2$")
axes[1].errorbar(ns, [r["dbar"] for r in rows_ber],
                 yerr=[r["dbar_std"] for r in rows_ber], fmt="o", ms=4,
                 color=C["blue"], label="estimate")
axes[1].fill_between(ns, 0, [r["floor"] for r in rows_ber], color=C["grey"],
                     alpha=0.35, label="noise floor")
axes[1].set_title("iid Bern(0.3) vs. iid Bern(0.5)")
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xlabel("block length $n$")
    ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "fig_validation.pdf")
fig.savefig(FIGS / "fig_validation.png", dpi=140)
plt.close(fig)
print("fig_validation done")

# ---------------- fig_lorenz_curves ------------------------------------------------
exp3 = json.loads((RES / "exp3_multiseed.json").read_text())["aggregate"]
exp1 = json.loads((RES / "exp1_convergence.json").read_text())
ORDER = [("truth2", "independent truth (neg. ctrl)", C["green"]),
         ("iaaft", "IAAFT", C["orange"]),
         ("speed2", r"speed$\times$2", C["blue"]),
         ("reversed", "time-reversed", C["purple"]),
         ("rho32", r"$\rho=32$ (pos. ctrl)", C["red"])]
fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2), sharex=True)
for ax, (key, label, c) in zip(axes.flat, ORDER):
    cv = exp3["systems"][key]["dbar_curve"]
    n = [r["n"] for r in cv]
    m = np.array([r["dbar_mean"] for r in cv])
    s = np.array([r["dbar_std"] for r in cv])
    fl = np.array([r["floor_mean"] for r in cv])
    ax.fill_between(n, 0, fl, color=C["grey"], alpha=0.35, label="noise floor")
    ax.plot(n, m, "-o", ms=3.5, color=c, label=r"$\bar d_n$")
    ax.fill_between(n, m - s, m + s, color=c, alpha=0.25)
    ax.set_xscale("log", base=2)
    ax.set_title(label)
    ax.legend()
for ax in axes[1]:
    ax.set_xlabel("block length $n$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\bar d_n$")
# last panel: sample-size scaling
ax = axes.flat[-1]
for nsub, c in ((10_000, C["red"]), (100_000, C["orange"]), (1_000_000, C["blue"])):
    rows = exp1["scaling"]["iaaft"][str(nsub)]
    ax.plot([r["n"] for r in rows], [r["dbar"] for r in rows], "-o", ms=3,
            color=c, label=f"$N=10^{int(np.log10(nsub))}$")
    ax.plot([r["n"] for r in rows], [r["floor"] for r in rows], ":", color=c, lw=1)
ax.set_xscale("log", base=2)
ax.set_title("IAAFT: sample-size scaling\n(dotted: floors)")
ax.set_xlabel("block length $n$")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "fig_lorenz_curves.pdf")
fig.savefig(FIGS / "fig_lorenz_curves.png", dpi=140)
plt.close(fig)
print("fig_lorenz_curves done")

# ---------------- fig_esn -----------------------------------------------------------
exp5 = json.loads((RES / "exp5_esn.json").read_text())
exp5b = json.loads((RES / "exp5b_esn_degraded.json").read_text())
models = [m for m in exp5["models"] if "w1_state" in m]
deg = [m for m in exp5b["models"] if "w1_state" in m]
fig, ax = plt.subplots(figsize=(5.6, 3.8))
floors = ([m["w1_state_floor"] for m in models] + [m["w1_state_floor"] for m in deg])
MARK = {"linear-readout": ("s", C["red"], "linear readout (no $r^2$)"),
        "undertrained": ("^", C["orange"], "undertrained"),
        "tiny-reservoir": ("D", C["purple"], "tiny reservoir"),
        "over-regularized": ("v", C["blue"], "over-regularized")}


def draw(ax, s_main, s_deg):
    ax.axvspan(min(floors), max(floors), color=C["grey"], alpha=0.3,
               label="$W_1$ same-vs-same floor")
    ax.axhline(0, color="k", lw=0.6)
    sc = ax.scatter([m["w1_state"] for m in models],
                    [m["dbar_sep"] for m in models],
                    c=[m["rho_spec"] for m in models], cmap="viridis", s=s_main,
                    edgecolor="k", linewidths=0.4, zorder=3,
                    label=r"healthy sweep ($\rho_s$)")
    seen = set()
    for m in deg:
        mk, c, lab = MARK[m["config"]]
        ax.scatter(m["w1_state"], m["dbar_sep"], marker=mk, color=c, s=s_deg,
                   edgecolor="k", linewidths=0.4, zorder=4,
                   label=lab if lab not in seen else None)
        seen.add(lab)
    return sc


# main axes: zoom on the region that matters (subtle dissociation)
sc = draw(ax, 46, 52)
ax.set_xlim(0.09, 0.42)
ax.set_ylim(-0.015, 0.055)
ax.set_xlabel(r"$W_1$ state-space (invariant measure)")
ax.set_ylabel(r"$\bar d$ separation (floor-adjusted)")
ax.legend(fontsize=6.5, loc="upper left")
plt.colorbar(sc, ax=ax, label=r"spectral radius $\rho_s$")
# inset: full range including gross failures
axin = ax.inset_axes([0.585, 0.30, 0.38, 0.34])
draw(axin, 14, 16)
axin.set_xscale("log")
axin.set_title("full range (log $x$)", fontsize=6.5)
axin.tick_params(labelsize=6)
fig.tight_layout()
fig.savefig(FIGS / "fig_esn.pdf")
fig.savefig(FIGS / "fig_esn.png", dpi=140)
plt.close(fig)
print("fig_esn done")
