"""Experiment 1 — THE CRUX: is the OT-on-n-blocks d̄ estimator usable in practice?

For each (truth, surrogate) pair: d̄_n for doubling n (monotone by superadditivity),
with same-process noise floors and repeat error bars. Kill condition (spec): the curve
fails to plateau above its floor, or is drowned by estimation noise at achievable N.
Also: stability of the estimate across N = 1e4 / 1e5 / 1e6 symbols.
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp_common import LABELS, RESULTS, SURROGATE_ORDER, make_datasets

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from ornstein.dbar import dbar_curve
from ornstein.symbolize import sign_symbols

NS = (1, 2, 4, 8, 16, 32)
N_BLOCKS = 4000
N = 1_000_000

t0 = time.time()
print(f"Generating datasets (N={N}) ...")
data = make_datasets(n_samples=N)
syms = {name: sign_symbols(d["x"])[0] for name, d in data.items()}
print(f"  done in {time.time()-t0:.1f}s")

out = {"N": N, "ns": list(NS), "n_blocks": N_BLOCKS, "pairs": {}, "scaling": {}}

for name in SURROGATE_ORDER:
    t1 = time.time()
    rows = dbar_curve(syms["truth"], syms[name], 2, ns=NS, n_blocks=N_BLOCKS,
                      repeats=4, seed=100)
    out["pairs"][name] = rows
    peak = max(rows, key=lambda r: r["dbar"] - r["floor"])
    print(f"  truth vs {name:9s} ({time.time()-t1:5.1f}s)  "
          + "  ".join(f"n={r['n']}:{r['dbar']:.4f}(f{r['floor']:.4f},{r['regime'][0]})"
                      for r in rows))
    print(f"    -> best separation at n={peak['n']}: d̄={peak['dbar']:.4f} vs floor "
          f"{peak['floor']:.4f} (std {peak['dbar_std']:.4f})")

print("\nSample-size scaling (truth vs iaaft / speed2):")
for name in ("iaaft", "speed2"):
    out["scaling"][name] = {}
    for n_sub in (10_000, 100_000, 1_000_000):
        rows = dbar_curve(syms["truth"][:n_sub], syms[name][:n_sub], 2, ns=NS,
                          n_blocks=N_BLOCKS, repeats=3, seed=200)
        out["scaling"][name][str(n_sub)] = rows
        peak = max(rows, key=lambda r: r["dbar"] - r["floor"])
        print(f"  {name:7s} N={n_sub:>9,d}: peak sep n={peak['n']}: "
              f"d̄={peak['dbar']:.4f} floor={peak['floor']:.4f} std={peak['dbar_std']:.4f}")

with open(RESULTS / "exp1_convergence.json", "w") as f:
    json.dump(out, f, indent=2)

# --- plots --------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
for ax, name in zip(axes.flat, SURROGATE_ORDER):
    rows = out["pairs"][name]
    n = [r["n"] for r in rows]
    v = np.array([r["dbar"] for r in rows])
    s = np.array([r["dbar_std"] for r in rows])
    fl = np.array([r["floor"] for r in rows])
    ax.errorbar(n, v, yerr=s, marker="o", label="d̄_n estimate", color="C0")
    ax.fill_between(n, 0, fl, alpha=0.25, color="gray", label="noise floor")
    ax.set_xscale("log", base=2)
    ax.set_title(f"truth vs {name}\n{LABELS[name]}", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_xlabel("block length n")
    ax.set_ylabel("d̄_n")
axes.flat[-1].axis("off")
fig.suptitle("d̄_n convergence curves (Lorenz-63, sign(x), τ=0.1, N=1e6)", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS / "exp1_dbar_curves.png", dpi=130)

fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, name in zip(axes2, ("iaaft", "speed2")):
    for n_sub, c in zip((10_000, 100_000, 1_000_000), ("C3", "C1", "C0")):
        rows = out["scaling"][name][str(n_sub)]
        n = [r["n"] for r in rows]
        v = [r["dbar"] for r in rows]
        fl = [r["floor"] for r in rows]
        ax.plot(n, v, marker="o", color=c, label=f"N={n_sub:,}")
        ax.plot(n, fl, ls=":", color=c)
    ax.set_xscale("log", base=2)
    ax.set_title(f"truth vs {name} (dotted = floors)")
    ax.set_xlabel("block length n")
    ax.legend(fontsize=8)
axes2[0].set_ylabel("d̄_n")
fig2.suptitle("Sample-size dependence of the d̄ estimator")
fig2.tight_layout()
fig2.savefig(RESULTS / "exp1_scaling.png", dpi=130)
print(f"\nwrote results/exp1_convergence.json + 2 plots ({time.time()-t0:.1f}s total)")
