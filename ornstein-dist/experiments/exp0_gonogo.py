"""Experiment 0 — the spec's one-hour go/no-go: entropy-rate pre-check.

d̄ ≥ g⁻¹(|Δh|) (Fano, RESEARCH_NOTES §2). If adversarial surrogates already show a clear
symbolic entropy-rate gap vs truth, d̄ is provably bounded below — the metric sees
something before we even run the OT estimator. GO/NO-GO for the rest of the project.
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
from ornstein.entropy import (entropy_rate, fano_lower_bound, lz78_entropy,
                              rigorous_gap_lb)
from ornstein.symbolize import sign_symbols

t0 = time.time()
N = 1_000_000
TAU = 0.1
print(f"Generating datasets (N={N}, tau={TAU}) ...")
data = make_datasets(n_samples=N, tau=TAU)
print(f"  done in {time.time()-t0:.1f}s")

out = {"N": N, "tau": TAU, "systems": {}}
curves = {}
for name in ["truth"] + SURROGATE_ORDER:
    sym, m = sign_symbols(data[name]["x"])
    h_block, n_used, curve = entropy_rate(sym, m, n_max=24)
    h_lz = lz78_entropy(sym, m)
    out["systems"][name] = {
        "p1": float(sym.mean()), "h_block": float(h_block), "h_block_n": n_used,
        "h_lz78": float(h_lz),
    }
    curves[name] = curve
    print(f"  {name:9s}  P(s=1)={sym.mean():.4f}  h_block={h_block:.4f} bits (n={n_used})"
          f"  h_lz78={h_lz:.4f}")

h_truth = out["systems"]["truth"]["h_block"]
print(f"\n{'system':9s} {'|dh| bits':>10s} {'LB(h_n gap)':>12s} {'LB rigorous':>12s}   note")
for name in SURROGATE_ORDER:
    dh = abs(out["systems"][name]["h_block"] - h_truth)
    lb = fano_lower_bound(dh, 2)
    # rigorous finite-n version: |H_n(X)-H_n(Y)|/n <= g(dbar), per trusted n, best n
    lb_rig, n_rig, gap_rig = rigorous_gap_lb(curves["truth"], curves[name], 2)
    out["systems"][name].update({"dh": dh, "fano_lb": lb, "fano_lb_rigorous": lb_rig,
                                 "fano_lb_rigorous_n": n_rig,
                                 "block_gap_bits_per_symbol": gap_rig})
    print(f"{name:9s} {dh:10.4f} {lb:12.4f} {lb_rig:12.4f}   {LABELS[name]}")

# decision: GO if at least one Wasserstein-matched adversarial surrogate has a clear
# entropy gap (LB > resolution) while the negative control shows ~none.
neg = out["systems"]["truth2"]["fano_lb_rigorous"]
adv = {k: out["systems"][k]["fano_lb_rigorous"] for k in ("iaaft", "speed2")}
go = any(v > max(5 * neg, 0.005) for v in adv.values())
out["decision"] = "GO" if go else "NO-GO"
print(f"\nnegative-control LB = {neg:.5f}; adversarial LBs = "
      f"{ {k: round(v, 5) for k, v in adv.items()} }")
print(f"DECISION: {out['decision']}")

with open(RESULTS / "exp0_gonogo.json", "w") as f:
    json.dump(out, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name in ["truth"] + SURROGATE_ORDER:
    c = curves[name]
    ok = c["support"] <= 0.02 * c["n_windows"]
    axes[0].plot(c["n"], c["h_cond"], label=name, alpha=0.8)
    axes[0].plot(c["n"][ok], c["h_cond"][ok], lw=2.5)
    axes[1].semilogy(c["n"], c["support"] / c["n_windows"], label=name, alpha=0.8)
axes[0].set_xlabel("block length n"); axes[0].set_ylabel("h_n = H_n - H_{n-1} (bits)")
axes[0].set_title("conditional block entropy (thick = trusted)"); axes[0].legend(fontsize=7)
axes[1].axhline(0.02, color="k", ls=":"); axes[1].set_xlabel("n")
axes[1].set_ylabel("observed support / windows"); axes[1].set_title("undersampling guard")
fig.tight_layout()
fig.savefig(RESULTS / "exp0_entropy_curves.png", dpi=130)
print(f"\nwrote results/exp0_gonogo.json, results/exp0_entropy_curves.png "
      f"({time.time()-t0:.1f}s total)")
