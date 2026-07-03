"""Experiment 2 — THE DECISIVE EXPERIMENT (spec: "this IS the paper").

Full matrix: surrogates x metrics. Win condition: d̄ is large exactly where
Wasserstein-on-invariant-measure (and the spectrum/ACF baselines) are small,
for surrogates constructed to be measure-matched but dynamically wrong.

Also: the spec's risk #1 (partition dependence) — repeat the d̄ ranking under three
different partitions and two sampling timescales; conclusions must not flip.
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
from ornstein.baselines import (acf_distance, delay_embed, psd_log_distance,
                                w1_marginal, w1_state)
from ornstein.dbar import dbar_curve
from ornstein.entropy import block_entropy_curve, rigorous_gap_lb
from ornstein.symbolize import (apply_edges, box_symbols_xz, quantile_edges,
                                sign_symbols)

NS = (1, 2, 4, 8, 16, 32)
N_BLOCKS = 4000
N = 1_000_000
t0 = time.time()
print(f"Generating datasets (N={N}) ...")
data = make_datasets(n_samples=N)
truth_x = data["truth"]["x"]
truth_X = data["truth"]["X"]
x_scale = float(np.std(truth_x))
print(f"  done in {time.time()-t0:.1f}s")


def dbar_summary(rows):
    """Peak floor-adjusted separation over n, plus where it happens."""
    peak = max(rows, key=lambda r: r["dbar"] - r["floor"])
    return {"dbar": peak["dbar"], "floor": peak["floor"], "std": peak["dbar_std"],
            "n": peak["n"], "sep": peak["dbar"] - peak["floor"], "rows": rows}


# --- main matrix (sign(x) partition, tau=0.1) -----------------------------------
out = {"N": N, "matrix": {}, "partition_sensitivity": {}, "tau_sensitivity": {}}
sym_truth = sign_symbols(truth_x)[0]
curve_truth = block_entropy_curve(sym_truth, 2)

print("\n=== metric x surrogate matrix (sign(x), tau=0.1) ===")
for name in SURROGATE_ORDER:
    t1 = time.time()
    d = data[name]
    row = {}
    row["w1_marginal_x"] = w1_marginal(truth_x, d["x"]) / x_scale
    emb_t = delay_embed(truth_x, dim=3, lag=1)
    emb_s = delay_embed(d["x"], dim=3, lag=1)
    row["w1_delay"], row["w1_delay_floor"] = w1_state(emb_t, emb_s, n_sub=2000,
                                                      repeats=3, seed=11)
    if d["X"] is not None:
        row["w1_state3d"], row["w1_state3d_floor"] = w1_state(truth_X, d["X"],
                                                              n_sub=2000, repeats=3,
                                                              seed=12)
    else:
        row["w1_state3d"], row["w1_state3d_floor"] = None, None
    row["psd_logdist_db"] = psd_log_distance(truth_x, d["x"])
    row["acf_rmse"] = acf_distance(truth_x, d["x"], max_lag=200)

    sym_s = sign_symbols(d["x"])[0]
    ds = dbar_summary(dbar_curve(sym_truth, sym_s, 2, ns=NS, n_blocks=N_BLOCKS,
                                 repeats=3, seed=300))
    row["dbar"] = ds["dbar"]; row["dbar_floor"] = ds["floor"]
    row["dbar_sep"] = ds["sep"]; row["dbar_n"] = ds["n"]; row["dbar_std"] = ds["std"]
    row["dbar_rows"] = ds["rows"]
    lb, n_lb, gap = rigorous_gap_lb(curve_truth, block_entropy_curve(sym_s, 2), 2)
    row["fano_lb"] = lb; row["block_gap_bits"] = gap
    out["matrix"][name] = row
    w3 = f"{row['w1_state3d']:.4f}" if row["w1_state3d"] is not None else "  n/a "
    print(f"{name:9s} W1x={row['w1_marginal_x']:.4f} W3d={w3} "
          f"Wdel={row['w1_delay']:.4f}(f{row['w1_delay_floor']:.4f}) "
          f"PSD={row['psd_logdist_db']:.2f}dB ACF={row['acf_rmse']:.4f} | "
          f"d̄={row['dbar']:.4f}(f{row['dbar_floor']:.4f},n={row['dbar_n']}) "
          f"FanoLB={row['fano_lb']:.4f}  [{time.time()-t1:.0f}s]")

# --- partition sensitivity (spec risk #1) ---------------------------------------
print("\n=== partition sensitivity (tau=0.1) ===")
q4_edges = quantile_edges(truth_x, 4)
xz_xedges = np.array([0.0])
xz_zedges = np.quantile(truth_X[:, 2], [1 / 3, 2 / 3])

partitions = {
    "sign(x) m=2": lambda d: sign_symbols(d["x"]),
    "quantile4(x) m=4": lambda d: apply_edges(d["x"], q4_edges),
    "box(x,z) m=6": lambda d: (box_symbols_xz(d["X"], xz_xedges, xz_zedges)
                               if d["X"] is not None else (None, None)),
}
SENS_NAMES = ("truth2", "iaaft", "speed2")  # conclusion-critical subset
for pname, fn in partitions.items():
    st, m = fn(data["truth"])
    res = {}
    for name in SENS_NAMES:
        ss, _ = fn(data[name])
        if ss is None:
            continue
        ns_ok = tuple(n for n in NS if n * np.log2(m) <= 52)
        ds = dbar_summary(dbar_curve(st, ss, m, ns=ns_ok, n_blocks=N_BLOCKS,
                                     repeats=2, seed=400))
        res[name] = {k: ds[k] for k in ("dbar", "floor", "sep", "n", "std")}
    out["partition_sensitivity"][pname] = res
    print(f"  {pname:18s} " + "  ".join(
        f"{k}:sep={v['sep']:.4f}" for k, v in res.items()))

# --- sampling-timescale sensitivity ----------------------------------------------
print("\n=== tau sensitivity (sign(x)) ===")
for tau in (0.05, 0.25):
    dat = make_datasets(n_samples=N, tau=tau)
    st = sign_symbols(dat["truth"]["x"])[0]
    res = {}
    for name in SENS_NAMES:
        ss = sign_symbols(dat[name]["x"])[0]
        ds = dbar_summary(dbar_curve(st, ss, 2, ns=NS, n_blocks=N_BLOCKS,
                                     repeats=2, seed=500))
        res[name] = {k: ds[k] for k in ("dbar", "floor", "sep", "n", "std")}
    out["tau_sensitivity"][str(tau)] = res
    print(f"  tau={tau}: " + "  ".join(f"{k}:sep={v['sep']:.4f}" for k, v in res.items()))

with open(RESULTS / "exp2_decisive.json", "w") as f:
    json.dump(out, f, indent=2)

# --- headline figure --------------------------------------------------------------
metrics = [("w1_state3d", "W1 invariant measure (3D state)"),
           ("w1_delay", "W1 delay-embedded measure"),
           ("psd_logdist_db", "power-spectrum dist (dB)"),
           ("acf_rmse", "ACF RMSE"),
           ("dbar_sep", "d̄ (floor-adjusted)"),
           ("fano_lb", "Fano LB on d̄ (entropy gap)")]
fig, axes = plt.subplots(1, len(metrics), figsize=(17, 3.6))
names = SURROGATE_ORDER
for ax, (key, title) in zip(axes, metrics):
    vals = [out["matrix"][n][key] if out["matrix"][n][key] is not None else np.nan
            for n in names]
    colors = ["C2" if n == "truth2" else "C3" if n == "rho32" else "C0" for n in names]
    ax.bar(range(len(names)), vals, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_title(title, fontsize=8)
fig.suptitle("Decisive experiment: static metrics (left) vs dynamical metrics (right). "
             "Green = negative control, red = positive control.", fontsize=10)
fig.tight_layout()
fig.savefig(RESULTS / "exp2_matrix.png", dpi=130)
print(f"\nwrote results/exp2_decisive.json + exp2_matrix.png ({time.time()-t0:.1f}s total)")
