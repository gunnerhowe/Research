"""Experiment 3 — multi-seed Lorenz metric matrix (the paper's main table).

8 independent realizations. Per seed: independent truth run, independently constructed
surrogates, full metric battery. Error bars = across-seed std. The IAAFT/reversed
surrogates are built from a SECOND independent truth run so surrogate and reference share
no sample noise.
"""
import json
import time

import numpy as np

from exp_common import RESULTS

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ornstein.baselines import (acf_distance, delay_embed, psd_log_distance,
                                rosenstein_lambda1, w1_marginal, w1_state)
from ornstein.dbar import dbar_curve
from ornstein.entropy import block_entropy_curve, entropy_rate, rigorous_gap_lb
from ornstein.surrogates import iaaft, time_reverse
from ornstein.symbolize import sign_symbols
from ornstein.systems import lorenz_trajectory

N = 500_000
TAU = 0.1
NS = (1, 2, 4, 8, 16, 32)
N_BLOCKS = 2000
N_SEEDS = 8
SURR = ["truth2", "iaaft", "speed2", "reversed", "rho32"]

t0 = time.time()
all_seeds = []
for s in range(N_SEEDS):
    t1 = time.time()
    truth = lorenz_trajectory(N, tau=TAU, seed=1000 + s)
    truth2 = lorenz_trajectory(N, tau=TAU, seed=2000 + s)
    speed2 = lorenz_trajectory(N, tau=TAU, seed=3000 + s, speed=2.0)
    rho32 = lorenz_trajectory(N, tau=TAU, seed=4000 + s, rho=32.0)
    data = {
        "truth2": {"X": truth2, "x": truth2[:, 0]},
        "iaaft": {"X": None, "x": iaaft(truth2[:, 0], n_iter=200, seed=5000 + s)},
        "speed2": {"X": speed2, "x": speed2[:, 0]},
        "reversed": {"X": time_reverse(truth2), "x": time_reverse(truth2)[:, 0]},
        "rho32": {"X": rho32, "x": rho32[:, 0]},
    }
    truth_x = truth[:, 0]
    x_scale = float(np.std(truth_x))
    sym_t = sign_symbols(truth_x)[0]
    curve_t = block_entropy_curve(sym_t, 2)
    h_t, n_used_t, _ = entropy_rate(sym_t, 2)
    lam_t, r2_t = rosenstein_lambda1(truth_x, dt=TAU, seed=s)

    seed_res = {"seed": s, "h_truth": float(h_t),
                "lambda1_truth": lam_t, "lambda1_truth_r2": r2_t, "systems": {}}
    for name in SURR:
        d = data[name]
        row = {}
        row["w1_marginal_x"] = w1_marginal(truth_x, d["x"]) / x_scale
        if d["X"] is not None:
            row["w1_state3d"], row["w1_state3d_floor"] = w1_state(
                truth, d["X"], n_sub=2000, repeats=2, seed=10 * s)
        else:
            row["w1_state3d"] = row["w1_state3d_floor"] = None
        row["w1_delay"], row["w1_delay_floor"] = w1_state(
            delay_embed(truth_x, 3, 1), delay_embed(d["x"], 3, 1),
            n_sub=2000, repeats=2, seed=10 * s + 1)
        row["psd_logdist_db"] = psd_log_distance(truth_x, d["x"])
        row["acf_rmse"] = acf_distance(truth_x, d["x"], max_lag=200)
        lam, r2 = rosenstein_lambda1(d["x"], dt=TAU, seed=s)
        row["lambda1"], row["lambda1_r2"] = lam, r2

        sym_s = sign_symbols(d["x"])[0]
        h_s, _, _ = entropy_rate(sym_s, 2)
        row["h_block"] = float(h_s)
        lb, n_lb, gap = rigorous_gap_lb(curve_t, block_entropy_curve(sym_s, 2), 2)
        row["fano_lb"], row["fano_lb_n"] = lb, n_lb
        rows = dbar_curve(sym_t, sym_s, 2, ns=NS, n_blocks=N_BLOCKS,
                          repeats=1, seed=100 * s)
        row["dbar_rows"] = rows
        peak = max(rows, key=lambda r: r["dbar"] - r["floor"])
        row["dbar"], row["dbar_floor"], row["dbar_n"] = (peak["dbar"], peak["floor"],
                                                         peak["n"])
        row["dbar_sep"] = peak["dbar"] - peak["floor"]
        seed_res["systems"][name] = row
    all_seeds.append(seed_res)
    print(f"seed {s}: {time.time()-t1:.0f}s  " + "  ".join(
        f"{k}:d̄sep={seed_res['systems'][k]['dbar_sep']:.4f}" for k in SURR), flush=True)

# aggregate mean/std across seeds
agg = {"N": N, "tau": TAU, "n_seeds": N_SEEDS, "ns": list(NS), "n_blocks": N_BLOCKS,
       "h_truth": [float(np.mean([r["h_truth"] for r in all_seeds])),
                   float(np.std([r["h_truth"] for r in all_seeds]))],
       "lambda1_truth": [float(np.mean([r["lambda1_truth"] for r in all_seeds])),
                         float(np.std([r["lambda1_truth"] for r in all_seeds]))],
       "systems": {}}
SCALARS = ["w1_marginal_x", "w1_state3d", "w1_state3d_floor", "w1_delay",
           "w1_delay_floor", "psd_logdist_db", "acf_rmse", "lambda1", "lambda1_r2",
           "h_block", "fano_lb", "dbar", "dbar_floor", "dbar_sep"]
for name in SURR:
    entry = {}
    for key in SCALARS:
        vals = [r["systems"][name][key] for r in all_seeds
                if r["systems"][name][key] is not None]
        entry[key] = ([float(np.mean(vals)), float(np.std(vals))] if vals
                      else [None, None])
    # per-n dbar curve stats
    curve_stats = []
    for i, n in enumerate(NS):
        dv = [r["systems"][name]["dbar_rows"][i]["dbar"] for r in all_seeds]
        fv = [r["systems"][name]["dbar_rows"][i]["floor"] for r in all_seeds]
        curve_stats.append({"n": n, "dbar_mean": float(np.mean(dv)),
                            "dbar_std": float(np.std(dv)),
                            "floor_mean": float(np.mean(fv)),
                            "floor_std": float(np.std(fv)),
                            "regime": all_seeds[0]["systems"][name]["dbar_rows"][i]["regime"]})
    entry["dbar_curve"] = curve_stats
    agg["systems"][name] = entry

with open(RESULTS / "exp3_multiseed.json", "w") as f:
    json.dump({"aggregate": agg, "per_seed": all_seeds}, f, indent=2)

print(f"\n=== aggregate (mean ± std over {N_SEEDS} seeds) ===")
for name in SURR:
    e = agg["systems"][name]
    print(f"{name:9s} d̄sep={e['dbar_sep'][0]:.4f}±{e['dbar_sep'][1]:.4f}  "
          f"W3d={e['w1_state3d'][0] if e['w1_state3d'][0] is not None else float('nan'):.4f}  "
          f"Wdel={e['w1_delay'][0]:.4f}  PSD={e['psd_logdist_db'][0]:.2f}dB  "
          f"λ1={e['lambda1'][0]:.3f}(r2={e['lambda1_r2'][0]:.2f})  "
          f"FanoLB={e['fano_lb'][0]:.4f}")
print(f"truth: h={agg['h_truth'][0]:.4f}±{agg['h_truth'][1]:.4f} bits, "
      f"λ1={agg['lambda1_truth'][0]:.3f}±{agg['lambda1_truth'][1]:.3f} nats/t "
      f"(lit: 0.906)")
print(f"wrote results/exp3_multiseed.json ({time.time()-t0:.0f}s)")
