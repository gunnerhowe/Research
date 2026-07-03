"""Experiment 4 — Kuramoto-Sivashinsky (L=22): does the metric generalize beyond
Lorenz to spatiotemporal chaos?

SYMMETRY REDUCTION: KS on a periodic domain has O(2) symmetry and its spatial phase
diffuses slowly. Comparing raw states (or a pointwise observable u(x0,t)) confounds
the comparison with phase position — independent truth runs then differ as much as
anything else (verified: results/exp4_ks_xobs_nonreduced.json, where the same-process
floors correctly flagged the drift). We therefore work in translation-invariant
coordinates: state = Fourier magnitudes (|u_hat_1|..|u_hat_10|), scalar observable
= |u_hat_1|(t), median partition from truth.

Systems per seed: truth, independent truth (neg ctrl), IAAFT of the observable built
from the independent run, and speed×2 (time-rescaled PDE — exactly the same invariant
measure).
"""
import json
import time

import numpy as np

from exp_common import RESULTS

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ornstein.baselines import acf_distance, psd_log_distance, w1_marginal, w1_state
from ornstein.dbar import dbar_curve
from ornstein.entropy import block_entropy_curve, entropy_rate, rigorous_gap_lb
from ornstein.surrogates import iaaft
from ornstein.symbolize import apply_edges

N_SAMP = 200_000
DT_SAMPLE = 1.0
NS = (1, 2, 4, 8, 16, 32)
N_BLOCKS = 2000
N_SEEDS = 4
SURR = ["truth2", "iaaft", "speed2"]

def reduce_sym(U):
    """Translation-invariant coordinates: leading Fourier magnitudes."""
    return np.abs(np.fft.rfft(U, axis=1))[:, 1:11]


t0 = time.time()
all_seeds = []
for s in range(N_SEEDS):
    t1 = time.time()
    from ornstein.ks import ks_trajectory
    truth = reduce_sym(ks_trajectory(N_SAMP, dt_sample=DT_SAMPLE, dt=0.25,
                                     seed=1000 + s))
    truth2 = reduce_sym(ks_trajectory(N_SAMP, dt_sample=DT_SAMPLE, dt=0.25,
                                      seed=2000 + s))
    speed2 = reduce_sym(ks_trajectory(N_SAMP, dt_sample=DT_SAMPLE, dt=0.125,
                                      speed=2.0, seed=3000 + s))
    print(f"seed {s}: integration {time.time()-t1:.0f}s", flush=True)
    obs_t = truth[:, 0]
    data = {
        "truth2": {"X": truth2, "x": truth2[:, 0]},
        "iaaft": {"X": None, "x": iaaft(truth2[:, 0], n_iter=200, seed=5000 + s)},
        "speed2": {"X": speed2, "x": speed2[:, 0]},
    }
    med = np.median(obs_t)
    x_scale = float(np.std(obs_t))
    sym_t = apply_edges(obs_t, np.array([med]))[0]
    curve_t = block_entropy_curve(sym_t, 2)
    h_t, _, _ = entropy_rate(sym_t, 2)
    seed_res = {"seed": s, "h_truth": float(h_t), "systems": {}}
    for name in SURR:
        d = data[name]
        row = {"w1_marginal": w1_marginal(obs_t, d["x"]) / x_scale}
        if d["X"] is not None:
            row["w1_state"], row["w1_state_floor"] = w1_state(
                truth, d["X"], n_sub=2000, repeats=2, seed=10 * s)
        else:
            row["w1_state"] = row["w1_state_floor"] = None
        row["psd_logdist_db"] = psd_log_distance(obs_t, d["x"])
        row["acf_rmse"] = acf_distance(obs_t, d["x"], max_lag=200)
        sym_s = apply_edges(d["x"], np.array([med]))[0]
        h_s, _, _ = entropy_rate(sym_s, 2)
        row["h_block"] = float(h_s)
        lb, n_lb, _ = rigorous_gap_lb(curve_t, block_entropy_curve(sym_s, 2), 2)
        row["fano_lb"] = lb
        rows = dbar_curve(sym_t, sym_s, 2, ns=NS, n_blocks=N_BLOCKS, repeats=1,
                          seed=100 * s)
        row["dbar_rows"] = rows
        peak = max(rows, key=lambda r: r["dbar"] - r["floor"])
        row["dbar"], row["dbar_floor"], row["dbar_n"] = (peak["dbar"], peak["floor"],
                                                         peak["n"])
        row["dbar_sep"] = peak["dbar"] - peak["floor"]
        seed_res["systems"][name] = row
    all_seeds.append(seed_res)
    print(f"seed {s}: total {time.time()-t1:.0f}s  " + "  ".join(
        f"{k}:d̄sep={seed_res['systems'][k]['dbar_sep']:.4f}" for k in SURR), flush=True)

agg = {"N": N_SAMP, "dt_sample": DT_SAMPLE, "n_seeds": N_SEEDS, "ns": list(NS),
       "h_truth": [float(np.mean([r["h_truth"] for r in all_seeds])),
                   float(np.std([r["h_truth"] for r in all_seeds]))],
       "systems": {}}
SCALARS = ["w1_marginal", "w1_state", "w1_state_floor", "psd_logdist_db", "acf_rmse",
           "h_block", "fano_lb", "dbar", "dbar_floor", "dbar_sep"]
for name in SURR:
    entry = {}
    for key in SCALARS:
        vals = [r["systems"][name][key] for r in all_seeds
                if r["systems"][name][key] is not None]
        entry[key] = ([float(np.mean(vals)), float(np.std(vals))] if vals
                      else [None, None])
    curve_stats = []
    for i, n in enumerate(NS):
        dv = [r["systems"][name]["dbar_rows"][i]["dbar"] for r in all_seeds]
        fv = [r["systems"][name]["dbar_rows"][i]["floor"] for r in all_seeds]
        curve_stats.append({"n": n, "dbar_mean": float(np.mean(dv)),
                            "dbar_std": float(np.std(dv)),
                            "floor_mean": float(np.mean(fv)),
                            "regime": all_seeds[0]["systems"][name]["dbar_rows"][i]["regime"]})
    entry["dbar_curve"] = curve_stats
    agg["systems"][name] = entry

with open(RESULTS / "exp4_ks.json", "w") as f:
    json.dump({"aggregate": agg, "per_seed": all_seeds}, f, indent=2)

print(f"\n=== KS aggregate (mean ± std over {N_SEEDS} seeds) ===")
print(f"truth h = {agg['h_truth'][0]:.4f}±{agg['h_truth'][1]:.4f} bits/symbol")
for name in SURR:
    e = agg["systems"][name]
    ws = e["w1_state"][0]
    print(f"{name:7s} d̄sep={e['dbar_sep'][0]:.4f}±{e['dbar_sep'][1]:.4f}  "
          f"Wstate={ws if ws is not None else float('nan'):.4f}  "
          f"W1m={e['w1_marginal'][0]:.4f}  PSD={e['psd_logdist_db'][0]:.2f}dB  "
          f"h={e['h_block'][0]:.4f}  FanoLB={e['fano_lb'][0]:.4f}")
print(f"wrote results/exp4_ks.json ({time.time()-t0:.0f}s)")
