"""Experiment 5 — real learned surrogates: ESN family over spectral radius.

Question for the paper: among trained surrogates, does d̄ resolve dynamical-fidelity
differences that the invariant-measure Wasserstein cannot? Each model gets:
one-step NRMSE (trajectory-level quality), W1 state-space (measure-level quality),
d̄ + entropy (process-level quality). Divergent/collapsed rollouts are reported as such.
"""
import json
import time

import numpy as np

from exp_common import RESULTS

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ornstein.baselines import psd_log_distance, w1_state
from ornstein.dbar import dbar_curve
from ornstein.entropy import block_entropy_curve, entropy_rate, rigorous_gap_lb
from ornstein.esn import ESN
from ornstein.symbolize import sign_symbols
from ornstein.systems import lorenz_trajectory

TAU = 0.1
N_TRAIN = 100_000
N_ROLL = 300_000
N_REF = 300_000
NS = (1, 2, 4, 8, 16, 32)
N_BLOCKS = 2000
RHOS = (0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4)
SEEDS = (0, 1)

t0 = time.time()
train = lorenz_trajectory(N_TRAIN + 1, tau=TAU, seed=7000)
ref = lorenz_trajectory(N_REF, tau=TAU, seed=8000)  # independent evaluation reference
test = lorenz_trajectory(20_001, tau=TAU, seed=9000)  # held-out one-step test
mu, sd = train.mean(axis=0), train.std(axis=0)
U_train = (train - mu) / sd
U_test = (test - mu) / sd

sym_ref = sign_symbols(ref[:, 0])[0]
curve_ref = block_entropy_curve(sym_ref, 2)
h_ref, _, _ = entropy_rate(sym_ref, 2)
print(f"data ready ({time.time()-t0:.0f}s); truth h = {h_ref:.4f} bits/sym", flush=True)

models = []
for rho in RHOS:
    for seed in SEEDS:
        t1 = time.time()
        esn = ESN(3, n_res=400, rho_spec=rho, seed=seed).fit(U_train)
        # held-out one-step NRMSE (teacher forced)
        r = np.zeros(esn.n_res)
        errs = []
        for t in range(len(U_test) - 1):
            r = np.tanh(esn.W @ r + esn.W_in @ U_test[t] + esn.b)
            if t >= 1000:
                errs.append(esn._features(r) @ esn.W_out - U_test[t + 1])
        nrmse = float(np.sqrt(np.mean(np.square(errs))))
        roll_n, status = esn.rollout(N_ROLL)
        roll = roll_n * sd + mu  # de-normalize
        row = {"rho_spec": rho, "seed": seed, "one_step_nrmse": nrmse,
               "status": status, "rollout_len": len(roll)}
        if len(roll) >= 50_000:
            row["w1_state"], row["w1_state_floor"] = w1_state(
                ref, roll, n_sub=2000, repeats=2, seed=seed)
            row["psd_logdist_db"] = psd_log_distance(ref[:, 0], roll[:, 0])
            sym_s = sign_symbols(roll[:, 0])[0]
            h_s, _, _ = entropy_rate(sym_s, 2)
            row["h_block"] = float(h_s)
            lb, _, _ = rigorous_gap_lb(curve_ref, block_entropy_curve(sym_s, 2), 2)
            row["fano_lb"] = lb
            rows = dbar_curve(sym_ref, sym_s, 2, ns=NS, n_blocks=N_BLOCKS,
                              repeats=1, seed=seed)
            row["dbar_rows"] = rows
            peak = max(rows, key=lambda r_: r_["dbar"] - r_["floor"])
            row["dbar"], row["dbar_floor"] = peak["dbar"], peak["floor"]
            row["dbar_sep"], row["dbar_n"] = peak["dbar"] - peak["floor"], peak["n"]
        models.append(row)
        msg = (f"rho={rho:4.2f} seed={seed}  nrmse={nrmse:.2e}  status={status:9s}")
        if "w1_state" in row:
            msg += (f"  W1={row['w1_state']:.4f}(f{row['w1_state_floor']:.4f})"
                    f"  d̄sep={row['dbar_sep']:.4f}  h={row['h_block']:.4f}")
        print(msg + f"  [{time.time()-t1:.0f}s]", flush=True)

with open(RESULTS / "exp5_esn.json", "w") as f:
    json.dump({"tau": TAU, "n_train": N_TRAIN, "n_roll": N_ROLL,
               "h_truth": float(h_ref), "models": models}, f, indent=2)
print(f"wrote results/exp5_esn.json ({time.time()-t0:.0f}s)")
