"""Experiment 5b — deliberately degraded ESN surrogates ("trained but dynamically
wrong" specimens): linear-only readout (breaks the r^2 symmetry device), undertrained,
tiny reservoir, over-regularized. Same metric battery as exp5.
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
NS = (1, 2, 4, 8, 16, 32)
N_BLOCKS = 2000


class LinearESN(ESN):
    """Readout on r only (no r^2): the standard symmetry-breaking device removed."""

    def _features(self, r):
        return r

    def fit(self, U, washout=1000):
        T, d = U.shape
        r = np.zeros(self.n_res)
        feats = np.empty((T - 1 - washout, self.n_res))
        targets = U[washout + 1:]
        for t in range(T - 1):
            r = np.tanh(self.W @ r + self.W_in @ U[t] + self.b)
            if t >= washout:
                feats[t - washout] = r
        A = feats.T @ feats + self.ridge * np.eye(self.n_res)
        self.W_out = np.linalg.solve(A, feats.T @ targets)
        self.r_end = r
        self.u_end = U[-1]
        pred = feats @ self.W_out
        self.train_nrmse = float(np.sqrt(np.mean((pred - targets) ** 2)))
        return self


CONFIGS = [
    ("linear-readout", dict(cls=LinearESN, n_res=400, rho_spec=0.9, ridge=1e-6,
                            n_train=N_TRAIN)),
    ("undertrained", dict(cls=ESN, n_res=400, rho_spec=0.9, ridge=1e-6, n_train=2000)),
    ("tiny-reservoir", dict(cls=ESN, n_res=50, rho_spec=0.9, ridge=1e-6,
                            n_train=N_TRAIN)),
    ("over-regularized", dict(cls=ESN, n_res=400, rho_spec=0.9, ridge=1e-1,
                              n_train=N_TRAIN)),
]

t0 = time.time()
train = lorenz_trajectory(N_TRAIN + 1, tau=TAU, seed=7000)
ref = lorenz_trajectory(300_000, tau=TAU, seed=8000)
test = lorenz_trajectory(20_001, tau=TAU, seed=9000)
mu, sd = train.mean(axis=0), train.std(axis=0)
U_train_full = (train - mu) / sd
U_test = (test - mu) / sd
sym_ref = sign_symbols(ref[:, 0])[0]
curve_ref = block_entropy_curve(sym_ref, 2)
h_ref, _, _ = entropy_rate(sym_ref, 2)

models = []
for label, cfg in CONFIGS:
    for seed in (0, 1):
        t1 = time.time()
        esn = cfg["cls"](3, n_res=cfg["n_res"], rho_spec=cfg["rho_spec"],
                         ridge=cfg["ridge"], seed=seed)
        esn.fit(U_train_full[: cfg["n_train"] + 1])
        r = np.zeros(esn.n_res)
        errs = []
        for t in range(len(U_test) - 1):
            r = np.tanh(esn.W @ r + esn.W_in @ U_test[t] + esn.b)
            if t >= 1000:
                errs.append(esn._features(r) @ esn.W_out - U_test[t + 1])
        nrmse = float(np.sqrt(np.mean(np.square(errs))))
        roll_n, status = esn.rollout(N_ROLL)
        roll = roll_n * sd + mu
        row = {"config": label, "seed": seed, "one_step_nrmse": nrmse,
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
        msg = f"{label:16s} seed={seed}  nrmse={nrmse:.2e}  status={status:9s}"
        if "w1_state" in row:
            msg += (f"  W1={row['w1_state']:.4f}(f{row['w1_state_floor']:.4f})"
                    f"  d̄sep={row['dbar_sep']:.4f}  h={row['h_block']:.4f}")
        print(msg + f"  [{time.time()-t1:.0f}s]", flush=True)

with open(RESULTS / "exp5b_esn_degraded.json", "w") as f:
    json.dump({"h_truth": float(h_ref), "models": models}, f, indent=2)
print(f"wrote results/exp5b_esn_degraded.json ({time.time()-t0:.0f}s)")
