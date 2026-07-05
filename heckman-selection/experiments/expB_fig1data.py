"""Illustration data for Paper B Figure 1 (committed JSON).

One draw at sigma=0.02: 60 observed curves, SH rungs, survivor set at rung
0, and the naive pow3 extrapolations of a few survivors vs their true
continuations (regression-to-the-mean survivor bias made visible).

Run: python experiments/expB_fig1data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_json

from heckesel.lc import (POP_MU, default_pop_cov, curve_values, run_sh,
                         fit_pow3_ls, phi_final)

T, RUNGS, ETA = 52, [4, 12, 36], 3.0
SIGMA, SEED, N = 0.02, 3, 60


def main():
    rng = np.random.default_rng(SEED)
    L = np.linalg.cholesky(default_pop_cov())
    phi = POP_MU + rng.standard_normal((N, 3)) @ L.T
    t_full = np.arange(1, T + 1, dtype=float)
    y_true = curve_values(phi, t_full)
    y_obs = y_true + SIGMA * rng.standard_normal(y_true.shape)
    sh = run_sh(y_obs, RUNGS, ETA)
    surv0 = np.where(sh.alive[:, 1])[0]

    t4 = np.arange(1, 5, dtype=float)
    fits = {}
    for i in surv0[:8]:
        ph = fit_pow3_ls(t4, y_obs[i, :4])
        fits[int(i)] = {
            "extrap": curve_values(ph[None], t_full)[0].tolist(),
            "pred_final": float(phi_final(ph, T)[0]),
            "true_final": float(y_true[i, -1]),
        }
    save_json("expB_fig1_illustration.json", {
        "sigma": SIGMA, "seed": SEED, "rungs": RUNGS, "eta": ETA,
        "y_obs": y_obs.tolist(), "y_true": y_true.tolist(),
        "alive": sh.alive.tolist(), "thresholds": sh.thresholds,
        "survivor_fits": fits,
    })


if __name__ == "__main__":
    main()
