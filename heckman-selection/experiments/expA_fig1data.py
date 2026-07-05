"""Illustration data for Paper A Figure 1 (committed JSON; house rule 1).

One controlled draw at rho=0.9, alpha=1: the data cloud (selected vs
unselected), f0, and fitted predictive bands for a selection-blind deep
ensemble vs the Heckman ensemble on a dense x grid.

Run: python experiments/expA_fig1data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_json

from heckesel.synth import make_selection_data
from heckesel.deep import HeckmanEnsemble
from heckesel.uq import DeepEnsembleUQ

RHO, ALPHA, SIGMA, SEED = 0.9, 1.0, 0.5, 0


def main():
    data = make_selection_data(2000, RHO, alpha=ALPHA, sigma=SIGMA, d=1,
                               seed=SEED)
    sel = data.s > 0.5
    grid = np.linspace(-3, 3, 241)[:, None]
    w_all = np.column_stack([data.x, data.z])

    ens = DeepEnsembleUQ.fit(data.x[sel], data.y[sel], k=5, seed=SEED,
                             epochs=1500)
    mu_e, var_e = ens.predict(grid)
    hk = HeckmanEnsemble.fit(data.x, w_all, data.y, data.s, k=5, seed=SEED,
                             epochs=1500)
    mu_h, var_h = hk.predict(grid)

    from heckesel.synth import f0_smooth
    save_json("expA_fig1_illustration.json", {
        "rho": RHO, "alpha": ALPHA, "sigma": SIGMA, "seed": SEED,
        "x_sel": data.x[sel, 0].tolist(), "y_sel": data.y[sel].tolist(),
        "x_uns": data.x[~sel, 0].tolist(),
        "y_uns": data.y_full[~sel].tolist(),
        "grid": grid[:, 0].tolist(),
        "f0": f0_smooth(grid).tolist(),
        "prop_x_grid": __import__("scipy.special", fromlist=["ndtr"]).ndtr(
            (-1.8 * grid[:, 0] / 3.0) / np.sqrt(1 + ALPHA**2)).tolist(),
        "ens_mu": mu_e.tolist(), "ens_sd": np.sqrt(var_e).tolist(),
        "heck_mu": mu_h.tolist(), "heck_sd": np.sqrt(var_h).tolist(),
        "heck_rho_hat": hk.rho,
    })


if __name__ == "__main__":
    main()
