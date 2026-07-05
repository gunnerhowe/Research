"""Calibrate realistic learning-curve observation noise from LCBench
residuals (feeds the B-E0 gate's 'realistic noise' condition).

Two estimators per curve:
- diff:  robust sd of lag-1 differences / sqrt(2) over the last 40 epochs
         (captures epoch-to-epoch observation noise, robust to the smooth
         trend; MAD-based).
- pow3:  residual sd around a per-curve pow3 fit (upper bound; includes
         model misfit).

Writes data/lcbench_calib.json with the medians and per-dataset values.

Run: python experiments/expB_calib.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT

from heckesel.lc import fit_pow3_ls, curve_values

DATASETS = ["Fashion-MNIST", "adult", "higgs", "jasmine", "vehicle",
            "volkert"]
N_POW3 = 300  # configs per dataset for the pow3-residual estimator


def main():
    z = np.load(ROOT / "data" / "lcbench_cache.npz")
    out = {"datasets": {}}
    diff_all, pow3_all = [], []
    rng = np.random.default_rng(0)
    for ds in DATASETS:
        m = z[ds]                      # (n, 52)
        tail = m[:, 12:]               # last 40 epochs (trend flattens)
        d = np.diff(tail, axis=1)
        mad = np.median(np.abs(d - np.median(d, axis=1, keepdims=True)),
                        axis=1)
        sig_diff = 1.4826 * mad / np.sqrt(2.0)

        idx = rng.choice(len(m), N_POW3, replace=False)
        t = np.arange(1, m.shape[1] + 1, dtype=float)
        sig_pow3 = []
        for i in idx:
            phi = fit_pow3_ls(t, m[i])
            resid = m[i] - curve_values(phi[None], t)[0]
            sig_pow3.append(float(np.std(resid[4:])))
        out["datasets"][ds] = {
            "sigma_diff_median": float(np.median(sig_diff)),
            "sigma_diff_q90": float(np.quantile(sig_diff, 0.9)),
            "sigma_pow3_median": float(np.median(sig_pow3)),
        }
        diff_all.extend(sig_diff.tolist())
        pow3_all.extend(sig_pow3)
        print(ds, out["datasets"][ds])

    out["sigma_median"] = float(np.median(diff_all))
    out["sigma_q90"] = float(np.quantile(diff_all, 0.9))
    out["sigma_pow3_median"] = float(np.median(pow3_all))
    (ROOT / "data" / "lcbench_calib.json").write_text(json.dumps(out,
                                                                 indent=1))
    print("\nsigma_median (diff) =", out["sigma_median"])
    print("sigma_q90    (diff) =", out["sigma_q90"])
    print("sigma_median (pow3) =", out["sigma_pow3_median"])


if __name__ == "__main__":
    main()
