"""A-E0: faithfulness numbers for the paper (the gate itself is tests/).

Fits our two-step and MLE on RandHIE (Cameron & Trivedi spec) and records
the deviations from the published seven-digit Stata reference output, plus
the Mroz87 (Greene spec) fits.

Run: python experiments/expA_e0.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_json

from heckesel.datasets import load_mroz_greene, load_randhie_ct
from heckesel.selection import heckman_mle, heckman_two_step

# Stata reference (data/mma16p3selection.txt)
STATA = {
    "mle_rho": .7355982, "mle_sigma": 1.570053, "mle_ll": -10170.11,
    "ts_lambda": .2358048, "ts_rho": 0.16833, "ts_sigma": 1.4008246,
}


def main():
    y, X, s, W, *_ = load_randhie_ct()
    ts = heckman_two_step(y, X, s, W)
    ml = heckman_mle(y, X, s, W)
    out = {
        "randhie": {
            "n": int(len(s)), "n_selected": int(s.sum()),
            "ts_rho": ts.rho, "ts_sigma": ts.sigma,
            "ts_lambda": ts.beta_lambda,
            "mle_rho": ml.rho, "mle_sigma": ml.sigma,
            "mle_ll": ml.loglik,
            "stata": STATA,
            "max_abs_dev_ts": max(abs(ts.rho - STATA["ts_rho"]),
                                  abs(ts.sigma - STATA["ts_sigma"]),
                                  abs(ts.beta_lambda - STATA["ts_lambda"])),
            "max_abs_dev_mle": max(abs(ml.rho - STATA["mle_rho"]),
                                   abs(ml.sigma - STATA["mle_sigma"])),
            "ll_dev": abs(ml.loglik - STATA["mle_ll"]),
        }
    }
    ym, Xm, sm, Wm, *_ = load_mroz_greene()
    Wn = Wm / np.max(np.abs(Wm), axis=0)
    tsm = heckman_two_step(ym, Xm, sm, Wn)
    mlm = heckman_mle(ym, Xm, sm, Wn)
    out["mroz"] = {"n": int(len(sm)), "n_selected": int(sm.sum()),
                   "ts_rho": tsm.rho, "mle_rho": mlm.rho,
                   "mle_sigma": mlm.sigma, "mle_ll": mlm.loglik}
    save_json("expA_e0.json", out)
    print(out["randhie"]["max_abs_dev_ts"], out["randhie"]["max_abs_dev_mle"])


if __name__ == "__main__":
    main()
