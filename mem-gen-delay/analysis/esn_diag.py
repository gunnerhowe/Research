"""POST-HOC DIAGNOSTIC (not a sealed rung; disclosed as such): what does the LOSSONLY
reservoir read — slow loss shape, or fast oscillations?

Re-scores the already-scored cells with the frozen LOSSONLY artifacts (esn_frozen.json,
untouched) after filtering the train_loss channel per run:
  SMOOTH k: centered moving average of width k evals  -> keeps slow shape, removes fast
  RESID  k: frozen-mean + (loss - SMOOTH k)           -> keeps fast, removes slow shape
If SMOOTH preserves leads -> the signal is slow curvature (learning-curve-shape family).
If RESID preserves leads  -> the signal is oscillation micro-structure (the
Notsawo/slingshot family). Output: analysis/out6/esn_diag_smooth.json
"""
import copy
import json

import numpy as np

from esn_anchor import (CELLS, OUT, corpus, reservoir_weights, score_cellset)

K_LIST = [5, 15]


def moving_avg(v, k):
    pad = k // 2
    vp = np.pad(v, (pad, pad), mode="edge")
    ker = np.ones(k) / k
    return np.convolve(vp, ker, mode="valid")[: len(v)]


def filtered(runs, mode, k, mu_loss):
    out = []
    for r in runs:
        r2 = dict(r)
        r2["ch"] = dict(r["ch"])
        v = r["ch"]["train_loss"]
        sm = moving_avg(v, k)
        r2["ch"]["train_loss"] = sm if mode == "SMOOTH" else mu_loss + (v - sm)
        out.append(r2)
    return out


def main():
    frozen = json.load(open(f"{OUT}/esn_frozen.json"))
    f = frozen["LOSSONLY"]
    W, W_in = reservoir_weights(len(f["chans"]))
    w = np.array(f["readout"], np.float32)
    cells_runs = {
        "gateB_grid6r5": corpus("grid6r5"), "gateC_grid6r7": corpus("grid6r7"),
        "gap_grid6r8": corpus("grid6r8"), "law_grid6r9": corpus("grid6r9"),
        "TRAP_grid6r6": corpus("grid6r6"),
    }
    res = {}
    for mode in ("RAW", "SMOOTH", "RESID"):
        for k in ([0] if mode == "RAW" else K_LIST):
            key = mode if mode == "RAW" else f"{mode}{k}"
            res[key] = {}
            for name, runs in cells_runs.items():
                rr = runs if mode == "RAW" else filtered(runs, mode, k,
                                                         f["mu"]["train_loss"])
                sc = score_cellset(rr, f["chans"], f["mu"], f["sd"], W, W_in, w,
                                   f["tau"])
                res[key][name] = {kk: vv for kk, vv in sc.items() if kk != "per_run"}
                print(f"{key:9s} {name:16s} lead={sc['median_lead']:>7.0f} "
                      f"miss={sc['miss']:>5s} FA={sc['fa']:>5s} c={sc['median_c']}")
    json.dump(res, open(f"{OUT}/esn_diag_smooth.json", "w"), indent=1)
    print("wrote analysis/out6/esn_diag_smooth.json (POST-HOC DIAGNOSTIC)")


if __name__ == "__main__":
    main()
