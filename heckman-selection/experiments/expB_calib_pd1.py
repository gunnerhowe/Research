"""Calibrate PD1 per-task observation noise (pow3-residual median), same
method as expB_calib.py for LCBench. Writes data/pd1_calib.json.

Run: python experiments/expB_calib_pd1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT

from heckesel.lc import fit_pow3_ls, curve_values

N_POW3 = 200


def main():
    z = np.load(ROOT / "data" / "pd1_cache.npz")
    out = {"datasets": {}}
    rng = np.random.default_rng(0)
    for key in z.files:
        m = z[key]
        idx = rng.choice(len(m), min(N_POW3, len(m)), replace=False)
        T = m.shape[1]
        t = np.arange(1, T + 1, dtype=float)
        sig = []
        for i in idx:
            phi = fit_pow3_ls(t, m[i])
            resid = m[i] - curve_values(phi[None], t)[0]
            k0 = max(2, T // 10)
            sig.append(float(np.std(resid[k0:])))
        out["datasets"][key] = {"sigma_pow3_median": float(np.median(sig)),
                                "T": int(T), "n": int(len(m))}
        print(key, out["datasets"][key])
    (ROOT / "data" / "pd1_calib.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
