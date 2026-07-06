"""E5 -- fold the ON-SILICON noise measurement into the model and re-test.

Reads results/bss2_silicon_noise.json (measured on real BrainScaleS-2 silicon,
chip hxcube7fpga3chip61_1, via EBRAINS), fits (i) the additive/multiplicative
split of the intrinsic MAC noise and (ii) the num_sends averaging exponent, then
re-runs the retention sweep with a MEASURED-informed BSS-2 noise model to confirm
the inverted-U survives the real device's noise STRUCTURE (additive, white).

This upgrades E2 from assumed-parameter emulation to measured-noise-calibrated.
It is NOT on-chip training (still future work); it grounds the noise model in the
chip. Kill condition K2's first half -- is the intrinsic noise the right
structure/scale? -- is answered here with silicon data.
"""
from __future__ import annotations

import json
import numpy as np

from common import (PRIMARY, SIGMAS, HEADLINE_SEEDS, FIXED, TASKS_KW, DEVICE,
                    RESULTS, save, stamp)
from doobsyn.bss2 import measured_bss2_params
from doobsyn.data import get_tasks
from doobsyn.sim import run_sequence
from doobsyn.stats import inverted_u_test


def fit_silicon(sil):
    s = np.array(sil["noise_vs_signal"]["signal"], float)
    ts = np.array(sil["noise_vs_signal"]["trial_std"], float)
    b, a = np.polyfit(s, ts, 1)                      # trial_std = a + b*signal
    sref = float(np.median(s))
    mult_frac = float(b * sref / (a + b * sref))     # multiplicative fraction at median signal
    ns = np.array(sil["noise_vs_num_sends"]["num_sends"], float)
    cv = np.array(sil["noise_vs_num_sends"]["cv"], float)
    p = float(-np.polyfit(np.log(ns), np.log(cv), 1)[0])   # CV ~ num_sends^-p
    cvsig = np.array(sil["noise_vs_signal"]["cv"], float)
    return dict(additive=float(a), mult_slope=float(b),
                mult_frac_at_median=mult_frac,
                additive_frac_at_median=float(1 - mult_frac),
                cv_min=float(cvsig.min()), cv_max=float(cvsig.max()),
                num_sends_exponent=p, signal_ref=sref,
                trial_std_mean=float(ts.mean()))


def main():
    sil = json.load(open(RESULTS / "bss2_silicon_noise.json"))
    fit = fit_silicon(sil)
    print("silicon fit:", {k: round(v, 4) for k, v in fit.items()})

    params = measured_bss2_params()
    ret = []
    for s in HEADLINE_SEEDS:
        tasks = get_tasks(PRIMARY, seed=s, **TASKS_KW)
        ret.append([run_sequence(PRIMARY, tasks, method="doob", sigma=float(sig),
                                 seed=s, device=DEVICE, bss2_noise=params,
                                 quantize=True, **FIXED)["retention"]
                    for sig in SIGMAS])
    ret = np.array(ret)
    iu = inverted_u_test(SIGMAS, ret)
    print(f"measured-noise E2: U={iu['inverted_u']} sig*={iu['sigma_star']:.3f} "
          f"lift={iu['lift_over_zero']:+.3f}")

    out = {"provenance": sil["provenance"], "chip": sil["chip"],
           "silicon_fit": fit,
           "measured_noise_params": {"color": params.color,
              "multiplicative": params.multiplicative,
              "fixed_pattern": params.fixed_pattern, "weight_bits": params.weight_bits},
           "measured_noise_sweep": {"sigmas": SIGMAS,
              "retention_mean": ret.mean(0).tolist(),
              "retention_sd": ret.std(0, ddof=1).tolist(),
              "inverted_u": iu},
           "env": stamp()}
    save("exp5_silicon.json", out)


if __name__ == "__main__":
    main()
