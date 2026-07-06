"""E2 -- BrainScaleS-2 intrinsic-noise EMULATION (the moat, in emulation).

We re-run the retention sweep with the diffusion noise replaced by the device-
faithful BSS-2 model (temporally COLORED + MULTIPLICATIVE + FIXED-PATTERN + 6-bit
QUANTIZED; bss2.py). Two questions, both pre-registered:

  (1) Does the inverted-U SURVIVE device-realistic noise? (moat holds in emulation)
  (2) A `color` scan: where, if anywhere, does the temporal correlation of the
      real device noise break the mechanism? This is the honest boundary and the
      concrete content of K2.

IMPORTANT (scientific integrity): this is an EMULATION on the GPU. No BrainScaleS-2
silicon or joules are measured here. The on-silicon reproduction is the pre-
registered remaining step (PLAN.md K2); bss2.Bss2Backend documents that port and
raises without the hardware stack, which is absent in this environment.
"""
from __future__ import annotations

import numpy as np

from common import (PRIMARY, SIGMAS, HEADLINE_SEEDS, ABLATION_SEEDS, FIXED,
                    TASKS_KW, DEVICE, save, stamp)
from doobsyn.bss2 import Bss2NoiseParams, Bss2Backend
from doobsyn.data import get_tasks
from doobsyn.sim import run_sequence
from doobsyn.stats import inverted_u_test

COLORS = [0.0, 0.3, 0.6, 0.9]           # AR(1) temporal color of the device noise
DEVICE_PARAMS = dict(multiplicative=0.3, fixed_pattern=0.4, weight_bits=6)


def sweep(seeds, params, quantize):
    ret = []
    for s in seeds:
        tasks = get_tasks(PRIMARY, seed=s, **TASKS_KW)
        ret.append([run_sequence(PRIMARY, tasks, method="doob", sigma=float(sig),
                                 seed=s, device=DEVICE, bss2_noise=params,
                                 quantize=quantize, **FIXED)["retention"]
                    for sig in SIGMAS])
    return np.array(ret)


def main():
    out = {"config": {"sigmas": SIGMAS, "colors": COLORS,
                      "device_params": DEVICE_PARAMS,
                      "seeds_headline": HEADLINE_SEEDS,
                      "seeds_colorscan": ABLATION_SEEDS},
           "env": stamp(),
           "hardware_available": Bss2Backend().available,
           "note": ("EMULATION only; no silicon measured. On-silicon run is the "
                    "pre-registered remaining step (K2)."),
           "color_scan": {}}

    # headline device-faithful run (color=0.5, quantized), 8 seeds
    p = Bss2NoiseParams(color=0.5, **DEVICE_PARAMS)
    ret = sweep(HEADLINE_SEEDS, p, quantize=True)
    iu = inverted_u_test(SIGMAS, ret)
    out["device_faithful"] = {
        "params": {"color": 0.5, **DEVICE_PARAMS, "quantized": True},
        "retention_mean": ret.mean(0).tolist(),
        "retention_sd": ret.std(0, ddof=1).tolist(),
        "retention_by_seed": ret.tolist(),
        "inverted_u": iu,
    }
    print(f"device-faithful (color=0.5, 6-bit) U={iu['inverted_u']} "
          f"sig*={iu['sigma_star']:.3f} lift={iu['lift_over_zero']:+.3f}")

    # color scan (5 seeds): find the boundary
    print("\n== color scan ==")
    for c in COLORS:
        pc = Bss2NoiseParams(color=c, **DEVICE_PARAMS)
        rc = sweep(ABLATION_SEEDS, pc, quantize=True)
        iuc = inverted_u_test(SIGMAS, rc)
        out["color_scan"][f"{c}"] = {
            "retention_mean": rc.mean(0).tolist(),
            "retention_sd": rc.std(0, ddof=1).tolist(),
            "inverted_u": iuc,
        }
        print(f"color={c:.1f} U={iuc['inverted_u']} sig*={iuc['sigma_star']:.3f} "
              f"lift={iuc['lift_over_zero']:+.3f}")

    surv = out["device_faithful"]["inverted_u"]["inverted_u"]
    out["e2_verdict"] = {
        "inverted_u_survives_emulation": surv,
        "verdict": ("inverted-U SURVIVES device-faithful BSS-2 noise in EMULATION; "
                    "on-silicon measurement remains (K2)") if surv else
                   "inverted-U does NOT survive the device noise model (emulation) "
                   "-- K2 risk flagged",
    }
    save("exp2_bss2.json", out)


if __name__ == "__main__":
    main()
