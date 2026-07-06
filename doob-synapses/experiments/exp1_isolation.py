"""E1 -- mechanism isolation. Show the effect is the BARRIER CONDITIONING, not
just any noise.

  (1) kappa scan: interpolate the Doob-steering strength kappa from 0 (= plain OU,
      unconditioned) to 1 (full h-transform). The inverted-U must EMERGE as kappa
      grows and be ABSENT at kappa=0. (Ablating the conditioning flattens the
      curve -- the pre-registered E1/K3 test.)
  (2) barrier scan: vary barrier_scale (tighter <-> looser memory-critical
      barrier). The retention optimum sigma* / lift must TRACK the barrier, not sit
      at a fixed noise level -- evidence the optimum is set by the conditioning
      geometry, not a generic noise sweet-spot.
"""
from __future__ import annotations

import numpy as np

from common import (PRIMARY, SIGMAS, ABLATION_SEEDS, FIXED, TASKS_KW, DEVICE,
                    save, stamp)
from doobsyn.data import get_tasks
from doobsyn.sim import run_sequence
from doobsyn.stats import inverted_u_test

KAPPAS = [0.0, 0.25, 0.5, 1.0]
BARRIERS = [0.1, 0.2, 0.4]


def sweep(seeds, *, kappa, barrier_scale):
    ret = []
    for s in seeds:
        tasks = get_tasks(PRIMARY, seed=s, **TASKS_KW)
        kw = dict(FIXED); kw["kappa"] = kappa; kw["barrier_scale"] = barrier_scale
        ret.append([run_sequence(PRIMARY, tasks, method="doob", sigma=float(sig),
                                 seed=s, device=DEVICE, **kw)["retention"]
                    for sig in SIGMAS])
    return np.array(ret)


def main():
    out = {"config": {"sigmas": SIGMAS, "seeds": ABLATION_SEEDS,
                      "kappas": KAPPAS, "barriers": BARRIERS}, "env": stamp(),
           "kappa_scan": {}, "barrier_scan": {}}

    print("== kappa scan (barrier_scale fixed at 0.2) ==")
    for k in KAPPAS:
        ret = sweep(ABLATION_SEEDS, kappa=k, barrier_scale=FIXED["barrier_scale"])
        iu = inverted_u_test(SIGMAS, ret)
        out["kappa_scan"][f"{k}"] = {
            "retention_mean": ret.mean(0).tolist(),
            "retention_sd": ret.std(0, ddof=1).tolist(),
            "inverted_u": iu,
        }
        print(f"kappa={k:.2f} U={iu['inverted_u']} sig*={iu['sigma_star']:.3f} "
              f"lift={iu['lift_over_zero']:+.3f} p0={iu['p_peak_gt_zero']:.3f}")

    print("\n== barrier scan (kappa=1) ==")
    for b in BARRIERS:
        ret = sweep(ABLATION_SEEDS, kappa=1.0, barrier_scale=b)
        iu = inverted_u_test(SIGMAS, ret)
        out["barrier_scan"][f"{b}"] = {
            "retention_mean": ret.mean(0).tolist(),
            "retention_sd": ret.std(0, ddof=1).tolist(),
            "inverted_u": iu,
        }
        print(f"b={b:.2f} U={iu['inverted_u']} sig*={iu['sigma_star']:.3f} "
              f"lift={iu['lift_over_zero']:+.3f} peakret={iu['ret_at_peak']:.3f}")

    # isolation verdict: kappa=0 must NOT be an inverted-U; kappa=1 must be.
    k0 = out["kappa_scan"]["0.0"]["inverted_u"]["inverted_u"]
    k1 = out["kappa_scan"]["1.0"]["inverted_u"]["inverted_u"]
    lifts = [out["kappa_scan"][f"{k}"]["inverted_u"]["lift_over_zero"] for k in KAPPAS]
    out["isolation"] = {
        "kappa0_inverted_u": k0, "kappa1_inverted_u": k1,
        "lift_monotone_in_kappa": bool(np.all(np.diff(lifts) >= -0.02)),
        "lifts_by_kappa": dict(zip(map(str, KAPPAS), lifts)),
        "pass": bool((not k0) and k1),
        "verdict": ("PASS -- ablating the conditioning (kappa->0) removes the "
                    "inverted-U; it is the barrier-conditioning, not generic noise")
                   if ((not k0) and k1) else
                   "FAIL -- K3 (conditioning inert; noise helps generically)",
    }
    print(f"\nE1 isolation: {'PASS' if out['isolation']['pass'] else 'FAIL -- K3'}")
    save("exp1_isolation.json", out)


if __name__ == "__main__":
    main()
