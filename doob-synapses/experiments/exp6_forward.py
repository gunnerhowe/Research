"""E6 -- the hardware-faithful FORWARD-noise realization.

On analog silicon the intrinsic noise is in the multiply-accumulate (the forward
pass), not injected on the weights. We show:
  (a) the Doob mechanism SURVIVES this realization -- doob is a retention inverted-U,
      the matched OU control is flat -- once the importance is clamped;
  (b) the clamp is load-bearing: without it the anchored drift blows up and BOTH
      methods collapse to chance (this was the on-hardware-port failure mode);
  (c) the Doob-steering coupling tunes the retention optimum to LOW, device-
      reachable noise (~few % CV), so the mechanism is portable to a given chip's
      intrinsic-noise level.
"""
from __future__ import annotations

import numpy as np

from common import (PRIMARY, HEADLINE_SEEDS, ABLATION_SEEDS, TASKS_KW, DEVICE,
                    save, stamp)
from doobsyn.data import get_tasks
from doobsyn.hwloop import run_forward_sequence
from doobsyn.stats import inverted_u_test

SIGMAS = [0.0, 0.1, 0.25, 0.4, 0.6, 0.9, 1.4, 2.0]      # activation-noise amplitudes
LOW_SIGMAS = [0.0, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35]   # device-reachable band
COUPLINGS = [1.0, 4.0, 10.0]
FIXED = dict(lr=0.1, lr_c=0.1, epochs=2, bs=128, hidden=100, barrier=0.2)


def sweep(sigmas, method, seeds, *, imp_clip=10.0, sig_doob_k=1.0):
    ret = []
    for sd in seeds:
        tasks = get_tasks(PRIMARY, seed=sd, **TASKS_KW)
        ret.append([run_forward_sequence(PRIMARY, tasks, method=method, sigma=float(s),
                                         seed=sd, realization="forward", device=DEVICE,
                                         imp_clip=imp_clip, sig_doob_k=sig_doob_k, **FIXED)[0]
                    for s in sigmas])
    return np.array(ret)


def main():
    out = {"config": {"sigmas": SIGMAS, "low_sigmas": LOW_SIGMAS,
                      "couplings": COUPLINGS, "seeds": HEADLINE_SEEDS, "fixed": FIXED},
           "env": stamp()}

    # (a) headline: forward-noise inverted-U, doob vs ou, 8 seeds
    print("== forward-noise realization (8 seeds) ==")
    res = {}
    for meth in ("doob", "ou"):
        r = sweep(SIGMAS, meth, HEADLINE_SEEDS)
        iu = inverted_u_test(SIGMAS, r)
        res[meth] = {"retention_mean": r.mean(0).tolist(),
                     "retention_sd": r.std(0, ddof=1).tolist(),
                     "retention_by_seed": r.tolist(), "inverted_u": iu}
        print(f"{meth:5s} " + " ".join(f"{v:.3f}" for v in r.mean(0)) +
              f"  U={iu['inverted_u']} sig*={iu['sigma_star']:.2f} lift={iu['lift_over_zero']:+.3f} "
              f"p0={iu['p_peak_gt_zero']:.3f}")
    out["forward"] = res

    # (b) the clamp is the fix: NOCLAMP collapses
    print("\n== clamp ablation (5 seeds) ==")
    clamp = {}
    for tag, ic in (("clamp", 10.0), ("noclamp", 1e12)):
        r = sweep(SIGMAS, "doob", ABLATION_SEEDS, imp_clip=ic)
        clamp[tag] = {"retention_mean": r.mean(0).tolist(),
                      "collapsed": bool(r.std() < 0.02)}
        print(f"{tag:8s} " + " ".join(f"{v:.3f}" for v in r.mean(0)))
    out["clamp_ablation"] = clamp

    # (c) coupling tunes the optimum to device-reachable noise
    print("\n== coupling scan (low-noise band, 5 seeds) ==")
    cs = {}
    for k in COUPLINGS:
        r = sweep(LOW_SIGMAS, "doob", ABLATION_SEEDS, sig_doob_k=k)
        mean = r.mean(0)
        i = int(np.argmax(mean))
        cs[f"{k}"] = {"retention_mean": mean.tolist(),
                      "sigma_star": float(LOW_SIGMAS[i]),
                      "lift": float(mean[i] - mean[0])}
        print(f"k={k:4.1f} " + " ".join(f"{v:.3f}" for v in mean) +
              f"  sig*={LOW_SIGMAS[i]:.3f} lift={mean[i]-mean[0]:+.3f}")
    out["coupling_scan"] = cs

    out["e6_verdict"] = {
        "forward_doob_lift": res["doob"]["inverted_u"]["lift_over_zero"],
        "forward_ou_lift": res["ou"]["inverted_u"]["lift_over_zero"],
        "noclamp_collapses": clamp["noclamp"]["collapsed"],
        "min_reachable_sigma_star": min(cs[f"{k}"]["sigma_star"] for k in COUPLINGS),
    }
    save("exp6_forward.json", out)


if __name__ == "__main__":
    main()
