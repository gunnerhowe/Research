"""Coarse pre-gate calibration of the ONE fixed operating point (gamma and the
proximity factor), BEFORE any gated seed. Disclosed in PLAN.md Deviations, in
the spirit of fixing a training recipe / box size before a gated run. The gate
QUANTILE rules are pre-registered; here we only pick gamma and dmax_factor by
the E1 objective: high complement-hit-rate for the real selector, low
off-manifold rejection, and a clear method>decoy/B2 separation.

Run:  python experiments/calibrate.py
"""
from __future__ import annotations

import numpy as np

import common as C
from selamp import data
from selamp.bridge import generate_labeled


def complement_metrics(stack, X, y):
    """On-manifold-and-censored hit rate + off-manifold rejection for a
    synthesized set."""
    s_true = data.selection_prob(X, stack.beta, stack.testbed)
    val = stack.val.score(X)
    on_manifold = ~val["reject_mask"]
    censored = s_true < 0.5
    return {
        "complement_hit": float((on_manifold & censored).mean()),
        "reject_rate": val["reject_rate"],
        "mean_s_true": float(s_true.mean()),
        "in_slice": float((data.phi_std(X, stack.testbed) > 0.5).mean()),
    }


def run():
    beta, seed = 4.0, 0
    print("Calibration @ beta=4 seed=0 (two_moons)\n" + "=" * 60)
    for gamma in (3.0, 6.0, 10.0):
        for dmax_factor in (4.0, 6.0, 8.0):
            op = dict(C.OP, gamma=gamma, dmax_factor=dmax_factor)
            stack = C.Stack("two_moons", beta, seed, op=op)
            Xm, ym, _ = generate_labeled(stack.dm, stack.sel, stack.gate,
                                         C.N_SYNTH_PER_CLASS, stack.cfg, seed=seed)
            Xd, yd, _ = generate_labeled(stack.dm, stack.sel, stack.gate,
                                         C.N_SYNTH_PER_CLASS, stack.cfg,
                                         decoy="rotate", seed=seed)
            mm = complement_metrics(stack, Xm, ym)
            md = complement_metrics(stack, Xd, yd)
            print(f"gamma={gamma:4.1f} dfac={dmax_factor:.0f} | "
                  f"method hit={mm['complement_hit']:.3f} rej={mm['reject_rate']:.3f} "
                  f"s={mm['mean_s_true']:.3f} | "
                  f"decoy hit={md['complement_hit']:.3f} rej={md['reject_rate']:.3f} "
                  f"s={md['mean_s_true']:.3f} | gap={mm['complement_hit']-md['complement_hit']:+.3f}")


if __name__ == "__main__":
    run()
