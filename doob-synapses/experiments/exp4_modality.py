"""E4 -- second modality + the noise-optimum vs task-similarity relationship.

  (1) Continual Yin-Yang (the BrainScaleS group's own procedural benchmark, 5
      rotations, shared head): the inverted-U must reproduce on a second, very
      different task stream (doob U, ou monotone).
  (2) Noise-optimum vs task similarity: the per-task rotation gap sets task
      SIMILARITY (small gap = similar, large = dissimilar / more interference).
      We map how the retention optimum sigma* and its lift depend on it -- the
      mechanism's operating regime.
"""
from __future__ import annotations

import math
import numpy as np

from common import ABLATION_SEEDS, HEADLINE_SEEDS, SIGMAS, DEVICE, save, stamp
from doobsyn.data import get_tasks
from doobsyn.sim import run_sequence
from doobsyn.stats import inverted_u_test

YY_FIXED = dict(lr_task=0.1, lr_c=0.1, epochs=4, batch_size=64,
                barrier_scale=0.2, kappa=1.0, anchor_strength=1.0,
                fisher_batches=8, hidden=30, n_layers=2)
YY_TASKS = dict(n_tasks=5, n_train=2000, n_test=1000)
MAX_ROTS = [math.pi / 4, math.pi / 2, math.pi]     # increasing dissimilarity


def sweep(method, seeds, max_rot, kappa=1.0):
    ret = []
    for s in seeds:
        tasks = get_tasks("yin_yang", seed=s, max_rot=max_rot, **YY_TASKS)
        kw = dict(YY_FIXED); kw["kappa"] = kappa
        ret.append([run_sequence("yin_yang", tasks, method=method, sigma=float(sig),
                                 seed=s, device=DEVICE, **kw)["retention"]
                    for sig in SIGMAS])
    return np.array(ret)


def main():
    out = {"config": {"sigmas": SIGMAS, "seeds_headline": HEADLINE_SEEDS,
                      "seeds_similarity": ABLATION_SEEDS, "max_rots": MAX_ROTS,
                      "yy_fixed": YY_FIXED, "yy_tasks": YY_TASKS},
           "env": stamp(), "similarity_scan": {}}

    # headline second-modality inverted-U at max_rot=pi (strong interference)
    print("== Yin-Yang (max_rot=pi) ==")
    doob = sweep("doob", HEADLINE_SEEDS, math.pi)
    ou = sweep("ou", HEADLINE_SEEDS, math.pi)
    iud = inverted_u_test(SIGMAS, doob)
    iuo = inverted_u_test(SIGMAS, ou)
    out["yin_yang"] = {
        "doob": {"retention_mean": doob.mean(0).tolist(),
                 "retention_sd": doob.std(0, ddof=1).tolist(),
                 "retention_by_seed": doob.tolist(), "inverted_u": iud},
        "ou": {"retention_mean": ou.mean(0).tolist(),
               "retention_sd": ou.std(0, ddof=1).tolist(), "inverted_u": iuo},
    }
    print(f"doob U={iud['inverted_u']} sig*={iud['sigma_star']:.3f} lift={iud['lift_over_zero']:+.3f}")
    print(f"ou   U={iuo['inverted_u']} peak@sig={iuo['sigma_star']:.3f}")

    # similarity scan
    print("\n== noise-optimum vs task similarity ==")
    for mr in MAX_ROTS:
        ret = sweep("doob", ABLATION_SEEDS, mr)
        iu = inverted_u_test(SIGMAS, ret)
        out["similarity_scan"][f"{mr:.4f}"] = {
            "max_rot_deg": math.degrees(mr),
            "retention_mean": ret.mean(0).tolist(),
            "retention_sd": ret.std(0, ddof=1).tolist(),
            "inverted_u": iu,
        }
        print(f"max_rot={math.degrees(mr):5.0f}deg U={iu['inverted_u']} "
              f"sig*={iu['sigma_star']:.3f} lift={iu['lift_over_zero']:+.3f}")

    out["e4_verdict"] = {
        "second_modality_inverted_u": iud["inverted_u"],
        "verdict": ("inverted-U reproduces on continual Yin-Yang" if iud["inverted_u"]
                    else "inverted-U does NOT reproduce on Yin-Yang (single-substrate risk)"),
    }
    save("exp4_modality.json", out)


if __name__ == "__main__":
    main()
