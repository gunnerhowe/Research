"""E3 (post-hoc, reviewer-proofing; PLAN.md Addendum) — is DSA's behavior on this
testbed an artifact of the one pre-registered configuration?

For the four pairs the paper's argument rests on (recoded null, pruned twin, noise
twin, different-task control), recompute DSA under five configurations spanning the
reasonable design space: the pre-registered config, a deeper/higher-rank embedding, no
delay embedding, the reduced-rank-regression variant, and the package-default
Wasserstein score. Scores are compared WITHIN a config only (does the config rank
different-task above its own null?). Models and evaluation states are identical to
E0 (cached checkpoints, same eval seeds).
"""
import time

import numpy as np

from exp_common import (SEEDS, M, TASKS, get_model, get_states, save_json)
from dbar_diff.baselines import dsa_distance

CONFIGS = {
    "pre8r32":  dict(n_delays=8, rank=32, score_method="angular"),
    "deep16r64": dict(n_delays=16, rank=64, score_method="angular"),
    "flat1r32": dict(n_delays=1, rank=32, score_method="angular"),
    "rrr8r32":  dict(n_delays=8, rank=32, score_method="angular",
                     reduced_rank_reg=True),
    "wass8r32": dict(n_delays=8, rank=32, score_method="wasserstein"),
}
ITERS = 800

t0 = time.time()
gm = TASKS["gm"]()
rows = []
for s in SEEDS:
    A, _ = get_model("base", "gm", s)
    stA = get_states(A, gm, 0.0)
    sig = 0.2   # sigma* from E0 calibration (constant across seeds)

    twins = {
        "null": (M.recode_permute(A, seed=s + 400), 0.0),
        "noise": (A, sig),
        "prune": (get_model("prune", "gm", s, frac=0.5)[0], 0.0),
        "difftask": (get_model("feven-base", "feven", s)[0], 0.0),
    }
    for pair, (mB, sigB) in twins.items():
        stB = get_states(mB, gm, sigB, seed=555 + s)
        for name, cfg in CONFIGS.items():
            d = dsa_distance(stA, stB, device=M.DEVICE, iters=ITERS, **cfg)
            rows.append({"seed": s, "pair": pair, "config": name, "dsa": d})
            print(f"[s{s}] {pair:9s} {name:10s} DSA={d:.4f} ({time.time()-t0:.0f}s)")

# per-config verdict: does the config separate different-task from its own null?
verdicts = {}
for name in CONFIGS:
    null = np.array([r["dsa"] for r in rows
                     if r["config"] == name and r["pair"] == "null"])
    diff = np.array([r["dsa"] for r in rows
                     if r["config"] == name and r["pair"] == "difftask"])
    noise = np.array([r["dsa"] for r in rows
                      if r["config"] == name and r["pair"] == "noise"])
    prune = np.array([r["dsa"] for r in rows
                      if r["config"] == name and r["pair"] == "prune"])
    thresh = max(null.mean() + 2 * null.std(), null.max())
    verdicts[name] = {
        "null_mean": float(null.mean()), "null_std": float(null.std()),
        "null_max": float(null.max()),
        "difftask_mean": float(diff.mean()), "difftask_std": float(diff.std()),
        "difftask_min": float(diff.min()),
        "noise_mean": float(noise.mean()), "prune_mean": float(prune.mean()),
        "difftask_above_null": bool(diff.mean() > thresh),
        "n_difftask_seeds_above_null_max": int((diff > null.max()).sum()),
        "noise_over_difftask": float(noise.mean() / max(diff.mean(), 1e-12)),
    }
    print(f"{name:10s} null={null.mean():.3f}±{null.std():.3f} "
          f"difftask={diff.mean():.3f} noise={noise.mean():.3f} "
          f"prune={prune.mean():.3f} separates_difftask="
          f"{verdicts[name]['difftask_above_null']}")

save_json("exp3_dsa_robustness.json",
          {"configs": {k: {kk: str(vv) for kk, vv in v.items()}
                       for k, v in CONFIGS.items()},
           "iters": ITERS, "rows": rows, "verdicts": verdicts})
print(f"done ({time.time()-t0:.0f}s)")
