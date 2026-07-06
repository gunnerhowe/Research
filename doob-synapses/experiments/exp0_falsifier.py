"""E0 -- GATE F, the afternoon falsifier.

Sweep the intrinsic-noise amplitude sigma and measure retention (mean past-task
accuracy) for the barrier-conditioned rule (doob) and the matched anchored-drift
controls (ou, ewc, mesu, none), 8 seeds, at the fixed operating point.

GATE F PASS iff doob is an inverted-U (stats.inverted_u_test) AND none of the
controls is. FAIL => K1 (mechanism reduces to OUA/MESU/EWC; project dead).
"""
from __future__ import annotations

import numpy as np

from common import (PRIMARY, SIGMAS, HEADLINE_SEEDS, FIXED, TASKS_KW, EWC_ANCHOR,
                    DEVICE, save, stamp)
from doobsyn.data import get_tasks
from doobsyn.sim import run_sequence
from doobsyn.stats import inverted_u_test, monotone_decreasing_frac, curvature_sign

METHODS = ["doob", "ou", "ewc", "mesu", "none"]


def run_method(method, seeds):
    """retention/avg/plasticity arrays shaped (n_seed, n_sigma) + wall."""
    ret, avg, pla, forg, wall = [], [], [], [], 0.0
    for s in seeds:
        tasks = get_tasks(PRIMARY, seed=s, **TASKS_KW)
        rr, aa, pp, ff = [], [], [], []
        for sig in SIGMAS:
            kw = dict(FIXED)
            if method == "ewc":
                kw["anchor_strength"] = EWC_ANCHOR
            r = run_sequence(PRIMARY, tasks, method=method, sigma=float(sig),
                             seed=s, device=DEVICE, **kw)
            rr.append(r["retention"]); aa.append(r["avg_acc"])
            pp.append(r["plasticity"]); ff.append(r["forgetting"]); wall += r["wall_s"]
        ret.append(rr); avg.append(aa); pla.append(pp); forg.append(ff)
    return (np.array(ret), np.array(avg), np.array(pla), np.array(forg), wall)


def main():
    out = {"config": {"testbed": PRIMARY, "sigmas": SIGMAS,
                      "seeds": HEADLINE_SEEDS, "fixed": FIXED,
                      "ewc_anchor": EWC_ANCHOR}, "env": stamp(), "methods": {}}
    for method in METHODS:
        ret, avg, pla, forg, wall = run_method(method, HEADLINE_SEEDS)
        iu = inverted_u_test(SIGMAS, ret)
        out["methods"][method] = {
            "retention_mean": ret.mean(0).tolist(),
            "retention_sd": ret.std(0, ddof=1).tolist(),
            "retention_by_seed": ret.tolist(),
            "avg_acc_mean": avg.mean(0).tolist(),
            "plasticity_mean": pla.mean(0).tolist(),
            "forgetting_mean": forg.mean(0).tolist(),
            "inverted_u": iu,
            "monotone_dec_frac": monotone_decreasing_frac(SIGMAS, ret),
            "curvature": curvature_sign(SIGMAS, ret),
            "wall_s": wall,
        }
        v = iu["inverted_u"]
        print(f"{method:5s} U={v} sig*={iu['sigma_star']:.3f} "
              f"lift={iu['lift_over_zero']:+.3f} p0={iu['p_peak_gt_zero']:.3f} "
              f"phi={iu['p_peak_gt_hi']:.3f} monodec={out['methods'][method]['monotone_dec_frac']:.2f}")

    # ---- GATE F verdict --------------------------------------------------------
    doob_u = out["methods"]["doob"]["inverted_u"]["inverted_u"]
    controls_u = [out["methods"][m]["inverted_u"]["inverted_u"]
                  for m in ("ou", "ewc", "mesu", "none")]
    gate_f = bool(doob_u and not any(controls_u))
    out["gate_f"] = {
        "pass": gate_f,
        "doob_inverted_u": doob_u,
        "controls_inverted_u": dict(zip(("ou", "ewc", "mesu", "none"), controls_u)),
        "verdict": ("PASS -- GO (barrier-conditioning produces a retention "
                    "inverted-U the matched anchored-drift controls do not)")
                   if gate_f else "FAIL -- K1 (mechanism reduces to OUA/MESU/EWC)",
    }
    print(f"\nGATE F: {'PASS -- GO' if gate_f else 'FAIL -- K1'}")
    save("exp0_falsifier.json", out)
    return gate_f


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
