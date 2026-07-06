"""E3 -- matched-budget baselines.

Compare the barrier-conditioned rule at its retention optimum sigma* (read from
E0) against the incumbents on the same 5-task Split-MNIST stream, same
architecture/epochs, 8 seeds:
  unconditioned OU / OUA-limit (ou, sigma=0), MESU (mesu, sigma=0), EWC (best
  lambda, sigma=0), Benna-Fusi cascade synapse, plain reservoir replay, and naive
  SGD. Report retention (paired Wilcoxon vs ours) and the retention-vs-energy point
  (energy.py operation-count model), including the noise TAX a GPU pays and the
  BSS-2 (emulated) energy where the same noise is intrinsic/free.
"""
from __future__ import annotations

import json
import numpy as np

from common import (PRIMARY, SIGMAS, HEADLINE_SEEDS, FIXED, TASKS_KW, DEVICE,
                    RESULTS, save, stamp)
from doobsyn.data import get_tasks
from doobsyn.sim import run_sequence
from doobsyn.energy import step_energy_pj, noise_tax_ratio
from doobsyn.stats import paired_wilcoxon
from doobsyn.models import build_model, n_params

EWC_LAMBDAS = [1.0, 2.0, 4.0, 8.0, 16.0]
REPLAY_BUFFER = 250          # total stored exemplars across the stream


def run_all_seeds(seeds, **run_kw):
    ret, avg, pla, forg, wall = [], [], [], [], []
    for s in seeds:
        tasks = get_tasks(PRIMARY, seed=s, **TASKS_KW)
        r = run_sequence(PRIMARY, tasks, seed=s, device=DEVICE, **run_kw)
        ret.append(r["retention"]); avg.append(r["avg_acc"])
        pla.append(r["plasticity"]); forg.append(r["forgetting"]); wall.append(r["wall_s"])
    return dict(retention=ret, avg_acc=avg, plasticity=pla, forgetting=forg,
                wall_s=float(np.mean(wall)))


def steps_per_run():
    # 5 tasks x epochs x ceil(2000/batch) minibatches
    import math
    return 5 * FIXED["epochs"] * math.ceil(2 * TASKS_KW["n_per_class_train"] / FIXED["batch_size"])


def main():
    with open(RESULTS / "exp0_falsifier.json") as f:
        e0 = json.load(f)
    iu = e0["methods"]["doob"]["inverted_u"]
    sig_star = iu["sigma_star"]
    i_star = iu["i_peak"]
    doob_ret_star = np.array(e0["methods"]["doob"]["retention_by_seed"])[:, i_star]

    np_ = n_params(build_model(PRIMARY, hidden=FIXED["hidden"],
                               n_layers=FIXED["n_layers"]))
    nsteps = steps_per_run()

    out = {"config": {"sigma_star": sig_star, "n_params": np_, "steps_per_run": nsteps,
                      "ewc_lambdas": EWC_LAMBDAS, "replay_buffer": REPLAY_BUFFER,
                      "seeds": HEADLINE_SEEDS}, "env": stamp(), "methods": {}}

    # ours at sigma*
    out["methods"]["doob*"] = {
        "retention": doob_ret_star.tolist(),
        "retention_mean": float(doob_ret_star.mean()),
        "sigma": sig_star,
        "energy_pj_gpu": step_energy_pj("doob", np_, sig_star, substrate="gpu") * nsteps,
        "energy_pj_bss2": step_energy_pj("doob", np_, sig_star, substrate="bss2") * nsteps,
        "noise_tax_gpu": noise_tax_ratio("doob", np_, sig_star),
    }

    # baselines at sigma=0 (their best per E0 monotonicity), except EWC lambda scan
    plan = {
        "ou": dict(method="ou", sigma=0.0),
        "mesu": dict(method="mesu", sigma=0.0),
        "none": dict(method="none", sigma=0.0),
        "benna_fusi": dict(method="none", sigma=0.0, benna_fusi=True),
        "replay": dict(method="none", sigma=0.0, replay_buffer=REPLAY_BUFFER),
    }
    for name, extra in plan.items():
        kw = dict(FIXED); kw.update(extra)
        res = run_all_seeds(HEADLINE_SEEDS, **kw)
        rm = float(np.mean(res["retention"]))
        _, p, med = paired_wilcoxon(doob_ret_star, res["retention"])
        out["methods"][name] = {
            **res, "retention_mean": rm,
            "energy_pj_gpu": step_energy_pj(extra.get("method", "none"), np_,
                                            extra.get("sigma", 0.0), substrate="gpu") * nsteps,
            "wilcoxon_vs_doob_p": p, "median_diff_doob_minus": med,
        }
        print(f"{name:11s} ret={rm:.3f}  ours-them={med:+.3f} p={p:.3f}")

    # EWC lambda scan -> best
    best_ewc, best_lam = None, None
    ewc_scan = {}
    for lam in EWC_LAMBDAS:
        kw = dict(FIXED); kw.update(method="ewc", sigma=0.0, anchor_strength=lam)
        res = run_all_seeds(HEADLINE_SEEDS, **kw)
        rm = float(np.mean(res["retention"]))
        ewc_scan[f"{lam}"] = rm
        if best_ewc is None or rm > best_ewc["retention_mean"]:
            best_ewc = {**res, "retention_mean": rm, "lambda": lam}; best_lam = lam
    _, p, med = paired_wilcoxon(doob_ret_star, best_ewc["retention"])
    best_ewc["wilcoxon_vs_doob_p"] = p; best_ewc["median_diff_doob_minus"] = med
    best_ewc["energy_pj_gpu"] = step_energy_pj("ewc", np_, 0.0, substrate="gpu") * nsteps
    out["methods"]["ewc_best"] = best_ewc
    out["ewc_scan"] = ewc_scan
    print(f"ewc(best lam={best_lam}) ret={best_ewc['retention_mean']:.3f} "
          f"ours-them={med:+.3f} p={p:.3f}")

    # summary: ours vs best baseline
    base_names = ["ou", "mesu", "ewc_best", "benna_fusi", "replay", "none"]
    best_base = max(base_names, key=lambda n: out["methods"][n]["retention_mean"])
    out["summary"] = {
        "doob_star_retention": out["methods"]["doob*"]["retention_mean"],
        "best_baseline": best_base,
        "best_baseline_retention": out["methods"][best_base]["retention_mean"],
        "ours_minus_best_baseline":
            out["methods"]["doob*"]["retention_mean"] - out["methods"][best_base]["retention_mean"],
        "gpu_energy_ratio_doob_over_ou":
            out["methods"]["doob*"]["energy_pj_gpu"] / out["methods"]["ou"]["energy_pj_gpu"],
        "bss2_energy_ratio_doob_over_gpu_ou":
            out["methods"]["doob*"]["energy_pj_bss2"] / out["methods"]["ou"]["energy_pj_gpu"],
    }
    print(f"\nours {out['summary']['doob_star_retention']:.3f} vs best baseline "
          f"{best_base} {out['summary']['best_baseline_retention']:.3f} "
          f"(+{out['summary']['ours_minus_best_baseline']:.3f})")
    save("exp3_baselines.json", out)


if __name__ == "__main__":
    main()
