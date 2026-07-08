"""E2 — the readout ports beyond toy RNNs (runs only if E0 passed).

Small causal Transformer-LMs on GM and MESS3; pairs: noise-in-residual twin
(sigma* recalibrated by the pre-registered rule) and a distilled twin; states =
final-layer residual stream; the GM-vs-MESS3 cross pair anchors "different" for
CKA/DSA. Budget: B=64 x T=8192 per run (PLAN.md), claims at n <= k_wall.
"""
import json
import time

from exp_common import (GEN_KW_TF, RESULTS, M, TASKS, calibrate_sigma, eval_pair,
                        gen_runs, get_model, get_states, save_json)

# gated on E0 (PLAN.md): pre-registered pass OR amended pass (AMENDMENT 1)
_e0 = json.loads((RESULTS / "exp0_existence.json").read_text())["gate"]["E0_PASS"]
_am = json.loads((RESULTS / "amended_gate.json").read_text())
assert _e0 or _am["AMENDED_PASS_sign_consistent"], \
    "neither pre-registered nor amended E0 gate passed; E2 is gated (PLAN.md)"

t0 = time.time()
SEEDS_TF = (0, 1, 2)
NS = (1, 2, 3, 4, 6, 8, 12, 16)
results = {"gm": {"noise": [], "distill": []},
           "mess3": {"noise": [], "distill": []},
           "cross": [], "meta": {"sigma_star": {}, "val_ce": {}, "gen_kw": GEN_KW_TF}}

for task_name in ("gm", "mess3"):
    task = TASKS[task_name]()
    for s in SEEDS_TF:
        A, _ = get_model("tf-base", task_name, s)
        results["meta"]["val_ce"][f"{task_name}_s{s}"] = M.val_ce_bits(A, task)
        runsA = gen_runs(A, task, 0.0, seed_base=s * 100_000 + 1, arch="tf")
        stA = get_states(A, task, 0.0, arch="tf")

        sig, calib = calibrate_sigma(A, task, seed=s, arch="tf")
        results["meta"]["sigma_star"][f"{task_name}_s{s}"] = {"sigma": sig,
                                                              "calib": calib}
        print(f"[{task_name} s{s}] sigma* = {sig} ({time.time()-t0:.0f}s)")

        D, _ = get_model("tf-distill", task_name, s)
        twins = {"noise": (A, sig), "distill": (D, 0.0)}
        for name, (mB, sigB) in twins.items():
            runsB = gen_runs(mB, task, sigB, arch="tf",
                             seed_base=s * 100_000 + 40_000
                             + 3_000 * list(twins).index(name))
            stB = get_states(mB, task, sigB, arch="tf", seed=555 + s)
            r = eval_pair(runsA, runsB, stA, stB, task.m, ns=NS, repeats=2,
                          seed=s, with_belief=(task.m == 2), belief_ns_max=8,
                          dsa_kw={"iters": 800})   # AMENDMENT 2 budget
            r.update(seed=s, pair=name, task=task_name)
            results[task_name][name].append(r)
            print(f"[{task_name} s{s}] {name}: dbar*={r['plateau']['dbar']:.4f} "
                  f"(floor {r['plateau']['floor']:.4f}, n*={r['plateau']['n']}) "
                  f"CKA={r['cka']:.4f} DSA={r['dsa']:.4f} "
                  f"tv1={r['tv_unigram']:.4f} ({time.time()-t0:.0f}s)")

# cross-task anchor for "different" under CKA/DSA (same eval-input task: gm inputs)
from dbar_diff.baselines import cka_states, dsa_distance  # noqa: E402
for s in SEEDS_TF:
    A, gm = get_model("tf-base", "gm", s)
    B, _ = get_model("tf-base", "mess3", s)
    stA = get_states(A, gm, 0.0, arch="tf")
    stB = get_states(B, gm, 0.0, arch="tf", seed=556 + s)   # driven by GM inputs
    results["cross"].append({"seed": s,
                             "cka": cka_states(stA, stB),
                             "dsa": dsa_distance(stA, stB, device=M.DEVICE)})
    print(f"[cross s{s}] CKA={results['cross'][-1]['cka']:.4f} "
          f"DSA={results['cross'][-1]['dsa']:.4f}")

save_json("exp2_transformer.json", results)
print(f"done ({time.time()-t0:.0f}s)")
