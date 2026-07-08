"""E0 — the existence proof and HARD GATE (PLAN.md).

Per seed: base GM model + five twins (recoded null, independent seed, noise-injected,
distilled, pruned) + the FEVEN different-task control. For every pair: CKA, DSA, and
d̄ side-by-side with same-process floors, entropy/wall diagnostics, and the marginal
(d̄_1) check. Then the pre-registered gate + validation battery + the convergence
study (V3).
"""
import time

import numpy as np

from exp_common import (GEN_KW, NS_FULL, SEEDS, M, TASKS, calibrate_sigma,
                        eval_pair, gate_report, gen_runs, get_model, get_states,
                        plateau, save_json, dbar_pair_curve)

t0 = time.time()
gm = TASKS["gm"]()
feven = TASKS["feven"]()

# ---------------------------------------------------------------- models + runs
base, base_runs, base_states = {}, {}, {}
for s in SEEDS:
    base[s], _ = get_model("base", "gm", s)
    base_runs[s] = gen_runs(base[s], gm, 0.0, seed_base=s * 100_000 + 1)
    base_states[s] = get_states(base[s], gm, 0.0)
    print(f"[seed {s}] base ready ({time.time()-t0:.0f}s)")

results = {p: [] for p in ("null", "seed", "noise", "distill", "prune", "difftask")}
meta = {"sigma_star": {}, "calib": {}, "val_ce": {},
        "task_h": gm.entropy_rate(), "gen_kw": GEN_KW}

for s in SEEDS:
    A, runsA, stA = base[s], base_runs[s], base_states[s]
    meta["val_ce"][s] = M.val_ce_bits(A, gm)

    # --- twins
    recoded = M.recode_permute(A, seed=s + 400)
    sig, calib = calibrate_sigma(A, gm, seed=s)
    meta["sigma_star"][s], meta["calib"][s] = sig, calib
    print(f"[seed {s}] sigma* = {sig}")
    distilled, _ = get_model("distill", "gm", s)
    pruned, _ = get_model("prune", "gm", s, frac=0.5)
    fev, _ = get_model("feven-base", "feven", s)

    twins = {
        "null": (recoded, 0.0, gm),
        "noise": (A, sig, gm),
        "distill": (distilled, 0.0, gm),
        "prune": (pruned, 0.0, gm),
        "difftask": (fev, 0.0, feven),
    }
    for name, (mB, sigB, taskB) in twins.items():
        runsB = gen_runs(mB, taskB, sigB, seed_base=s * 100_000 + 30_000
                         + 3_000 * list(twins).index(name))
        stB = get_states(mB, gm, sigB, seed=555 + s)   # shared GM eval inputs
        r = eval_pair(runsA, runsB, stA, stB, gm.m, seed=s)
        r["seed"], r["pair"] = s, name
        results[name].append(r)
        print(f"[seed {s}] {name}: dbar*={r['plateau']['dbar']:.4f} "
              f"(floor {r['plateau']['floor']:.4f}, n*={r['plateau']['n']}) "
              f"CKA={r['cka']:.4f} DSA={r['dsa']:.4f} tv1={r['tv_unigram']:.4f} "
              f"({time.time()-t0:.0f}s)")

    # --- independent-seed pair (report only)
    s2 = (s + 1) % len(SEEDS)
    r = eval_pair(runsA, base_runs[s2], stA, base_states[s2], gm.m, seed=s + 50)
    r["seed"], r["pair"] = s, f"seed{s}v{s2}"
    results["seed"].append(r)
    print(f"[seed {s}] seed-pair vs {s2}: dbar*={r['plateau']['dbar']:.4f} "
          f"CKA={r['cka']:.4f} DSA={r['dsa']:.4f}")

# ------------------------------------------------------------------- gate + V1/V2
nu = float(np.mean([r["dsa"] for r in results["null"]]))
kappa = float(np.mean([r["dsa"] for r in results["difftask"]]))
gates = {p: gate_report(results[p], nu, kappa)
         for p in ("noise", "distill", "prune")}
e0_pass = any(g["PASS"] for g in gates.values())

v1_rows = []
for r in results["null"]:
    ok = all(row["dbar"] <= 2 * row["floor"] for row in r["dbar_curve"]
             if 2 <= row["n"] <= r["k_wall"]) and r["cka"] >= 0.99
    v1_rows.append(bool(ok))
v2_rows = []
for r in results["difftask"]:
    ok = (r["plateau"]["dbar"] >= 10 * r["plateau"]["floor"]
          and r["dsa"] >= 5 * nu)
    v2_rows.append(bool(ok))

# ------------------------------------------------------------- V3 convergence
def convergence_study(runsA, runsB, m, n_grid=(10_000, 30_000, 100_000, 300_000,
                                               1_000_000, 2_097_152)):
    B = len(runsA[0]["sym"])
    out = []
    for N in n_grid:
        t = max(64, N // B)
        cut = lambda runs: [[c[:t] for c in runs[i]["sym"]] for i in (0, 1)]
        a1, a2 = cut(runsA)
        b1, b2 = cut(runsB)
        rows = dbar_pair_curve(a1, b1, a2, b2, m, ns=NS_FULL, repeats=2, seed=N % 7919)
        out.append({"N": int(B * t), "rows": rows})
    return out

best_pair = max(gates, key=lambda p: gates[p]["delta_mean"])
conv = convergence_study(base_runs[0],
                         gen_runs(base[0] if best_pair == "noise"
                                  else get_model(best_pair, "gm", 0,
                                                 **({"frac": 0.5} if best_pair == "prune" else {}))[0],
                                  gm, meta["sigma_star"][0] if best_pair == "noise" else 0.0,
                                  seed_base=777_000),
                         gm.m)

save_json("exp0_existence.json", {
    "meta": meta, "results": results,
    "gate": {"nu_dsa": nu, "kappa_dsa": kappa, "pairs": gates,
             "E0_PASS": bool(e0_pass)},
    "validation": {"V1_null_ok": v1_rows, "V2_difftask_ok": v2_rows},
    "convergence": {"pair": best_pair, "curves": conv},
})

print("\n================ E0 GATE ================")
for p, g in gates.items():
    print(f"{p:8s} PASS={g['PASS']} (CKAsim={g['cka_similar']} "
          f"DSAsim={g['dsa_similar']} dbar_sep={g['dbar_separates']} "
          f"checks={g['checks']})")
print(f"V1 null ok: {v1_rows}\nV2 difftask ok: {v2_rows}")
print(f"E0 {'PASSES' if e0_pass else 'FAILS -> K1 (honest negative)'} "
      f"({time.time()-t0:.0f}s total)")
