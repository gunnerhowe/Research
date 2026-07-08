"""E1 — generality of the separation (runs only if E0 passed).

Sweep (i) hidden-state noise sigma and (ii) prune fraction; report all three metrics
per condition x seed. Deliverable: the d̄-vs-DSA gap as a function of how stochastic
the "sameness" is. Reduced d̄ budget (ns subset of the claims set, repeats=3, no
belief readout) — pre-registered metrics unchanged.
"""
import json
import time

from exp_common import (RESULTS, SEEDS, M, TASKS, eval_pair, gen_runs, get_model,
                        get_states, save_json)

assert json.loads((RESULTS / "exp0_existence.json").read_text())["gate"]["E0_PASS"], \
    "E0 gate did not pass; E1 is gated (PLAN.md)"

t0 = time.time()
gm = TASKS["gm"]()
NS = (1, 2, 4, 8, 16, 24)
SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
FRACS = (0.1, 0.3, 0.5, 0.7, 0.9)

noise_rows, prune_rows = [], []
for s in SEEDS:
    A, _ = get_model("base", "gm", s)
    runsA = gen_runs(A, gm, 0.0, seed_base=s * 100_000 + 1)
    stA = get_states(A, gm, 0.0)

    for sig in SIGMAS:
        runsB = gen_runs(A, gm, sig, seed_base=s * 100_000 + 60_000
                         + int(sig * 10_000))
        stB = get_states(A, gm, sig, seed=555 + s)
        r = eval_pair(runsA, runsB, stA, stB, gm.m, ns=NS, repeats=3,
                      seed=s, with_belief=False)
        r.update(seed=s, sigma=sig)
        noise_rows.append(r)
        print(f"[noise] s{s} sigma={sig}: dbar*={r['plateau']['dbar']:.4f} "
              f"CKA={r['cka']:.4f} DSA={r['dsa']:.4f} tv1={r['tv_unigram']:.4f} "
              f"({time.time()-t0:.0f}s)")

    for frac in FRACS:
        P, _ = get_model("prune", "gm", s, frac=frac)
        runsB = gen_runs(P, gm, 0.0, seed_base=s * 100_000 + 80_000
                         + int(frac * 100))
        stB = get_states(P, gm, 0.0, seed=555 + s)
        r = eval_pair(runsA, runsB, stA, stB, gm.m, ns=NS, repeats=3,
                      seed=s + 7, with_belief=False)
        r.update(seed=s, frac=frac, ce=M.val_ce_bits(P, gm))
        prune_rows.append(r)
        print(f"[prune] s{s} frac={frac}: dbar*={r['plateau']['dbar']:.4f} "
              f"CKA={r['cka']:.4f} DSA={r['dsa']:.4f} tv1={r['tv_unigram']:.4f} "
              f"CE={r['ce']:.4f} ({time.time()-t0:.0f}s)")

save_json("exp1_generality.json", {"noise": noise_rows, "prune": prune_rows,
                                   "sigmas": SIGMAS, "fracs": FRACS, "ns": NS})
print(f"done ({time.time()-t0:.0f}s)")
