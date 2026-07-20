"""P7 R2 sealed one-shot scorer (prereg plan_p7.md commit 1affb4a).

Refuses on a partial fleet or an existing scorefile. Endpoints reused per the disclosed
non-ideality: beta=0 from grid6r2 rep (n=30), beta=8 from grid7c hard (n=10).
Scores P-D1/K-D1 (dose), P-D2/K-D2 (placement), P-D3/K-D3 (two-gate floor), K-D4
(manipulation). Output: analysis/out7/p7d_scored.json
"""
import glob
import json
import os

import numpy as np

T0 = 6300.0
BETAS = [1, 2, 3, 4, 6]
FULL_ACCEL = T0 - 3600.0            # grid7c hard endpoint
BAR_D1 = T0 - 0.50 * FULL_ACCEL     # 4,950 — P-D1 requires beta=4 at or below this
KILL_D1 = T0 - 0.25 * FULL_ACCEL    # 5,670 — K-D1 fires above this
BAR_D2 = 0.80 * T0                  # 5,040 — P-D2 null band
KILL_D2 = 0.75 * T0                 # 4,725 — K-D2 fires at or below this
FLOOR_D3 = 3400.0


def load(pattern, root="runs/grid7d"):
    out = []
    for d in sorted(glob.glob(f"{root}/{pattern}")):
        if not os.path.exists(f"{d}/summary.json"):
            continue
        s = json.load(open(f"{d}/summary.json"))
        recs = [json.loads(l) for l in open(f"{d}/metrics.jsonl") if l.strip()]
        out.append(dict(name=os.path.basename(d), t_event=s["t_event"], recs=recs))
    return out


def med(runs):
    te = [r["t_event"] for r in runs if r["t_event"] is not None]
    return (float(np.median(te)) if te else None, f"{len(te)}/{len(runs)}", sorted(te))


def main():
    os.makedirs("analysis/out7", exist_ok=True)
    if os.path.exists("analysis/out7/p7d_scored.json"):
        raise RuntimeError("p7d_scored.json exists — one-shot already ran")
    dose = {b: load(f"dose{b}_s*") for b in BETAS}
    l1 = load("hardL1_s*")
    counts = {f"dose{b}": len(v) for b, v in dose.items()}
    counts["hardL1"] = len(l1)
    if any(v != 5 for v in counts.values()):
        raise RuntimeError(f"fleet incomplete: {counts} — not scoring")

    res = dict(T0=T0, counts=counts, bars=dict(
        P_D1_beta4_at_or_below=BAR_D1, K_D1_above=KILL_D1,
        P_D2_at_or_above=BAR_D2, K_D2_at_or_below=KILL_D2, P_D3_floor=FLOOR_D3))
    # dose curve, with the two disclosed reused endpoints flagged
    curve = {}
    ctrl = [json.load(open(f))["t_event"] for f in glob.glob("runs/grid6r2/rep_s*/summary.json")]
    curve["0"] = dict(median=float(np.median([t for t in ctrl if t])), n=30, reused=True)
    for b in BETAS:
        m, ev, te = med(dose[b])
        first = [r["recs"][0]["prevtok_by_layer"][0] for r in dose[b]]
        curve[str(b)] = dict(median=m, evented=ev, events=te, n=5, reused=False,
                             first_eval_prevtok0=round(float(np.median(first)), 4))
    hard7c = [json.load(open(f))["t_event"] for f in glob.glob("runs/grid7c/hard_s*/summary.json")]
    curve["8"] = dict(median=float(np.median([t for t in hard7c if t])), n=10, reused=True)
    res["dose_curve"] = curve
    m1, ev1, te1 = med(l1)
    first_l1 = [r["recs"][0]["prevtok_by_layer"][-1] for r in l1]
    res["hardL1"] = dict(median_t_event=m1, evented=ev1, events=te1,
                         first_eval_prevtok_layer1=round(float(np.median(first_l1)), 4))

    v = {}
    b4 = curve["4"]["median"]
    order = [curve[str(b)]["median"] for b in [0] + BETAS + [8]]
    mono = all(a is None or b is None or a >= b - 1e-9
               for a, b in zip(order, order[1:]))
    v["P_D1_graded"] = (f"{'PASS' if (b4 is not None and b4 <= BAR_D1) else 'FAIL'} "
                        f"(beta4 {b4} vs bar {BAR_D1}); monotone_nonincreasing={mono}")
    v["K_D1_needs_converged_head"] = ("FIRES — retire 'supplying the precursor'"
                                      if (b4 is None or b4 > KILL_D1) else "no-fire")
    v["P_D2_placement_null"] = (f"{'PASS' if (m1 is None or m1 >= BAR_D2) else 'FAIL'} "
                                f"(hardL1 {m1} vs bar {BAR_D2})")
    v["K_D2_not_composition"] = ("FIRES — composition framing dies"
                                 if (m1 is not None and m1 <= KILL_D2) else "no-fire")
    cells = {f"dose{b}": curve[str(b)]["median"] for b in BETAS}
    cells["hardL1"] = m1
    below = {k: x for k, x in cells.items() if x is not None and x < FLOOR_D3}
    v["P_D3_two_gate_floor"] = f"{'PASS' if not below else 'FAIL'} (below floor: {below})"
    v["K_D3_floor_breached"] = ("FIRES — two-gate account incomplete" if below else "no-fire")
    manip_l1 = res["hardL1"]["first_eval_prevtok_layer1"]
    doses_mono = [curve[str(b)]["first_eval_prevtok0"] for b in BETAS]
    v["K_D4_manipulation"] = (
        f"{'PASS' if (manip_l1 >= 0.8 and all(a <= b + 1e-9 for a, b in zip(doses_mono, doses_mono[1:]))) else 'FAIL — RUNG VOID'}"
        f" (L1 prevtok {manip_l1}; dose prevtok0 {doses_mono})")
    res["verdicts"] = v
    json.dump(res, open("analysis/out7/p7d_scored.json", "w"), indent=1)
    print(json.dumps(v, indent=1))
    print("\ndose curve (median t_event by beta):")
    for k in ["0"] + [str(b) for b in BETAS] + ["8"]:
        c = curve[k]
        print(f"  beta={k:>2s} n={c['n']:>2d} median={c['median']} "
              f"{'(reused endpoint)' if c['reused'] else 'prevtok0@first=' + str(c.get('first_eval_prevtok0'))}")
    print("hardL1:", res["hardL1"])


if __name__ == "__main__":
    main()
