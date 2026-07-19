"""P7 R1 sealed one-shot scorer (prereg plan_p7.md commit b8eb219).

Refuses to run on a partial fleet or if the scorefile exists. Controls = grid6r2 rep
(n=30, T0 = 6,300). Scores P-C1/P-C1s/P-C2 (hard vs T0), P-C3/K-C1 (sink placebo),
P-C4 (seed stickiness), P-C5/K-C3 (norephard), K-C2 (manipulation check), and reports
the exploratory near arm. Output: analysis/out7/p7c_scored.json
"""
import glob
import json
import os

import numpy as np

T0 = 6300.0
BAR_CLOCK = 0.5 * T0        # P-C1
LAW_BAND = (0.08 * T0, 0.31 * T0)  # P-C1s
BAR_PASSENGER = 0.8 * T0    # P-C2 / P-C3


def load(pat):
    out = []
    for d in sorted(glob.glob(f"runs/grid7c/{pat}")):
        if not os.path.exists(f"{d}/summary.json"):
            continue
        s = json.load(open(f"{d}/summary.json"))
        recs = [json.loads(l) for l in open(f"{d}/metrics.jsonl") if l.strip()]
        out.append(dict(name=os.path.basename(d), t_event=s["t_event"], recs=recs))
    return out


def med_events(runs):
    te = [r["t_event"] for r in runs if r["t_event"] is not None]
    return (float(np.median(te)) if te else None,
            f"{len(te)}/{len(runs)}", sorted(te))


def main():
    os.makedirs("analysis/out7", exist_ok=True)
    if os.path.exists("analysis/out7/p7c_scored.json"):
        raise RuntimeError("p7c_scored.json exists — one-shot already ran")
    arms = {a: load(f"{a}_s*") for a in ("hard", "seed", "sink", "near", "norephard")}
    counts = {a: len(v) for a, v in arms.items()}
    if counts != {"hard": 10, "seed": 10, "sink": 5, "near": 5, "norephard": 5}:
        raise RuntimeError(f"fleet incomplete: {counts} — not scoring")

    # K-C2 manipulation check: first eval prevtok0 >= 0.8 on every biased-prev run
    manip = {}
    for a in ("hard", "seed", "norephard"):
        manip[a] = [round(r["recs"][0]["prevtok_by_layer"][0], 4) for r in arms[a]]
    kc2_pass = all(v >= 0.8 for a in manip for v in manip[a])

    res = dict(T0=T0, counts=counts, manipulation_first_eval_prevtok0=manip,
               K_C2_manipulation=("PASS" if kc2_pass else "FAIL — RUNG VOID"))
    for a in ("hard", "seed", "sink", "near", "norephard"):
        m, evented, te = med_events(arms[a])
        res[a] = dict(median_t_event=m, evented=evented, events=te,
                      frac_T0=(round(m / T0, 4) if m else None))

    h = res["hard"]["median_t_event"]
    s = res["sink"]["median_t_event"]
    verdicts = {}
    if h is None:
        verdicts["P-C1"] = "NO EVENTS in hard arm — scaffold BLOCKS emergence (report)"
    else:
        verdicts["P-C1_clock"] = f"{'PASS' if h <= BAR_CLOCK else 'FAIL'} (hard {h:.0f} vs bar {BAR_CLOCK:.0f})"
        verdicts["P-C1s_lawband"] = (
            f"{'PASS' if LAW_BAND[0] <= h <= LAW_BAND[1] else 'FAIL'} "
            f"(hard {h:.0f} vs [{LAW_BAND[0]:.0f}, {LAW_BAND[1]:.0f}], residual pred ~989)")
        verdicts["P-C2_passenger"] = ("FIRES (precursor not rate-limiting)"
                                      if h >= BAR_PASSENGER else "no-fire")
    sink_speeds = s is not None and s <= BAR_CLOCK
    verdicts["K_C1_placebo"] = ("FIRES — nonspecific speedup, clock claim dead"
                                if sink_speeds else
                                f"no-fire (sink {'none' if s is None else f'{s:.0f}'})")
    verdicts["P-C3_specificity"] = ("PASS" if (s is None or s >= BAR_PASSENGER)
                                    else f"FAIL (sink {s:.0f} < {BAR_PASSENGER:.0f})")
    # P-C4 stickiness: seed median in [hard, T0]; decay = prevtok0 < 0.5 pre-event
    sd = res["seed"]["median_t_event"]
    if sd is not None and h is not None:
        verdicts["P-C4_seed"] = f"{'PASS' if h <= sd <= T0 else 'FAIL'} (seed {sd:.0f})"
    decay = 0
    for r in arms["seed"]:
        te = r["t_event"] if r["t_event"] is not None else float("inf")
        if any(rec["prevtok_by_layer"][0] < 0.5 for rec in r["recs"]
               if rec["step"] <= te):
            decay += 1
    res["seed"]["prevtok_decay_pre_event"] = f"{decay}/10"
    ne = int(res["norephard"]["evented"].split("/")[0])
    verdicts["P-C5_data_necessity"] = f"{'PASS' if ne == 0 else 'FAIL'} ({ne}/5 evented)"
    verdicts["K_C3_scaffold_suffices"] = ("FIRES — STOP AND RE-EXAMINE"
                                          if ne >= 2 else "no-fire")
    res["verdicts"] = verdicts
    json.dump(res, open("analysis/out7/p7c_scored.json", "w"), indent=1)
    print(json.dumps(res["verdicts"], indent=1))
    for a in ("hard", "seed", "sink", "near", "norephard"):
        print(a, res[a]["median_t_event"], res[a]["evented"], "frac", res[a]["frac_T0"])


if __name__ == "__main__":
    main()
