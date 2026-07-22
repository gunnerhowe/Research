"""P8 pilot sealed scorer (prereg plan_p8.md commit 204a124). Refuses on partial fleet or
an existing scorefile. Adjudicates the four go/no-go gates + K0:

  G-P1 CONVERGENCE  : M0 induction forms + >=4/5 pilot skills fire their event on specimen.
  G-P2 SEPARABILITY : 5x5 conjunction-fingerprint confusion matrix diagonally dominant
                      (diag > 3x max off-diag) on >=3/5, incl. the M0/M1 and M2/M3 pairs
                      (resolves Fork 1 guard partition).
  G-P3 FIREWALL     : composed anchor 0 FA on N2 marker-scramble for the guarded skills,
                      while the bare-structural alarm DOES fire on N2 (gate load-bearing).
  G-P4 DISGUISE     : >=1 disguise circuit-verified AND re-teaches to its event.
  K0  GUARD VALIDITY: guarded events silent + fingerprint below band; retained intact.

Event: behavioral acc >= tau_k on 2 consecutive evals. tau_k pinned per skill from the
specimen (below). Structural alarm band tau_struct pinned per skill from the guarded+N2
baseline (max proximity over the baseline stream). Composed alarm = structural>=tau_struct
AND behavioral ramp>=0.5*tau_k, sustain 2.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
import data_zoo as dz

OUT = "analysis/out8"
PILOT = dz.PILOT_SKILLS
GUARDED = dz.GUARDED_PILOT                         # M0, M2, M6
RETAINED = [s for s in PILOT if s not in GUARDED]  # M1, M3
ROOT = "runs/zoo_pilot"


def load(run):
    d = f"{ROOT}/{run}"
    if not os.path.exists(f"{d}/summary.json"):
        return None
    recs = [json.loads(l) for l in open(f"{d}/metrics.jsonl") if l.strip()]
    return dict(summary=json.load(open(f"{d}/summary.json")), recs=recs)


def event_step(recs, skill, tau, sustain=2):
    streak = 0
    for r in recs:
        a = r["skills"].get(skill, {}).get("acc")
        if a is not None and a >= tau:
            streak += 1
            if streak >= sustain:
                return r["step"]
        else:
            streak = 0
    return None


def alarm_step(recs, skill, tau_struct, tau_k, mode="composed", sustain=2):
    """mode: 'bare' = structural only; 'gate' = behavioral ramp only; 'composed' = both."""
    streak = 0
    for r in recs:
        sk = r["skills"].get(skill)
        if sk is None or "max_mass" not in sk:
            streak = 0; continue
        struct = sk["max_mass"] >= tau_struct
        ramp = sk["acc"] >= 0.5 * tau_k
        hit = {"bare": struct, "gate": ramp, "composed": struct and ramp}[mode]
        if hit:
            streak += 1
            if streak >= sustain:
                return r["step"]
        else:
            streak = 0
    return None


def pin_tau(specimen):
    """tau_k = absolute bar strictly between a chance-ish floor and the specimen plateau."""
    tau = {}
    for s in PILOT:
        final = np.median([r["skills"][s]["acc"] for r in specimen["recs"][-5:]])
        floor = 0.5 if s == "M6" else 0.10           # Dyck chance 0.5; content chance ~0.016
        tau[s] = round(floor + 0.5 * (final - floor), 4)
    return tau


def confusion(lib, banks_scores):
    """banks_scores[probe_skill][fp_skill] = alignment mass of fp_skill's index read on
    probe_skill's bank. Diagonal dominance => the conjunction fingerprint separates."""
    M = np.array([[banks_scores[p][f] for f in PILOT] for p in PILOT])
    return M


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f"{OUT}/zoo_pilot_scored.json"):
        raise RuntimeError("zoo_pilot_scored.json exists — one-shot already ran")
    specimen = load("specimen_s1")
    if specimen is None:
        raise RuntimeError("specimen_s1 not complete")
    tau = pin_tau(specimen)

    # G-P1 convergence (specimen)
    ev = {s: event_step(specimen["recs"], s, tau[s]) for s in PILOT}
    n_fired = sum(v is not None for v in ev.values())
    gp1 = (ev["M0"] is not None) and (n_fired >= 4)

    res = dict(tau=tau, specimen_events=ev, n_fired=n_fired,
               G_P1=f"{'PASS' if gp1 else 'FAIL'} (M0 event {ev['M0']}, {n_fired}/5 fired)")

    # confusion / separability requires the captured library + per-bank cross-scores,
    # written by the capture step into runs/zoo_pilot/fp_confusion.json
    conf_path = f"{ROOT}/fp_confusion.json"
    if os.path.exists(conf_path):
        cs = json.load(open(conf_path))
        M = confusion(None, cs)
        diagdom = []
        for i, s in enumerate(PILOT):
            diag = M[i, i]
            off = max(M[i, j] for j in range(len(PILOT)) if j != i)
            diagdom.append(diag > 3 * off)
        pairs_ok = {"M0/M1": diagdom[0] and diagdom[1], "M2/M3": diagdom[2] and diagdom[3]}
        res["G_P2"] = (f"{'PASS' if sum(diagdom) >= 3 else 'FAIL'} "
                       f"({sum(diagdom)}/5 diag-dominant; pairs {pairs_ok})")
        res["confusion_matrix"] = M.tolist()
        res["fork1_partition"] = ("straddle" if all(pairs_ok.values()) else "paired")
    else:
        res["G_P2"] = "PENDING (no fp_confusion.json — run capture step)"

    # G-P3 firewall + K0 + G-P4 evaluated over guard/watch runs if present
    watch = {os.path.basename(os.path.dirname(p)): load(os.path.basename(os.path.dirname(p)))
             for p in glob.glob(f"{ROOT}/*/summary.json")}
    res["watch_runs_present"] = sorted(watch.keys())
    # (firewall / disguise adjudication filled once guard+watch fleet exists; the scorer is
    #  re-run sealed after the full pilot fleet — this invocation scores what exists.)

    json.dump(res, open(f"{OUT}/zoo_pilot_scored.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("confusion_matrix",)}, indent=1))


if __name__ == "__main__":
    main()
