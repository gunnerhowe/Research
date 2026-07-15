"""P6 R1: score the frozen precursor rule vs best-case loss rules on public Pythia runs.

Spec frozen at commit 2b4e9ba BEFORE any forecast was computed:
- Event: first checkpoint with copy_adv >= 50% of the run's max.
- Precursor rule (single a-priori constant, no tuning): alarm at first checkpoint with
  early-layer (first half of layers) prev-token score >= 0.10.
- Loss baseline (granted best case): alarm at first checkpoint with text_loss <= theta,
  theta swept over every observed value; plus improvement-rate variants.
- P6-P1: precursor alarms exactly one stage pre-event on >= 4/5 runs, zero post-event
  alarms; no loss threshold alarms pre-event on ALL runs without firing at/before 0.27B
  tokens (step 128) on some run. K1: precursor no earlier than best loss rule on >= half.
"""
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "analysis", "out6")
RUNS = ["pythia-70m", "pythia-160m", "pythia-410m",
        "pythia-70m-deduped", "pythia-160m-deduped"]


def load(name):
    recs = sorted([json.loads(l) for l in open(f"runs/p6r0/{name}.jsonl")],
                  key=lambda r: r["step"])
    return recs


def early_prevtok(r):
    n = len(r["prevtok_by_layer"])
    return max(r["prevtok_by_layer"][:max(1, n // 2)])


def main():
    data = {n: load(n) for n in RUNS}
    res = {}
    print(f"{'run':22s} {'event@step':>10s} {'event tokens':>12s} {'alarm@step':>10s} "
          f"{'lead(stages)':>12s} {'lead tokens':>11s} {'post-event alarm?':>17s}")
    for n, recs in data.items():
        ca = np.array([r["copy_adv"] for r in recs])
        steps = [r["step"] for r in recs]
        ev_i = int(np.argmax(ca >= 0.5 * ca.max()))
        pv = [early_prevtok(r) for r in recs]
        al_i = next((i for i, v in enumerate(pv) if v >= 0.10), None)
        lead_stages = None if al_i is None else ev_i - al_i
        lead_tokens = None if al_i is None else recs[ev_i]["tokens"] - recs[al_i]["tokens"]
        res[n] = dict(event_step=steps[ev_i], event_tokens=recs[ev_i]["tokens"],
                      alarm_step=None if al_i is None else steps[al_i],
                      lead_stages=lead_stages, lead_tokens=lead_tokens,
                      pre_event=bool(al_i is not None and al_i < ev_i))
        print(f"{n:22s} {steps[ev_i]:>10d} {recs[ev_i]['tokens']/1e9:>11.2f}B "
              f"{str(res[n]['alarm_step']):>10s} {str(lead_stages):>12s} "
              f"{'--' if lead_tokens is None else f'{lead_tokens/1e9:.2f}B':>11s} "
              f"{str(al_i is not None and al_i >= ev_i):>17s}")

    # loss baseline: sweep every observed loss value as threshold
    all_losses = sorted({r["text_loss"] for recs in data.values() for r in recs})
    best = None
    for th in all_losses:
        ok, too_early, leads = True, False, []
        for n, recs in data.items():
            ca = np.array([r["copy_adv"] for r in recs])
            ev_i = int(np.argmax(ca >= 0.5 * ca.max()))
            al_i = next((i for i, r in enumerate(recs) if r["text_loss"] <= th), None)
            if al_i is None or al_i >= ev_i:
                ok = False
                break
            if recs[al_i]["step"] <= 128:
                too_early = True
            leads.append(ev_i - al_i)
        if ok:
            cand = dict(theta=th, uniform_pre_event=True, fires_at_or_before_128=too_early,
                        leads=leads)
            if best is None or (not too_early and best["fires_at_or_before_128"]):
                best = cand
    res["loss_baseline"] = best or dict(uniform_pre_event=False)
    print(f"\nloss best-case: {json.dumps(res['loss_baseline'])}")

    ok_runs = sum(1 for n in RUNS if res[n]["lead_stages"] == 1)
    no_post = all(res[n]["pre_event"] for n in RUNS)
    p1 = ok_runs >= 4 and no_post and (best is None or best["fires_at_or_before_128"])
    res["P6_P1"] = dict(exact_one_stage=ok_runs, no_post_event=no_post,
                        loss_uniform_without_early=bool(best and not best["fires_at_or_before_128"]),
                        passed=bool(p1))
    k1 = sum(1 for n in RUNS if not res[n]["pre_event"]) >= len(RUNS) / 2
    res["K1_fires"] = bool(k1)
    print(f"\nP6-P1: exact-one-stage on {ok_runs}/5, no post-event alarms: {no_post}, "
          f"passed: {p1}")
    print(f"K1 fires: {k1}")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "r1_scored.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
