# emergence-watch

**Training telemetry for capability emergence** — circuit-precursor alarms, calibrated
timing forecasts, and certified false-alarm rates, live during training or over
checkpoint streams. The methodology of *"Capability Emergence Can Be Forecast"* as a
library.

```python
from emergence_watch import (LiveMonitor, EmergenceWatchCallback, ComposedAnchor,
                             Signal, paper_induction_event, make_induction_probe_fn,
                             MultiplicativeForecaster)

probe_fn = make_induction_probe_fn(vocab_size=50257, device="cuda",
                                   behavioral_fn=my_indist_copy_ramp)   # your 2nd gate
anchor   = ComposedAnchor([Signal("prevtok_early", 0.10),
                           Signal("behavioral_ramp", 0.10)])            # calibrate your own
monitor  = LiveMonitor(probe_fn, anchor, event=paper_induction_event(),
                       forecaster=MultiplicativeForecaster.load(my_frozen_json),
                       cadence=200, jsonl_path="emergence.jsonl")

trainer = Trainer(..., callbacks=[EmergenceWatchCallback(monitor)])
# ... alarm fires mid-training with a calibrated arrival interval attached.
```

## What this is (and is not)

A **named-capability tripwire with a scored forecast** — you pick the capability, watch
its mechanistic precursor, and get an alarm with a calibrated arrival interval. It is
**not** an unknown-capability radar: signals that claim to detect "emergence in
general" failed our pre-registered confirmations (they detected the training
intervention, not the emergence).

Every default here encodes a failure mode we measured:

| Default | The result that bought it |
|---|---|
| Anchors are **composed** (circuit AND behavior, same eval) | bare precursor: 10/10 false alarms in trap languages; conjunction: 0/33, rho 1.000 |
| Events are **absolute** crossings with sustain | %-of-max criteria silently broke twice on late outliers |
| Forecasts are **multiplicative** (t_event ≈ 1.19 × t_anchor in our substrate) | fixed-step offsets break under any shift that moves the emergence timescale |
| FA rates need **≥ 5 manufactured negatives** | a 1-negative cap once certified a degenerate fit |
| Frozen artifacts are **hashed**; blind gates score **once** | that's why our 10/10, 9/10, 5/5 blind coverages are believable |

## The workflow

1. **Probes**: use the induction instruments (validated on Pythia, OLMo-1, OLMo-2) or
   implement your capability's probe pair (precursor circuit + behavioral ramp).
2. **Fleet**: train N seeds at your config; detect events with `AbsoluteEvent`.
3. **Negatives**: manufacture capability-blocked runs (ablate the capability's training
   signal; block the architecture) — false-alarm rates are meaningless without them.
4. **Calibrate**: `calibrate_multiplicative(anchor_times, event_times)` → freeze →
   commit the hash.
5. **Monitor**: attach `EmergenceWatchCallback` (or run `LiveMonitor.step` from any
   loop; or probe checkpoint streams offline).
6. **Certify**: `report_card()` gives (ranking, lead) pairs, misses, FA; `BlindGate`
   enforces one-shot blind validation. Re-certify after any recipe change.

Flash-attention note: probe forwards request eager attention for the probe batch only
(`eager_attention` context manager); overhead at cadence 200 is negligible.

## Status

v0.1.0.dev0 — extracted from the research code behind the paper; CPU test suite in
`tests/`. Release smoke (2026-07-18): `examples/watch_checkpoint_stream.py` on the
public Pythia-70m suite reproduces the paper's ordering end-to-end — precursor alarm at
step256 (prevtok 0.156, copy_adv 0.0004 ≈ absent), capability lands by step1000
(copy_adv 9.6 → 11.5 nats), ~3 s/probe on CPU. The full experimental record (fleets,
negatives, blind gates, kill ledger) lives in the companion benchmark (`BENCHMARK.md`,
tag `emergence-benchmark-v0.1`).
