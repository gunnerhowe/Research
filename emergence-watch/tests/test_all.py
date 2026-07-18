"""CPU-only test suite for emergence-watch. Run: python tests/test_all.py
(pytest-compatible names; zero GPU, zero network, zero HF dependency)."""
import json
import sys
import pathlib
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from emergence_watch import (
    repeated_probe_batch, copy_advantage, prefix_matching_score, prev_token_score,
    Signal, ComposedAnchor, AbsoluteEvent, MultiplicativeForecaster,
    calibrate_multiplicative, BlindGate, report_card, LiveMonitor,
)


def test_probe_math_ground_truth():
    B, H, L = 2, 3, 8
    T = 2 * L
    # attention exactly to i-1 -> prev_token_score == 1
    a = torch.zeros(B, H, T, T)
    for t in range(1, T):
        a[:, :, t, t - 1] = 1.0
    a[:, :, 0, 0] = 1.0
    assert abs(prev_token_score(a) - 1.0) < 1e-6
    # attention from second-half queries exactly to (prev occurrence + 1) -> prefix == 1
    a2 = torch.zeros(B, H, T, T)
    for i in range(L, 2 * L - 1):
        a2[:, :, i, i - L + 1] = 1.0
    assert abs(prefix_matching_score(a2, L) - 1.0) < 1e-6
    # copy advantage: uniform logits first half, near-delta on the target second half
    V = 50
    ids = repeated_probe_batch(V, batch=4, half_len=L, lo=1)
    logits = torch.zeros(4, T, V)
    for b in range(4):
        for t in range(L - 1, T - 1):
            logits[b, t, ids[b, t + 1]] = 10.0
    adv = copy_advantage(logits, ids, L)
    assert adv > 2.0, f"expected strong positive copy advantage, got {adv}"
    print("probe math: OK")


def test_anchor_composition_and_sustain():
    anchor = ComposedAnchor([Signal("a", 0.5), Signal("b", 0.5)])
    assert not anchor.update(0, {"a": 0.9, "b": 0.1})     # only one gate up
    assert not anchor.update(1, {"a": 0.1, "b": 0.9})     # other gate, not same eval
    assert anchor.update(2, {"a": 0.9, "b": 0.9})         # conjunction fires
    assert anchor.fired_at == 2
    assert not anchor.update(3, {"a": 0.9, "b": 0.9})     # fires once
    sust = ComposedAnchor([Signal("a", 0.5, sustain=2)])
    assert not sust.update(0, {"a": 0.9})
    assert sust.update(1, {"a": 0.9}) and sust.fired_at == 1
    print("anchors: OK")


def test_event_convention_matches_paper():
    ev = AbsoluteEvent("copy_adv", 2.0, sustain=2)
    assert not ev.update(100, {"copy_adv": 2.5})
    assert not ev.update(125, {"copy_adv": 1.0})          # streak broken
    assert not ev.update(150, {"copy_adv": 2.5})
    assert ev.update(175, {"copy_adv": 2.5})
    assert ev.fired_at == 175                             # completing eval, paper convention
    print("events: OK")


def test_forecaster_and_blind_gate():
    events = [6000.0, 6500.0, 7000.0, 8000.0]
    anchors = [e * 0.84 for e in events]
    f = calibrate_multiplicative(anchors, events)
    assert abs(f.median_multiplier - 1 / 0.84) < 1e-6 and f.note == ""
    assert f.covers(5040.0, 6000.0)
    frozen = f.freeze()
    assert MultiplicativeForecaster.load(frozen).median_multiplier == f.median_multiplier
    frozen["median_multiplier"] = 99.0
    try:
        MultiplicativeForecaster.load(frozen)
        raise AssertionError("tamper not detected")
    except ValueError:
        pass
    wide = calibrate_multiplicative([100.0, 100.0, 900.0], [1000.0, 1000.0, 1000.0])
    assert "WARNING" in wide.note                          # spread kill flagged
    gate = BlindGate(f)
    out = gate.score([{"t_anchor": 5040, "t_event": 6000}, {"t_anchor": None, "t_event": None}])
    assert out["coverage"] == "1/1" and out["false_alarms"] == "0/1"
    try:
        gate.score([])
        raise AssertionError("blind gate scored twice")
    except RuntimeError:
        pass
    print("forecaster + blind gate: OK")


def test_report_card_conventions():
    pos = [{"t_anchor": 800, "t_event": 1000}, {"t_anchor": 850, "t_event": 1100},
           {"t_anchor": 950, "t_event": 1200}, {"t_anchor": 1300, "t_event": 1250}]
    neg = [{"t_anchor": None}] * 2
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        card = report_card(pos, neg)
        assert any("UNCERTIFIABLE" in str(x.message) for x in w)   # < 5 negatives flagged
    assert card["miss_rate"] == 0.25                               # post-event anchor = miss
    assert card["median_lead"] > 0 and card["fa_certified"] is False
    print("report card: OK")


def test_live_monitor_end_to_end():
    class Phase:                                       # stand-in "model": phase drives probes
        def __init__(self):
            self.v = 0.0

        def parameters(self):
            yield torch.zeros(1)

    m = Phase()

    def probe_fn(model):
        return {"prevtok_early": model.v, "behavioral_ramp": model.v * 0.8,
                "copy_adv": max(0.0, (model.v - 0.5) * 10)}

    anchor = ComposedAnchor([Signal("prevtok_early", 0.10), Signal("behavioral_ramp", 0.10)])
    event = AbsoluteEvent("copy_adv", 2.0, sustain=2)
    fc = MultiplicativeForecaster(1.19, 1.11, 1.36, n_calibration=4)
    out = pathlib.Path(__file__).parent / "_tmp_monitor.jsonl"
    out.unlink(missing_ok=True)
    alarms = []
    mon = LiveMonitor(probe_fn, anchor, event, fc, cadence=10, jsonl_path=out,
                      on_alarm=lambda r: alarms.append(r["step"]))
    for step in range(0, 400, 10):
        m.v = step / 300.0                              # ramps 0 -> 1.3
        mon.step(m, step)
    s = mon.summary()
    assert s["anchor_at"] is not None and s["event_at"] is not None
    assert s["anchor_at"] < s["event_at"] and alarms == [s["anchor_at"]]
    recs = [json.loads(l) for l in open(out)]
    fired = [r for r in recs if r["anchor_fired"]]
    assert len(fired) == 1 and "forecast" in fired[0]
    lo, hi = fired[0]["forecast"]["interval"]
    assert lo < hi
    out.unlink()
    print(f"live monitor: OK (anchor {s['anchor_at']}, event {s['event_at']}, lead {s['lead']})")


if __name__ == "__main__":
    test_probe_math_ground_truth()
    test_anchor_composition_and_sustain()
    test_event_convention_matches_paper()
    test_forecaster_and_blind_gate()
    test_report_card_conventions()
    test_live_monitor_end_to_end()
    print("\nALL TESTS PASS (CPU only)")
