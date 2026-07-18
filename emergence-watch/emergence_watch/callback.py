"""Live emergence telemetry: probes, anchors, events and forecasts DURING training.

Core is framework-agnostic (`LiveMonitor.step(model, step)` from any training loop);
`EmergenceWatchCallback` wraps it as a HuggingFace `TrainerCallback`.

Overhead model: one small probe forward (default 64 x 128 tokens) every `cadence`
steps, run under no_grad with eager attention. On the paper's fleets this was <2% of
step time at cadence 25; at production scale choose cadence so the probe forward is
amortized (e.g., every 100-500 steps).

What it emits per probe evaluation (JSONL + optional user hook + optional wandb):
  {"step", <probe values>, "anchor_fired", "event_fired", "forecast": {point, interval}}

The forecast appears once the anchor fires, using a calibrated multiplicative
forecaster if provided — and is None otherwise, because shipping uncalibrated
constants is how monitors lie (calibration is regime-local; measure your own fleet).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import torch

from .anchors import ComposedAnchor
from .events import AbsoluteEvent
from .forecaster import MultiplicativeForecaster
from .probes import induction_probe, repeated_probe_batch


class LiveMonitor:
    """Streamed emergence watching for one training run.

    Parameters
    ----------
    probe_fn : callable(model) -> dict of probe values. Use `make_induction_probe_fn`
        for the paper's induction instruments, or supply your own capability's probes.
    anchor : ComposedAnchor (compose signals; bare single-signal anchors false-alarm
        in regimes where their circuit pays for the task — measured, not hypothetical).
    event : AbsoluteEvent or None (behavioral arrival criterion, for scoring/logging).
    forecaster : frozen MultiplicativeForecaster or None.
    cadence : probe every N training steps.
    jsonl_path : append-mode output file (one dict per probe evaluation).
    on_alarm / on_event : optional callables(record_dict) fired once each.
    """

    def __init__(self, probe_fn: Callable, anchor: ComposedAnchor,
                 event: Optional[AbsoluteEvent] = None,
                 forecaster: Optional[MultiplicativeForecaster] = None,
                 cadence: int = 100, jsonl_path: str | Path | None = None,
                 on_alarm: Optional[Callable] = None,
                 on_event: Optional[Callable] = None,
                 wandb_run=None):
        self.probe_fn = probe_fn
        self.anchor = anchor
        self.event = event
        self.forecaster = forecaster
        self.cadence = cadence
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.on_alarm = on_alarm
        self.on_event = on_event
        self.wandb_run = wandb_run
        self.history: list[dict] = []

    # ---- core ----
    @torch.no_grad()
    def step(self, model, step: int, force: bool = False) -> Optional[dict]:
        """Call every training step; probes run on the cadence (or when forced)."""
        if not force and step % self.cadence != 0:
            return None
        t0 = time.time()
        values = self.probe_fn(model)
        rec: dict = {"step": step, **values,
                     "probe_seconds": round(time.time() - t0, 4)}
        fired = self.anchor.update(step, values)
        rec["anchor_fired"] = fired
        rec["anchor_at"] = self.anchor.fired_at
        if fired and self.forecaster is not None:
            rec["forecast"] = self.forecaster.forecast(step)
        if self.event is not None:
            ev = self.event.update(step, values)
            rec["event_fired"] = ev
            rec["event_at"] = self.event.fired_at
        self._emit(rec)
        if fired and self.on_alarm:
            self.on_alarm(rec)
        if self.event is not None and rec.get("event_fired") and self.on_event:
            self.on_event(rec)
        return rec

    def _emit(self, rec: dict):
        self.history.append(rec)
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.log({f"emergence/{k}": v for k, v in rec.items()
                                if isinstance(v, (int, float))}, step=rec["step"])

    # ---- summary ----
    def summary(self) -> dict:
        return {
            "anchor_at": self.anchor.fired_at,
            "event_at": self.event.fired_at if self.event else None,
            "lead": (self.event.fired_at - self.anchor.fired_at)
            if (self.event and self.event.fired_at is not None
                and self.anchor.fired_at is not None) else None,
            "n_probes": len(self.history),
        }


def make_induction_probe_fn(vocab_size: int, batch: int = 64, half_len: int = 64,
                            seed: int = 1234, device: str | torch.device = "cpu",
                            behavioral_fn: Optional[Callable] = None,
                            capture: Optional[Callable] = None) -> Callable:
    """Probe function for the induction capability: the paper's three instruments on a
    fixed repeated batch, plus an optional in-distribution behavioral ramp (recommended
    — it is the second gate of the certified composed anchor). `behavioral_fn(model)`
    should return a float; its value is exposed as 'behavioral_ramp'."""
    ids = repeated_probe_batch(vocab_size, batch, half_len, seed, device)

    def probe(model) -> dict:
        vals = induction_probe(model, ids.to(next(model.parameters()).device),
                               half_len, capture)
        if behavioral_fn is not None:
            vals["behavioral_ramp"] = float(behavioral_fn(model))
        return vals

    return probe


class EmergenceWatchCallback:
    """HuggingFace `TrainerCallback` wrapper around LiveMonitor.

    Usage:
        monitor = LiveMonitor(probe_fn, anchor, event, forecaster, cadence=200,
                              jsonl_path="emergence.jsonl")
        trainer = Trainer(..., callbacks=[EmergenceWatchCallback(monitor)])

    Implemented duck-typed (on_step_end signature) so `transformers` stays an optional
    dependency; Trainer only needs the methods it calls.
    """

    def __init__(self, monitor: LiveMonitor):
        self.monitor = monitor

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None and state.global_step is not None:
            self.monitor.step(model, state.global_step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        s = self.monitor.summary()
        print(f"[emergence-watch] anchor_at={s['anchor_at']} event_at={s['event_at']} "
              f"lead={s['lead']}")
        return control
