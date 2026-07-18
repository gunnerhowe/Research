"""Calibrated timing forecasts and the freeze discipline.

The paper's central quantitative finding: the valid anchor fires at an approximately
scale-invariant FRACTION of time-to-emergence (0.843 across 80 runs spanning
architectures, data mixes, languages, learning rates and batch sizes; blind-validated
5/5 at a never-seen configuration). The deployable forecaster is therefore
MULTIPLICATIVE: t_event ~= t_anchor / fraction, with an interval from the calibration
envelope. Fixed-step offsets are timescale-local and break under any shift that moves
the emergence timescale — measured, not conjectured.

Freeze discipline: a forecaster is calibrated, then FROZEN (serialized with a content
hash) before any blind evaluation. `BlindGate` enforces one-shot scoring.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class MultiplicativeForecaster:
    """t_event forecast = t_anchor * multiplier, with an envelope interval."""
    median_multiplier: float
    lo_multiplier: float
    hi_multiplier: float
    n_calibration: int = 0
    note: str = ""

    def forecast(self, t_anchor: float) -> dict:
        return {
            "point": t_anchor * self.median_multiplier,
            "interval": (t_anchor * self.lo_multiplier, t_anchor * self.hi_multiplier),
        }

    def covers(self, t_anchor: float, t_event: float) -> bool:
        lo, hi = self.forecast(t_anchor)["interval"]
        return lo <= t_event <= hi

    def freeze(self) -> dict:
        """Serializable frozen artifact with a content hash. Commit this before running
        anything you intend to call blind."""
        payload = asdict(self)
        payload["sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return payload

    @staticmethod
    def load(frozen: dict) -> "MultiplicativeForecaster":
        payload = {k: v for k, v in frozen.items() if k != "sha256"}
        f = MultiplicativeForecaster(**payload)
        if frozen.get("sha256") != f.freeze()["sha256"]:
            raise ValueError("frozen forecaster hash mismatch — artifact was modified")
        return f


def calibrate_multiplicative(anchor_times: list[float], event_times: list[float],
                             spread_warn: float = 2.0) -> MultiplicativeForecaster:
    """Fit the multiplicative rule from calibration runs (anchor and event times paired
    per run). Envelope = full calibration range (conservative, as in the paper's blind
    gates). Raises on invalid pairs; warns loudly when the spread suggests no stable
    law (the pre-registered kill condition in our own work)."""
    if len(anchor_times) != len(event_times) or len(anchor_times) < 3:
        raise ValueError("need >= 3 paired (anchor, event) calibration runs")
    mults = []
    for a, e in zip(anchor_times, event_times):
        if not (0 < a < e):
            raise ValueError(f"invalid pair anchor={a}, event={e} (need 0 < anchor < event)")
        mults.append(e / a)
    mults.sort()
    n = len(mults)
    median = mults[n // 2] if n % 2 else 0.5 * (mults[n // 2 - 1] + mults[n // 2])
    spread = mults[-1] / mults[0]
    note = ""
    if spread > spread_warn:
        note = (f"WARNING: calibration spread {spread:.2f}x exceeds {spread_warn}x — "
                "no stable fraction law in this substrate; treat forecasts as unreliable")
    return MultiplicativeForecaster(median_multiplier=median, lo_multiplier=mults[0],
                                    hi_multiplier=mults[-1], n_calibration=n, note=note)


@dataclass
class AdditiveForecaster:
    """Fixed-offset baseline (t_event = t_anchor + offset ± q). Included for honest
    comparison; in our measurements it is TIMESCALE-LOCAL — it transfers only across
    shifts that preserve the emergence timescale. Prefer the multiplicative rule."""
    offset: float
    q: float

    def forecast(self, t_anchor: float) -> dict:
        p = t_anchor + self.offset
        return {"point": p, "interval": (p - self.q, p + self.q)}

    def covers(self, t_anchor: float, t_event: float) -> bool:
        lo, hi = self.forecast(t_anchor)["interval"]
        return lo <= t_event <= hi


class BlindGate:
    """One-shot evaluation enforcement: score a frozen forecaster against blind runs
    exactly once. A second scoring attempt raises — re-scoring after peeking is how
    blind validation quietly dies."""

    def __init__(self, forecaster, name: str = "blind-gate"):
        self.forecaster = forecaster
        self.name = name
        self._scored: Optional[dict] = None

    def score(self, runs: list[dict]) -> dict:
        """runs: [{'t_anchor': float|None, 't_event': float|None}, ...]
        (t_event None = non-eventing run: any anchor there is a false alarm)."""
        if self._scored is not None:
            raise RuntimeError(f"{self.name} already scored — blind gates run ONCE")
        cover = miss = fa = pos = neg = 0
        for r in runs:
            ta, te = r.get("t_anchor"), r.get("t_event")
            if te is None:
                neg += 1
                fa += int(ta is not None)
                continue
            pos += 1
            if ta is None:
                miss += 1
                continue
            cover += int(self.forecaster.covers(ta, te))
        self._scored = {"name": self.name, "n_pos": pos, "n_neg": neg,
                        "coverage": f"{cover}/{pos}", "misses": miss,
                        "false_alarms": f"{fa}/{neg}" if neg else "0/0 (no negatives!)"}
        return dict(self._scored)
