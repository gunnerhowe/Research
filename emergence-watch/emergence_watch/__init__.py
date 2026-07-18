"""emergence-watch: training telemetry for capability emergence.

Circuit-precursor alarms, calibrated timing forecasts, and certified false-alarm rates
— the methodology of "Capability Emergence Can Be Forecast" as a library.

Honest scope (read this before deploying): this is a NAMED-capability tripwire, not an
unknown-capability radar. You choose the capability, you supply (or reuse) its
mechanistic precursor probes, you calibrate on your own fleet, and you certify false
alarms on manufactured capability-blocked negatives. Every default in this package
encodes a failure mode we paid for; the paper documents each one.
"""
from .probes import (repeated_probe_batch, copy_advantage, prefix_matching_score,
                     prev_token_score, early_layer_prev_token_score, induction_probe,
                     capture_attentions, eager_attention)
from .anchors import Signal, ComposedAnchor, paper_default_anchor
from .events import AbsoluteEvent, paper_induction_event
from .forecaster import (MultiplicativeForecaster, AdditiveForecaster,
                         calibrate_multiplicative, BlindGate)
from .scoring import report_card, spearman, MIN_NEGATIVES
from .callback import LiveMonitor, EmergenceWatchCallback, make_induction_probe_fn

__version__ = "0.1.0.dev0"
__all__ = [
    "repeated_probe_batch", "copy_advantage", "prefix_matching_score",
    "prev_token_score", "early_layer_prev_token_score", "induction_probe",
    "capture_attentions", "eager_attention",
    "Signal", "ComposedAnchor", "paper_default_anchor",
    "AbsoluteEvent", "paper_induction_event",
    "MultiplicativeForecaster", "AdditiveForecaster", "calibrate_multiplicative",
    "BlindGate", "report_card", "spearman", "MIN_NEGATIVES",
    "LiveMonitor", "EmergenceWatchCallback", "make_induction_probe_fn",
]
