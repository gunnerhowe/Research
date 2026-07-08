"""Repo-relative paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGS = ROOT / "paper" / "figs"

CHECKPOINTS = {
    "M1": MODELS / "cot-baseline_best.pt",
    "M2": MODELS / "coconut_best.pt",
    "M3": MODELS / "pause-curriculum_best.pt",
    "M4": MODELS / "pause-multipass_best.pt",
}

FEEDBACK_MODE = {
    "M1": "cot",
    "M2": "continuous",
    "M3": "pause",
    "M4": "pause",  # multipass is numerically identical to single-pass at inference
}
