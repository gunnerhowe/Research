"""SemRF: Semantic Reference Frames for sequence models.

A small research library implementing a decoder-only transformer with a
*pluggable* positional / attention-bias mechanism, so that a single backbone can
be instantiated with any of:

    NoPE, sinusoidal, learned-absolute, RoPE, ALiBi, T5 relative bias, SemRF

SemRF (ours) represents context and time *relative to a small set of learned
semantic anchor vectors* rather than absolute positions.
"""

from .config import ModelConfig, TrainConfig
from .model import TransformerLM
from .positions import build_position_module, POSITION_TYPES

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "TransformerLM",
    "build_position_module",
    "POSITION_TYPES",
]
