"""Utilities: seeding, logging, metrics, and result IO."""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False):
    """Seed python / numpy / torch (incl. CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def count_params(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(
        p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)
    )


def human(n: int) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n}"
        n /= 1000.0
    return f"{n:.1f}T"


def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0


def get_amp_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


@torch.no_grad()
def masked_token_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    """Fraction of masked positions whose argmax prediction is correct.

    logits: (B, T, V), targets: (B, T), mask: (B, T) bool/float of scored positions.
    Returns (num_correct, num_scored) as python floats so callers can aggregate.
    """
    pred = logits.argmax(dim=-1)
    m = mask.bool()
    correct = ((pred == targets) & m).sum().item()
    total = m.sum().item()
    return float(correct), float(total)


@torch.no_grad()
def masked_sequence_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    """Fraction of *sequences* for which every masked position is correct."""
    pred = logits.argmax(dim=-1)
    m = mask.bool()
    correct_tok = (pred == targets) | (~m)          # unmasked count as correct
    seq_ok = correct_tok.all(dim=1)                 # (B,)
    # sequences with at least one scored position
    has_scored = m.any(dim=1)
    n_ok = (seq_ok & has_scored).sum().item()
    n_seq = has_scored.sum().item()
    return float(n_ok), float(n_seq)
