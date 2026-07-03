"""Evaluation: synthetic accuracy, accuracy-vs-distance, and char-LM bpc,
including length-extrapolation protocols.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch

from .utils import masked_token_accuracy, masked_sequence_accuracy


@torch.no_grad()
def evaluate_synthetic(model, task, n_batches, batch_size, device, rng, amp=True):
    model.eval()
    tok_c = tok_n = seq_c = seq_n = 0.0
    for _ in range(n_batches):
        b = task.sample_batch(batch_size, device, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits, _ = model(b["input_ids"])
        logits = logits.float()
        c, n = masked_token_accuracy(logits, b["targets"], b["loss_mask"])
        s, m = masked_sequence_accuracy(logits, b["targets"], b["loss_mask"])
        tok_c += c; tok_n += n; seq_c += s; seq_n += m
    return {
        "token_acc": tok_c / max(1.0, tok_n),
        "seq_acc": seq_c / max(1.0, seq_n),
        "n_scored": tok_n,
    }


@torch.no_grad()
def collect_distance_correct(model, task, n_batches, batch_size, device, rng, amp=True):
    """Return (distances, correct) flat arrays over scored answer positions."""
    model.eval()
    all_d: List[np.ndarray] = []
    all_c: List[np.ndarray] = []
    for _ in range(n_batches):
        b = task.sample_batch(batch_size, device, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits, _ = model(b["input_ids"])
        pred = logits.float().argmax(-1)                      # (B, T)

        if "ans_pos" in b:                                    # associative recall
            ans_pos = b["ans_pos"]                            # (B, Q)
            dist = b["distances"]                             # (B, Q)
            rows = torch.arange(ans_pos.shape[0], device=device)[:, None]
            p = pred[rows, ans_pos]                           # (B, Q)
            t = b["targets"][rows, ans_pos]
            correct = (p == t)
            all_d.append(dist.reshape(-1).cpu().numpy())
            all_c.append(correct.reshape(-1).cpu().numpy())
        elif "distances_full" in b:                           # temporal recency
            m = b["loss_mask"].bool()
            correct = (pred == b["targets"]) & m
            all_d.append(b["distances_full"][m].cpu().numpy())
            all_c.append(correct[m].cpu().numpy())
    if not all_d:
        return np.array([]), np.array([])
    return np.concatenate(all_d), np.concatenate(all_c).astype(np.float64)


def bucketize_accuracy(distances, correct, n_buckets=10):
    """Bucket accuracy by distance quantiles. Returns list of (center, acc, n)."""
    if len(distances) == 0:
        return []
    order = np.argsort(distances)
    distances = distances[order]
    correct = correct[order]
    edges = np.linspace(0, len(distances), n_buckets + 1).astype(int)
    out = []
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        d = distances[lo:hi]
        c = correct[lo:hi]
        out.append((float(d.mean()), float(c.mean()), int(hi - lo)))
    return out


@torch.no_grad()
def evaluate_bpc(model, data, split, seq_len, batch_size, device, max_tokens=None, amp=True):
    """Bits-per-character (log2) over non-overlapping windows."""
    model.eval()
    total_nll = 0.0
    total_tok = 0
    ln2 = math.log(2.0)
    for x, y in data.iter_eval(split, seq_len, batch_size, max_tokens=max_tokens):
        x = x.to(device); y = y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits, _ = model(x)
        logits = logits.float()
        nll = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        )
        total_nll += nll.item()
        total_tok += y.numel()
    return (total_nll / max(1, total_tok)) / ln2


@torch.no_grad()
def extrapolation_bpc(model, data, split, seq_lens, batch_size, device, max_tokens=None):
    return {int(L): evaluate_bpc(model, data, split, L, batch_size, device, max_tokens=max_tokens)
            for L in seq_lens}
