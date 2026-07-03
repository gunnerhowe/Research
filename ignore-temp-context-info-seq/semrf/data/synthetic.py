"""Synthetic diagnostic tasks for probing long-range and time-sensitive
sequence modeling.

Each task exposes:
    .vocab_size            : int
    .seq_len               : int (length of a produced example)
    .sample_batch(bs, dev, rng) -> dict(input_ids, targets, loss_mask, ...)

Data are generated on the fly (effectively infinite), seeded by a numpy
Generator so runs are reproducible.  Targets follow the standard next-token
convention (input = seq[:-1], target = seq[1:]) and `loss_mask` marks only the
positions that the task actually scores (e.g. the answer tokens).

Tasks
-----
AssociativeRecall : store key->value pairs, then recall values for queried keys.
                    Tests content-based long-range retrieval.  Returns per-answer
                    retrieval `distances` for accuracy-vs-distance analysis.
TemporalRecency   : variables are (re)assigned values over time; a query must
                    return the *most recent* value.  Tests time/order sensitivity.
SelectiveCopy     : copy sparse data tokens (in order) out of a field of blanks.
                    Tests position-robust ordered routing over long spans.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def _to_tensors(input_ids, targets, loss_mask, device, extra=None) -> Dict:
    out = {
        "input_ids": torch.from_numpy(input_ids).to(device),
        "targets": torch.from_numpy(targets).to(device),
        "loss_mask": torch.from_numpy(loss_mask).to(device),
    }
    if extra:
        out.update(extra)
    return out


# --------------------------------------------------------------------------- #
class AssociativeRecall:
    name = "assoc_recall"

    def __init__(self, key_vocab=64, value_vocab=64, n_pairs=16, n_queries=8, gap=0):
        """gap: filler (BLANK) tokens inserted between pairs and queries.
        int -> fixed gap; (lo, hi) -> sampled uniformly per sequence, so BLANK
        stays in-distribution during training while eval can grow the span."""
        assert n_pairs <= key_vocab
        self.key_vocab = key_vocab
        self.value_vocab = value_vocab
        self.n_pairs = n_pairs
        self.n_queries = n_queries
        self.gap = tuple(gap) if isinstance(gap, (tuple, list)) else (int(gap), int(gap))
        self.BLANK = 0
        self.key_offset = 1
        self.val_offset = 1 + key_vocab
        self.vocab_size = 1 + key_vocab + value_vocab
        self.max_gap = self.gap[1]
        self.seq_len = 2 * (n_pairs + n_queries) + self.max_gap - 1  # after shift

    def sample_batch(self, batch_size, device, rng: np.random.Generator):
        B, D, Q = batch_size, self.n_pairs, self.n_queries
        L = 2 * (D + Q) + self.max_gap
        seqs = np.full((B, L), self.BLANK, dtype=np.int64)
        ans_pos = np.zeros((B, Q), dtype=np.int64)
        def_pos = np.zeros((B, Q), dtype=np.int64)

        for b in range(B):
            keys = rng.choice(self.key_vocab, size=D, replace=False) + self.key_offset
            vals = rng.integers(0, self.value_vocab, size=D) + self.val_offset
            order = rng.permutation(D)
            g = int(rng.integers(self.gap[0], self.gap[1] + 1))
            # left-pad with the unused blanks so every sequence fills L exactly
            seq = [self.BLANK] * (self.max_gap - g)
            keypos = {}
            for idx in order:
                keypos[int(keys[idx])] = len(seq)
                seq.append(int(keys[idx]))
                seq.append(int(vals[idx]))
            seq.extend([self.BLANK] * g)       # span padding (grows retrieval distance)
            qsel = rng.integers(0, D, size=Q)
            for qi, s in enumerate(qsel):
                ans_pos[b, qi] = len(seq)          # position of query key; next token = answer
                def_pos[b, qi] = keypos[int(keys[s])]
                seq.append(int(keys[s]))
                seq.append(int(vals[s]))
            seqs[b] = np.array(seq, dtype=np.int64)

        input_ids = seqs[:, :-1]
        targets = seqs[:, 1:]
        loss_mask = np.zeros_like(input_ids)
        rows = np.arange(B)[:, None]
        loss_mask[rows, ans_pos] = 1
        distances = (ans_pos - def_pos).astype(np.int64)   # (B, Q), positive

        extra = {
            "ans_pos": torch.from_numpy(ans_pos).to(device),
            "distances": torch.from_numpy(distances).to(device),
        }
        return _to_tensors(input_ids, targets, loss_mask, device, extra)


# --------------------------------------------------------------------------- #
class TemporalRecency:
    name = "temporal_recency"

    def __init__(self, n_vars=20, n_vals=64, seq_len=256, p_query=0.25):
        self.n_vars = n_vars
        self.n_vals = n_vals
        self._len = seq_len
        self.p_query = p_query
        self.PAD = 0
        self.QUERY = 1
        self.var_offset = 2
        self.val_offset = 2 + n_vars
        self.vocab_size = 2 + n_vars + n_vals
        self.seq_len = seq_len - 1

    def sample_batch(self, batch_size, device, rng: np.random.Generator):
        B, T = batch_size, self._len
        seqs = np.zeros((B, T), dtype=np.int64)
        ans_list = []          # (b, pos, dist) records for optional analysis

        for b in range(B):
            latest = {}        # var_id -> (val_token, val_position)
            seq = []
            answers = []
            while len(seq) < T:
                do_query = latest and (rng.random() < self.p_query) and (len(seq) + 3 <= T)
                if do_query:
                    var = int(rng.choice(list(latest.keys())))
                    val_token, val_pos = latest[var]
                    var_pos = len(seq) + 1
                    seq.append(self.QUERY)
                    seq.append(self.var_offset + var)
                    seq.append(val_token)
                    answers.append((var_pos, var_pos - val_pos))
                elif len(seq) + 2 <= T:
                    var = int(rng.integers(0, self.n_vars))
                    val_token = int(self.val_offset + rng.integers(0, self.n_vals))
                    seq.append(self.var_offset + var)
                    seq.append(val_token)
                    latest[var] = (val_token, len(seq) - 1)
                else:
                    break
            arr = np.full(T, self.PAD, dtype=np.int64)
            arr[: len(seq)] = np.array(seq, dtype=np.int64)
            seqs[b] = arr
            ans_list.append(answers)

        input_ids = seqs[:, :-1]
        targets = seqs[:, 1:]
        loss_mask = np.zeros_like(input_ids)
        dist_full = np.zeros_like(input_ids)
        for b, answers in enumerate(ans_list):
            for var_pos, dist in answers:
                if var_pos < input_ids.shape[1]:
                    loss_mask[b, var_pos] = 1
                    dist_full[b, var_pos] = dist

        extra = {"distances_full": torch.from_numpy(dist_full).to(device)}
        return _to_tensors(input_ids, targets, loss_mask, device, extra)


# --------------------------------------------------------------------------- #
class SelectiveCopy:
    name = "selective_copy"

    def __init__(self, n_data=16, context_len=128, data_vocab=16):
        assert n_data <= context_len
        self.n_data = n_data
        self.context_len = context_len
        self.data_vocab = data_vocab
        self.BLANK = 0
        self.GO = 1
        self.data_offset = 2
        self.vocab_size = 2 + data_vocab
        self.seq_len = context_len + 1 + n_data - 1

    def sample_batch(self, batch_size, device, rng: np.random.Generator):
        B, C, N = batch_size, self.context_len, self.n_data
        L = C + 1 + N
        seqs = np.full((B, L), self.BLANK, dtype=np.int64)
        for b in range(B):
            data = rng.integers(0, self.data_vocab, size=N) + self.data_offset
            pos = np.sort(rng.choice(C, size=N, replace=False))
            seqs[b, pos] = data
            seqs[b, C] = self.GO
            seqs[b, C + 1 : C + 1 + N] = data

        input_ids = seqs[:, :-1]
        targets = seqs[:, 1:]
        loss_mask = np.zeros_like(input_ids)
        loss_mask[:, C : C + N] = 1          # predict the N data tokens in order
        return _to_tensors(input_ids, targets, loss_mask, device)


# --------------------------------------------------------------------------- #
SYNTHETIC_TASKS = {
    "assoc_recall": AssociativeRecall,
    "temporal_recency": TemporalRecency,
    "selective_copy": SelectiveCopy,
}


def build_task(name: str, **kwargs):
    if name not in SYNTHETIC_TASKS:
        raise ValueError(f"unknown task {name!r}; choose from {list(SYNTHETIC_TASKS)}")
    return SYNTHETIC_TASKS[name](**kwargs)
