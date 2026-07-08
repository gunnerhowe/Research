"""Fixed low-entropy symbolic readouts (PLAN.md).

- emitted-token readout: the free-running sample stream itself (alphabet = task
  alphabet). This is the stochastic process the model defines; it is invariant to any
  recoding of the model's internals, so the identity null is exact by construction.
- quantized-belief readout (binary tasks): P(next = 1 | history) recorded during free
  running, quantized into Q equal bins. The model's belief/predictive-state process
  (Shai et al. framing), still low-entropy, sensitive to conditional-probability
  perturbations that leave the emitted marginal unchanged.
"""
from __future__ import annotations

import numpy as np


def emitted_readout(syms):
    """(B, T) int8 -> list of chains."""
    return [syms[i] for i in range(syms.shape[0])]


def belief_readout(beliefs, symbol=1, q=4):
    """(B, T, m) beliefs -> list of chains over alphabet q: equal-width bins of
    P(next = symbol | history)."""
    p = beliefs[:, :, symbol].astype(np.float64)
    bins = np.clip((p * q).astype(np.int8), 0, q - 1)
    return [bins[i] for i in range(bins.shape[0])]


def unigram_tv(sym_a, sym_b, m):
    """Total variation between unigram marginals of two multi-chain streams
    (= d̄_1, the 'same invariant measure at the readout' check)."""
    ca = np.bincount(np.concatenate([np.asarray(c).ravel() for c in sym_a]),
                     minlength=m)
    cb = np.bincount(np.concatenate([np.asarray(c).ravel() for c in sym_b]),
                     minlength=m)
    pa, pb = ca / ca.sum(), cb / cb.sum()
    return float(0.5 * np.abs(pa - pb).sum())
