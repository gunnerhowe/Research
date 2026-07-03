"""Adversarial surrogate time series: IAAFT, phase-randomized, time reversal.

IAAFT (Schreiber & Schmitz 1996): exactly preserves the marginal distribution,
approximately preserves the power spectrum, destroys higher-order temporal structure.
"""
from __future__ import annotations

import numpy as np


def iaaft(x, n_iter=200, seed=0, tol=1e-8):
    """Iterative Amplitude-Adjusted Fourier Transform surrogate of a 1D series."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    rng = np.random.default_rng(seed)
    sorted_x = np.sort(x)
    target_amp = np.abs(np.fft.rfft(x))
    s = rng.permutation(x)
    prev_err = np.inf
    for _ in range(n_iter):
        spec = np.fft.rfft(s)
        phases = np.angle(spec)
        s = np.fft.irfft(target_amp * np.exp(1j * phases), n=n)
        ranks = np.argsort(np.argsort(s))
        s = sorted_x[ranks]
        err = np.mean((np.abs(np.fft.rfft(s)) - target_amp) ** 2)
        if abs(prev_err - err) < tol * max(err, 1e-30):
            break
        prev_err = err
    return s


def phase_randomized(x, seed=0):
    """FT surrogate: exact spectrum, random phases (marginal only approximately kept)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(x)
    phases = rng.uniform(0, 2 * np.pi, len(spec))
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    return np.fft.irfft(np.abs(spec) * np.exp(1j * phases), n=n)


def time_reverse(a):
    """Time-reversed copy (works on 1D series or (N,d) trajectories)."""
    return np.ascontiguousarray(np.asarray(a)[::-1])
