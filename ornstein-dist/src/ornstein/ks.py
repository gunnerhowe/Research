"""Kuramoto-Sivashinsky solver: u_t = -u u_x - u_xx - u_xxxx, periodic on [0, L].

ETDRK4 time stepping (Kassam & Trefethen 2005), rfft formulation with 2/3-rule
dealiasing. `speed` multiplies the whole right-hand side: identical attractor and
invariant measure, rescaled clock (the same-measure adversarial construction as for
Lorenz). L = 22 is the standard modestly chaotic regime.
"""
from __future__ import annotations

import numpy as np


def _etdrk4_coeffs(lin, h, M=32):
    LR = h * lin[:, None] + np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)[None, :]
    E = np.exp(h * lin)
    E2 = np.exp(h * lin / 2.0)
    Q = h * np.real(np.mean((np.exp(LR / 2.0) - 1) / LR, axis=1))
    f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3,
                             axis=1))
    f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
    f3 = h * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3,
                             axis=1))
    return E, E2, Q, f1, f2, f3


def ks_trajectory(n_samples, dt_sample=1.0, dt=0.25, L=22.0, N=64, speed=1.0,
                  seed=0, transient=1000.0):
    """Sampled KS trajectory, shape (n_samples, N) real space, sampled every dt_sample."""
    steps_per_sample = int(round(dt_sample / dt))
    if abs(steps_per_sample * dt - dt_sample) > 1e-12:
        raise ValueError("dt must divide dt_sample")
    rng = np.random.default_rng(seed)
    x = L * np.arange(N) / N
    u = np.zeros(N)
    for j in range(1, 4):
        u += rng.normal(0, 0.4) * np.cos(2 * np.pi * j * x / L + rng.uniform(0, 2 * np.pi))
    v = np.fft.rfft(u)
    k = 2 * np.pi * np.arange(N // 2 + 1) / L
    lin = speed * (k ** 2 - k ** 4)
    E, E2, Q, f1, f2, f3 = _etdrk4_coeffs(lin, dt)
    dealias = np.ones(N // 2 + 1)
    dealias[int(np.floor(N / 3)):] = 0.0  # 2/3 rule
    gcoef = speed * (-0.5j * k) * dealias

    def g(vv):
        return gcoef * np.fft.rfft(np.fft.irfft(vv, n=N) ** 2)

    n_trans = int(round(transient / dt))
    out = np.empty((n_samples, N))
    idx = 0
    count = 0
    total = n_trans + n_samples * steps_per_sample
    for step in range(total):
        Nv = g(v)
        a = E2 * v + Q * Nv
        Na = g(a)
        b = E2 * v + Q * Na
        Nb = g(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = g(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if step >= n_trans:
            count += 1
            if count == steps_per_sample:
                out[idx] = np.fft.irfft(v, n=N)
                idx += 1
                count = 0
    return out
