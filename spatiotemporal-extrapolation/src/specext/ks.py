"""Kuramoto-Sivashinsky solver: u_t = -u u_x - u_xx - u_xxxx, periodic on [0, L).

ETDRK4 time stepping (Kassam & Trefethen 2005), rfft pseudospectral formulation with
2/3-rule dealiasing.

`ks_trajectory` is vendored VERBATIM (minus the `speed` parameter, fixed to 1) from
E:/GitHub/Research/ornstein-dist src/ornstein/ks.py (Howe 2026, "Beyond the Invariant
Measure"), where it was validated; tests/test_integrator.py pins the regression.
`ks_stream_batch` generalizes it: arbitrary (L, N), several seeds integrated as a
batch, chunked streaming output, and an odd-parity ("Dirichlet-type", u = u_xx = 0
at both ends) variant via odd extension on a doubled periodic domain.
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


def _init_field(L, N, rng):
    """Random smooth mean-zero initial condition (3 long-wave cosines)."""
    x = L * np.arange(N) / N
    u = np.zeros(N)
    for j in range(1, 4):
        u += rng.normal(0, 0.4) * np.cos(2 * np.pi * j * x / L + rng.uniform(0, 2 * np.pi))
    return u


def ks_trajectory(n_samples, dt_sample=1.0, dt=0.25, L=22.0, N=64, seed=0,
                  transient=1000.0):
    """Sampled KS trajectory, shape (n_samples, N) real space (vendored reference)."""
    steps_per_sample = int(round(dt_sample / dt))
    if abs(steps_per_sample * dt - dt_sample) > 1e-12:
        raise ValueError("dt must divide dt_sample")
    rng = np.random.default_rng(seed)
    u = _init_field(L, N, rng)
    v = np.fft.rfft(u)
    k = 2 * np.pi * np.arange(N // 2 + 1) / L
    lin = k ** 2 - k ** 4
    E, E2, Q, f1, f2, f3 = _etdrk4_coeffs(lin, dt)
    dealias = np.ones(N // 2 + 1)
    dealias[int(np.floor(N / 3)):] = 0.0  # 2/3 rule
    gcoef = (-0.5j * k) * dealias

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


def ks_stream_batch(L, N, n_samples, seeds, dt=0.25, dt_sample=0.5, transient=500.0,
                    bc="periodic", chunk_samples=4096):
    """Batched streaming KS integration.

    Yields chunks of sampled fields, shape (n_chunk, n_seeds, N) float64, real space.
    All seeds advance in lockstep (vectorized over the batch axis), same ETDRK4 core
    and initial-condition family as `ks_trajectory`.

    bc="odd": integrates the odd-parity invariant subspace of the periodic problem on
    a domain of length 2L with 2N points (odd extension); the yielded field is the
    first half [0, L], i.e. a Dirichlet-type solution with u = u_xx = 0 at x = 0, L.
    Parity is re-projected every step (rfft of an odd real field is purely
    imaginary), so roundoff cannot leak into the even subspace.
    """
    steps_per_sample = int(round(dt_sample / dt))
    if abs(steps_per_sample * dt - dt_sample) > 1e-12:
        raise ValueError("dt must divide dt_sample")
    odd = bc == "odd"
    if bc not in ("periodic", "odd"):
        raise ValueError(f"unknown bc {bc!r}")
    Ld, Nd = (2 * L, 2 * N) if odd else (L, N)
    S = len(seeds)

    u0 = np.empty((S, Nd))
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        if odd:
            # random sine series: odd about x=0 and x=L by construction
            x = Ld * np.arange(Nd) / Nd
            u = np.zeros(Nd)
            for j in range(1, 4):
                u += rng.normal(0, 0.4) * np.sin(2 * np.pi * j * x / Ld)
        else:
            u = _init_field(Ld, Nd, rng)
        u0[i] = u - u.mean() if not odd else u
    v = np.fft.rfft(u0, axis=-1)
    v[:, 0] = 0.0  # mean-zero sector, conserved

    k = 2 * np.pi * np.arange(Nd // 2 + 1) / Ld
    lin = k ** 2 - k ** 4
    E, E2, Q, f1, f2, f3 = _etdrk4_coeffs(lin, dt)
    dealias = np.ones(Nd // 2 + 1)
    dealias[int(np.floor(Nd / 3)):] = 0.0
    gcoef = (-0.5j * k) * dealias

    def g(vv):
        return gcoef * np.fft.rfft(np.fft.irfft(vv, n=Nd, axis=-1) ** 2, axis=-1)

    n_trans = int(round(transient / dt))
    buf = np.empty((chunk_samples, S, N))
    idx = 0
    count = 0
    emitted = 0
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
        if odd:
            v = 1j * v.imag  # parity projection: odd real field <-> imaginary rfft
        if step >= n_trans:
            count += 1
            if count == steps_per_sample:
                field = np.fft.irfft(v, n=Nd, axis=-1)
                buf[idx] = field[:, :N]
                idx += 1
                count = 0
                emitted += 1
                if idx == chunk_samples:
                    yield buf[:idx].copy()
                    idx = 0
    if idx:
        yield buf[:idx].copy()
    assert emitted == n_samples
