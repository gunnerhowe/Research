"""Nikolaevskiy equation: extensive chaos with a marginal k=0 (Goldstone/
conservation) mode -> anomalously long correlation length and SLOW finite-size
convergence. This is the "when you need the flow" system (contrast with
Kuramoto-Sivashinsky, whose short correlation length converges by L~4x the base).

We use the mean-conserving Burgers nonlinearity form
    u_t = -u u_x + L(-i d_x) u,
with the Nikolaevskiy linear symbol (Tribelsky-Tsuboi soft-mode turbulence)
    sigma(k) = k^2 [ r - (1 - k^2)^2 ] = (r-1) k^2 + 2 k^4 - k^6.
sigma(0) = 0 exactly (the marginal mode); a band around k=1 is unstable for r>0;
the -k^6 tail makes the high wavenumbers strongly damped (stiff -> ETDRK4).
The -u u_x nonlinearity conserves the spatial mean, so the k=0 sector stays
marginal and is excluded from analysis exactly as for KS.

ETDRK4 (Kassam & Trefethen 2005); the coefficient helper is shared with the
vendored KS integrator (specext.ks). `r` and `dt` are the tunable knobs; the
long-correlation regime is r just above the onset of soft-mode turbulence.
"""
from __future__ import annotations

import numpy as np

from .ks import _etdrk4_coeffs


def nik_symbol(k, r):
    """Nikolaevskiy linear growth rate sigma(k) = k^2 [ r - (1-k^2)^2 ]."""
    return (k ** 2) * (r - (1.0 - k ** 2) ** 2)


def nik_stream_batch(L, N, n_samples, seeds, r=0.25, dt=0.1, dt_sample=0.5,
                     transient=2000.0, chunk_samples=4096):
    """Batched streaming Nikolaevskiy integration; yields chunks of sampled fields
    (n_chunk, n_seeds, N), real space, mean-zero sector. Same ETDRK4 core and
    interface as specext.ks.ks_stream_batch (periodic only)."""
    steps_per_sample = int(round(dt_sample / dt))
    if abs(steps_per_sample * dt - dt_sample) > 1e-12:
        raise ValueError("dt must divide dt_sample")
    S = len(seeds)
    u0 = np.empty((S, N))
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        x = L * np.arange(N) / N
        u = np.zeros(N)
        # seed the unstable band (k ~ 1) plus a little long-wave content
        for _ in range(6):
            m = rng.integers(1, max(2, int(1.5 * L / (2 * np.pi))))
            u += rng.normal(0, 0.3) * np.cos(2 * np.pi * m * x / L + rng.uniform(0, 2 * np.pi))
        u0[i] = u - u.mean()
    v = np.fft.rfft(u0, axis=-1)
    v[:, 0] = 0.0
    k = 2 * np.pi * np.arange(N // 2 + 1) / L
    lin = nik_symbol(k, r)
    E, E2, Q, f1, f2, f3 = _etdrk4_coeffs(lin, dt)
    dealias = np.ones(N // 2 + 1)
    dealias[int(np.floor(N / 3)):] = 0.0
    gcoef = (-0.5j * k) * dealias

    def g(vv):
        return gcoef * np.fft.rfft(np.fft.irfft(vv, n=N, axis=-1) ** 2, axis=-1)

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
        v[:, 0] = 0.0  # enforce the conserved mean-zero sector against roundoff
        if step >= n_trans:
            count += 1
            if count == steps_per_sample:
                buf[idx] = np.fft.irfft(v, n=N, axis=-1)
                idx += 1
                count = 0
                emitted += 1
                if idx == chunk_samples:
                    yield buf[:idx].copy()
                    idx = 0
    if idx:
        yield buf[:idx].copy()
    assert emitted == n_samples
