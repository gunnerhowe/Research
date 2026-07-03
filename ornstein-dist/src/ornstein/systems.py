"""Lorenz-63 trajectory generation (numba RK4), with time-rescaling and parameter variants.

The `speed` parameter integrates dx/dt = speed * f(x): identical attractor and invariant
measure, rescaled clock — the "same measure, different dynamics" adversarial construction.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _rk4_lorenz(n_keep, sample_every, dt, sigma, rho, beta, speed,
                x, y, z, n_transient_steps):
    out = np.empty((n_keep, 3))
    for _ in range(n_transient_steps):
        k1x = speed * (sigma * (y - x))
        k1y = speed * (x * (rho - z) - y)
        k1z = speed * (x * y - beta * z)
        x2, y2, z2 = x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z
        k2x = speed * (sigma * (y2 - x2))
        k2y = speed * (x2 * (rho - z2) - y2)
        k2z = speed * (x2 * y2 - beta * z2)
        x3, y3, z3 = x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z
        k3x = speed * (sigma * (y3 - x3))
        k3y = speed * (x3 * (rho - z3) - y3)
        k3z = speed * (x3 * y3 - beta * z3)
        x4, y4, z4 = x + dt * k3x, y + dt * k3y, z + dt * k3z
        k4x = speed * (sigma * (y4 - x4))
        k4y = speed * (x4 * (rho - z4) - y4)
        k4z = speed * (x4 * y4 - beta * z4)
        x += dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
        y += dt * (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0
        z += dt * (k1z + 2 * k2z + 2 * k3z + k4z) / 6.0
    idx = 0
    count = 0
    while idx < n_keep:
        k1x = speed * (sigma * (y - x))
        k1y = speed * (x * (rho - z) - y)
        k1z = speed * (x * y - beta * z)
        x2, y2, z2 = x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z
        k2x = speed * (sigma * (y2 - x2))
        k2y = speed * (x2 * (rho - z2) - y2)
        k2z = speed * (x2 * y2 - beta * z2)
        x3, y3, z3 = x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z
        k3x = speed * (sigma * (y3 - x3))
        k3y = speed * (x3 * (rho - z3) - y3)
        k3z = speed * (x3 * y3 - beta * z3)
        x4, y4, z4 = x + dt * k3x, y + dt * k3y, z + dt * k3z
        k4x = speed * (sigma * (y4 - x4))
        k4y = speed * (x4 * (rho - z4) - y4)
        k4z = speed * (x4 * y4 - beta * z4)
        x += dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
        y += dt * (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0
        z += dt * (k1z + 2 * k2z + 2 * k3z + k4z) / 6.0
        count += 1
        if count == sample_every:
            out[idx, 0] = x
            out[idx, 1] = y
            out[idx, 2] = z
            idx += 1
            count = 0
    return out


def lorenz_trajectory(n_samples, tau=0.1, dt=0.005, sigma=10.0, rho=28.0,
                      beta=8.0 / 3.0, speed=1.0, seed=0, transient=100.0):
    """Sampled Lorenz-63 trajectory, shape (n_samples, 3), sampled every tau time units.

    Integration step dt must divide tau. Transient (in model time units) is discarded.
    Initial condition is a seeded random perturbation so independent runs decorrelate.
    """
    sample_every = int(round(tau / dt))
    if abs(sample_every * dt - tau) > 1e-12:
        raise ValueError(f"dt={dt} does not divide tau={tau}")
    rng = np.random.default_rng(seed)
    x0, y0, z0 = np.array([1.0, 1.0, 20.0]) + rng.normal(0, 2.0, 3)
    n_transient_steps = int(round(transient / dt))
    return _rk4_lorenz(n_samples, sample_every, dt, sigma, rho, beta, speed,
                       x0, y0, z0, n_transient_steps)
