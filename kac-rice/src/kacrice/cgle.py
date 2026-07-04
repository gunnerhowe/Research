"""Complex Ginzburg-Landau reference solver + crossing-budget extraction.

Equation (ASPEN benchmark, arXiv 2512.03290):
    dA/dt = A + (1+ib) A_xx - (1+ic) |A|^2 A,   b=0.5, c=-1.3
    x in [-10, 7.5], t in [0, 10]
    A(x,0)   = tanh(-x) + 0i          (domain-wall front)
    Dirichlet: A(-10,t) = tanh(10),  A(7.5,t) = tanh(-7.5)

Dirichlet ends rule out the FFT/ETDRK4 route used for periodic KS
(ornstein-dist), so this is method-of-lines: 2nd-order central differences +
RK4 at ASPEN's own dt=1e-4 (satisfies the diffusive CFL at nx=1024).

Also provides the three crossing-budget sources of the Phase-2 spec:
  (a) empirical per-level budgets measured on the reference trajectory,
  (b) a physics prior for the monotone-front class (each level crossed once
      per front passage -> budget ~ m / L_domain, no simulation needed),
  (c) a single scalar cap.
"""

import numpy as np
import torch

from .crossing import crossing_density

B_DEFAULT, C_DEFAULT = 0.5, -1.3
X0, X1 = -10.0, 7.5
T1 = 10.0


def cgle_reference(b=B_DEFAULT, c=C_DEFAULT, nx=1024, dt=1e-4, t1=T1,
                   n_save=201, x0=X0, x1=X1):
    """Reference CGLE solution. Returns (x (nx,), t (n_save,), A (n_save, nx) complex)."""
    x = np.linspace(x0, x1, nx)
    dx = x[1] - x[0]
    A = np.tanh(-x).astype(np.complex128)
    bc_l, bc_r = np.tanh(-x0), np.tanh(-x1)  # tanh(10), tanh(-7.5)

    lin_coef = 1.0 + 1j * b
    nl_coef = 1.0 + 1j * c

    def rhs(a):
        axx = np.empty_like(a)
        axx[1:-1] = (a[2:] - 2 * a[1:-1] + a[:-2]) / dx**2
        axx[0] = axx[-1] = 0.0  # boundary values are pinned; rhs there unused
        r = a + lin_coef * axx - nl_coef * (np.abs(a) ** 2) * a
        r[0] = r[-1] = 0.0
        return r

    n_steps = int(round(t1 / dt))
    save_at = np.linspace(0, n_steps, n_save).round().astype(int)
    out = np.empty((n_save, nx), dtype=np.complex128)
    t_out = save_at * dt
    si = 0
    for step in range(n_steps + 1):
        if si < n_save and step == save_at[si]:
            out[si] = A
            si += 1
        if step == n_steps:
            break
        k1 = rhs(A)
        k2 = rhs(A + 0.5 * dt * k1)
        k3 = rhs(A + 0.5 * dt * k2)
        k4 = rhs(A + dt * k3)
        A = A + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        A[0], A[-1] = bc_l, bc_r
    return x, t_out, out


def spatial_crossing_profile(field, x, levels, eps):
    """Empirical spatial crossing density of a real (n_t, nx) field at each level,
    per time slice: c_t(u) = mean_x delta_eps(f - u) |df/dx|.  Returns (n_t, L)."""
    dx = x[1] - x[0]
    grad = np.gradient(field, dx, axis=1)
    out = np.empty((field.shape[0], len(levels)))
    for i in range(field.shape[0]):
        out[i] = crossing_density(
            torch.from_numpy(field[i]).float(),
            torch.from_numpy(np.abs(grad[i])).float(),
            torch.as_tensor(levels, dtype=torch.float32),
            eps,
        ).numpy()
    return out


def budgets_from_reference(A_ref, x, levels, eps, quantile=0.95, slack=1.5):
    """Budget source (a): per-level crossing budgets measured on the reference
    trajectory (Re and Im separately). quantile over time slices tolerates
    transients; slack keeps legitimate dynamics strictly inside the budget."""
    out = {}
    for name, field in (("re", A_ref.real), ("im", A_ref.imag)):
        prof = spatial_crossing_profile(field, x, levels, eps)
        out[name] = slack * np.quantile(prof, quantile, axis=0)
    return out


def budgets_from_physics(levels, m_crossings=4.0, domain_len=X1 - X0):
    """Budget source (b): monotone-front solution class. A front crosses each
    level in the amplitude range once per passage; allow m passages (front +
    shed transients). No simulation, no spectrum - a prior from the physics."""
    b = np.full(len(levels), m_crossings / domain_len)
    return {"re": b, "im": b.copy()}


def budgets_scalar(levels, cap):
    """Budget source (c): one global scalar cap for every level and component."""
    b = np.full(len(levels), cap)
    return {"re": b, "im": b.copy()}


def cgle_chaotic_reference(b=2.0, c=-1.2, L=64.0, nx=256, dt=5e-4, t1=5.0,
                           n_save=101, transient=20.0, seed=0):
    """Benjamin-Feir-UNSTABLE CGLE (1 + bc < 0) on a periodic domain: defect /
    phase turbulence with genuine broadband spatial content. ETDRK4 in Fourier
    space (adapting the ornstein-dist KS scheme to a complex field, where both
    positive and negative wavenumbers are retained via fft, not rfft).

    Integrates a transient to land on the attractor, then returns
    (x, t, A[n_save, nx]) over a window of length t1 starting at the attractor
    state. The PINN IC is A[0]."""
    rng = np.random.default_rng(seed)
    x = L * np.arange(nx) / nx
    k = 2 * np.pi * np.fft.fftfreq(nx, d=L / nx)
    lin = 1.0 - (1.0 + 1j * b) * k**2

    M = 32
    LR = dt * lin[:, None] + np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)[None, :]
    E = np.exp(dt * lin)
    E2 = np.exp(dt * lin / 2.0)
    Q = dt * np.mean((np.exp(LR / 2.0) - 1) / LR, axis=1)
    f1 = dt * np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1)
    f2 = dt * np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1)
    f3 = dt * np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1)

    dealias = np.abs(k) < (2.0 / 3.0) * np.abs(k).max()
    nl_coef = -(1.0 + 1j * c)

    def g(vv):
        a = np.fft.ifft(vv)
        return dealias * np.fft.fft(nl_coef * (np.abs(a) ** 2) * a)

    A = 0.8 * np.exp(1j * rng.uniform(0, 2 * np.pi, nx))
    A += 0.1 * rng.standard_normal(nx)
    v = np.fft.fft(A)

    n_trans = int(round(transient / dt))
    n_steps = int(round(t1 / dt))
    save_at = np.linspace(0, n_steps, n_save).round().astype(int)
    out = np.empty((n_save, nx), dtype=np.complex128)
    si = 0
    for step in range(n_trans + n_steps + 1):
        if step >= n_trans:
            rel = step - n_trans
            if si < n_save and rel == save_at[si]:
                out[si] = np.fft.ifft(v)
                si += 1
            if rel == n_steps:
                break
        Nv = g(v)
        a = E2 * v + Q * Nv
        Na = g(a)
        bb = E2 * v + Q * Na
        Nb = g(bb)
        cc = E2 * a + Q * (2 * Nb - Nv)
        Nc = g(cc)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
    t_out = save_at * dt
    return x, t_out, out
