"""PINN for the complex Ginzburg-Landau equation (ASPEN benchmark setting).

Residual for A = u + iv,   A_t = A + (1+ib) A_xx - (1+ic)|A|^2 A:
    r_u = u_t - [ u + (u_xx - b v_xx) - |A|^2 (u - c v) ]
    r_v = v_t - [ v + (b u_xx + v_xx) - |A|^2 (c u + v) ]

Exact plane-wave solutions A = sqrt(1-q^2) exp(i(qx - w t)), w = b q^2 + c(1-q^2)
serve as the residual unit test.

Models: vanilla tanh MLP (8x40 -- ASPEN's failing baseline) and the same MLP
behind a learnable Fourier-feature layer (m=128, K ~ N(0, 10^2) -- a faithful
reimplementation of ASPEN's adaptive spectral layer as described in the paper).
Inputs are normalized to [-1,1]^2 internally.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from .cgle import B_DEFAULT, C_DEFAULT, T1, X0, X1


class CGLEMlp(nn.Module):
    """tanh MLP (x,t) -> (u,v), optionally behind a learnable Fourier layer."""

    def __init__(self, hidden=40, layers=8, fourier_m=0, fourier_sigma=10.0,
                 normalize=True):
        super().__init__()
        self.normalize = normalize
        self.fourier = fourier_m > 0
        if self.fourier:
            self.K = nn.Parameter(torch.randn(fourier_m, 2) * fourier_sigma)
            in_dim = 2 * fourier_m
        else:
            in_dim = 2
        seq = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.Tanh()]
        seq.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*seq)

    def forward(self, xt):
        if self.normalize:
            x = 2 * (xt[:, 0:1] - X0) / (X1 - X0) - 1
            t = 2 * xt[:, 1:2] / T1 - 1
            z = torch.cat([x, t], dim=1)
        else:
            z = xt  # raw physical coordinates (diagnostic: ASPEN-style input)
        if self.fourier:
            proj = 2 * math.pi * z @ self.K.T
            z = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
        return self.net(z)


class CGLEMlpPeriodic(nn.Module):
    """Periodic-domain variant for the chaotic testbed: x enters only through
    sin/cos(2 pi n x / L) harmonics, so spatial periodicity holds by
    construction (no BC loss needed). t is normalized to [-1, 1]."""

    def __init__(self, L, t1, hidden=64, layers=6, harmonics=4):
        super().__init__()
        self.L, self.t1, self.h = L, t1, harmonics
        seq = [nn.Linear(2 * harmonics + 1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.Tanh()]
        seq.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*seq)

    def forward(self, xt):
        ang = 2 * math.pi * xt[:, 0:1] / self.L
        n = torch.arange(1, self.h + 1, device=xt.device, dtype=xt.dtype)
        feats = [torch.sin(ang * n), torch.cos(ang * n),
                 2 * xt[:, 1:2] / self.t1 - 1]
        return self.net(torch.cat(feats, dim=1))


def residual_and_grads(model, xt, b=B_DEFAULT, c=C_DEFAULT):
    """CGLE residual at collocation points, plus u, v and their spatial
    derivatives (u_x, v_x reused by the crossing-budget loss for free)."""
    xt = xt.detach().requires_grad_(True)
    uv = model(xt)
    u, v = uv[:, 0], uv[:, 1]

    (gu,) = torch.autograd.grad(u.sum(), xt, create_graph=True)
    (gv,) = torch.autograd.grad(v.sum(), xt, create_graph=True)
    u_x, u_t = gu[:, 0], gu[:, 1]
    v_x, v_t = gv[:, 0], gv[:, 1]

    (guu,) = torch.autograd.grad(u_x.sum(), xt, create_graph=True)
    (gvv,) = torch.autograd.grad(v_x.sum(), xt, create_graph=True)
    u_xx, v_xx = guu[:, 0], gvv[:, 0]

    amp2 = u * u + v * v
    r_u = u_t - (u + (u_xx - b * v_xx) - amp2 * (u - c * v))
    r_v = v_t - (v + (b * u_xx + v_xx) - amp2 * (c * u + v))
    return r_u, r_v, u, v, u_x, v_x, u_xx, v_xx


def make_points(n_res=20000, n_ic=1000, n_bc=1000, seed=0, device="cpu"):
    """Latin-hypercube collocation + IC/BC points (ASPEN protocol)."""
    from scipy.stats import qmc

    s = qmc.LatinHypercube(d=2, seed=seed).random(n_res)
    res = np.column_stack([X0 + s[:, 0] * (X1 - X0), s[:, 1] * T1])

    rng = np.random.default_rng(seed + 1)
    x_ic = X0 + rng.random(n_ic) * (X1 - X0)
    ic = np.column_stack([x_ic, np.zeros(n_ic)])
    ic_target = np.column_stack([np.tanh(-x_ic), np.zeros(n_ic)])

    t_bc = rng.random(n_bc) * T1
    side = rng.integers(0, 2, n_bc)
    x_bc = np.where(side == 0, X0, X1)
    bc = np.column_stack([x_bc, t_bc])
    bc_target = np.column_stack([np.tanh(-x_bc), np.zeros(n_bc)])

    to = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    return to(res), to(ic), to(ic_target), to(bc), to(bc_target)


def pinn_losses(model, pts, b=B_DEFAULT, c=C_DEFAULT):
    """Per-point squared residual + IC/BC loss; also returns the field and its
    derivatives at collocation points so auxiliary losses reuse the same graph.
    Callers aggregate the residual (mean, or causally weighted mean)."""
    res, ic, ic_t = pts[0], pts[1], pts[2]
    r_u, r_v, u, v, u_x, v_x, u_xx, v_xx = residual_and_grads(model, res, b, c)
    r2 = r_u**2 + r_v**2
    loss_icbc = ((model(ic) - ic_t) ** 2).mean()
    if len(pts) > 3:  # Dirichlet testbed; absent on the hard-periodic one
        bc, bc_t = pts[3], pts[4]
        loss_icbc = loss_icbc + ((model(bc) - bc_t) ** 2).mean()
    return r2, loss_icbc, (u, v, u_x, v_x, u_xx, v_xx)


def make_points_chaotic(x_ref, A0, L, t1, n_res=20000, n_ic=1000, seed=0,
                        device="cpu"):
    """Collocation + IC points for the periodic chaotic testbed. The IC is the
    attractor state A0 = A(x, 0) from the reference simulation, interpolated
    at random x. No BC points: periodicity is hard-wired in the model."""
    from scipy.stats import qmc

    s = qmc.LatinHypercube(d=2, seed=seed).random(n_res)
    res = np.column_stack([s[:, 0] * L, s[:, 1] * t1])

    rng = np.random.default_rng(seed + 1)
    x_ic = rng.random(n_ic) * L
    xp = np.concatenate([x_ref, [L]])  # close the periodic interval
    re = np.interp(x_ic, xp, np.concatenate([A0.real, [A0.real[0]]]))
    im = np.interp(x_ic, xp, np.concatenate([A0.imag, [A0.imag[0]]]))
    ic = np.column_stack([x_ic, np.zeros(n_ic)])
    ic_target = np.column_stack([re, im])

    to = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    return to(res), to(ic), to(ic_target)


@torch.no_grad()
def eval_on_reference(model, x_ref, t_ref, A_ref, device, chunk=131072):
    """Relative L2 error against the reference trajectory on its (t, x) grid."""
    tt, xx = np.meshgrid(t_ref, x_ref, indexing="ij")
    pts = torch.tensor(
        np.column_stack([xx.ravel(), tt.ravel()]), dtype=torch.float32,
        device=device)
    outs = []
    for i in range(0, pts.shape[0], chunk):
        outs.append(model(pts[i : i + chunk]).cpu())
    uv = torch.cat(outs).numpy()
    A_pred = (uv[:, 0] + 1j * uv[:, 1]).reshape(A_ref.shape)
    num = np.linalg.norm(A_pred - A_ref)
    den = np.linalg.norm(A_ref)
    return float(num / den), A_pred
