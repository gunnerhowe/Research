"""Differentiable Minkowski-functional profiles of neural fields (Phase 3).

For a field f and superlevel sets A_u = {f >= u}, the three 2D Minkowski
functionals (and their 3D analogues) admit co-area / Gauss-Bonnet integral
representations that are Monte-Carlo estimable at scattered sample points with
autograd derivatives only -- no grid, no complex, no persistence machinery:

  M0(u)  area/volume:       mean  sigmoid_eps(f - u)                  * V
  M1(u)  perimeter/area:    mean  delta_eps(f - u) |grad f|           * V
  M2(u)  Euler characteristic:
    2D:  chi(A_u) = -(V / 2 pi) mean[ delta_eps(f-u) * k2d ]
         k2d = (f_xx f_y^2 - 2 f_xy f_x f_y + f_yy f_x^2) / |grad f|^2
         (= div(grad f/|grad f|) * |grad f|; sign checked on the disc)
    3D:  chi(A_u) = chi(boundary surface)/2 and  int_S K dA = 2 pi chi(S):
         chi(A_u) = (V / 4 pi) mean[ delta_eps(f-u) * k3d ]
         k3d = (grad f^T adj(H) grad f) / |grad f|^3     (Goldman 2005)

Assumes level sets do not touch the domain boundary (fields decay below the
probed levels near the boundary); otherwise Gauss-Bonnet boundary terms are
missing -- validation fields are constructed accordingly, and the SDF
application probes interior levels only.
"""

import math

import torch
import torch.nn as nn

from .crossing import crossing_density, gaussian_delta


def field_derivatives(model, coords, order=2):
    """Values, gradient, and (order=2) Hessian rows of a scalar model at coords.

    coords: (N, d). Returns values (N,), grad (N, d), hess (N, d, d) with
    create_graph so downstream losses backprop into model parameters.
    """
    coords = coords.detach().requires_grad_(True)
    values = model(coords)
    if values.dim() > 1:
        values = values.squeeze(-1)
    (grad,) = torch.autograd.grad(values.sum(), coords, create_graph=True)
    if order < 2:
        return values, grad, None
    rows = []
    for k in range(coords.shape[1]):
        (row,) = torch.autograd.grad(grad[:, k].sum(), coords,
                                     create_graph=True)
        rows.append(row)
    hess = torch.stack(rows, dim=1)  # (N, d, d)
    return values, grad, hess


def curvature_density(grad, hess, eta=1e-8):
    """The Gauss-Bonnet integrand factor k(x) such that
    chi-density = delta_eps(f-u) * k(x), per the module docstring.

    2D: k2d = (f_xx f_y^2 - 2 f_xy f_x f_y + f_yy f_x^2) / (|grad|^2 + eta)
    3D: k3d = (g^T adj(H) g) / (|grad|^3 + eta)
    """
    d = grad.shape[1]
    if d == 2:
        fx, fy = grad[:, 0], grad[:, 1]
        fxx, fxy, fyy = hess[:, 0, 0], hess[:, 0, 1], hess[:, 1, 1]
        num = fxx * fy**2 - 2 * fxy * fx * fy + fyy * fx**2
        return num / (fx**2 + fy**2 + eta)
    if d == 3:
        # adjugate of symmetric 3x3 H, contracted with grad on both sides
        H = hess
        g = grad
        c00 = H[:, 1, 1] * H[:, 2, 2] - H[:, 1, 2] * H[:, 2, 1]
        c01 = H[:, 1, 2] * H[:, 2, 0] - H[:, 1, 0] * H[:, 2, 2]
        c02 = H[:, 1, 0] * H[:, 2, 1] - H[:, 1, 1] * H[:, 2, 0]
        c11 = H[:, 0, 0] * H[:, 2, 2] - H[:, 0, 2] * H[:, 2, 0]
        c12 = H[:, 0, 1] * H[:, 2, 0] - H[:, 0, 0] * H[:, 2, 1]
        c22 = H[:, 0, 0] * H[:, 1, 1] - H[:, 0, 1] * H[:, 1, 0]
        quad = (g[:, 0] ** 2 * c00 + g[:, 1] ** 2 * c11 + g[:, 2] ** 2 * c22
                + 2 * g[:, 0] * g[:, 1] * c01
                + 2 * g[:, 0] * g[:, 2] * c02
                + 2 * g[:, 1] * g[:, 2] * c12)
        gn = grad.norm(dim=1)
        return quad / (gn**3 + eta)
    raise ValueError(f"curvature_density supports d=2,3; got d={d}")


def euler_profile(values, grad, hess, levels, eps, volume):
    """chi-hat(u_j) for each level: differentiable Euler-characteristic profile.

    2D: chi = -(V/2pi) mean[delta * k2d];  3D: chi = (V/4pi) mean[delta * k3d].
    """
    d = grad.shape[1]
    k = curvature_density(grad, hess)
    z = values.unsqueeze(0) - levels.unsqueeze(1)          # (L, N)
    w = gaussian_delta(z, eps) * k.unsqueeze(0)            # (L, N)
    if d == 2:
        return -(volume / (2 * math.pi)) * w.mean(dim=1)
    return (volume / (4 * math.pi)) * w.mean(dim=1)


def minkowski_profiles(values, grad, hess, levels, eps, volume):
    """(M0, M1, M2) profiles, all differentiable. M0 uses a sigmoid of matched
    bandwidth; M1 is the Phase-1 crossing density scaled to absolute measure."""
    z = levels.unsqueeze(1) - values.unsqueeze(0)
    m0 = torch.sigmoid(-z / eps).mean(dim=1) * volume
    m1 = crossing_density(values, grad.norm(dim=1), levels, eps) * volume
    m2 = euler_profile(values, grad, hess, levels, eps, volume)
    return m0, m1, m2


class EulerProfileLoss(nn.Module):
    """Match or one-sidedly cap the Euler-characteristic profile.

    mode='match':  L = mean_j (chi(u_j) - target_j)^2 / (1 + |target_j|)^2
    mode='cap':    L = mean_j relu(chi(u_j) - target_j)^2 / (1 + |target_j|)^2
                   (penalize only EXCESS components/handles above target;
                   one-sided, like Phase 2's budget)
    """

    def __init__(self, levels, targets, eps, volume, mode="cap"):
        super().__init__()
        self.register_buffer("levels", torch.as_tensor(levels,
                                                       dtype=torch.float32))
        self.register_buffer("targets", torch.as_tensor(targets,
                                                        dtype=torch.float32))
        self.eps = float(eps)
        self.volume = float(volume)
        self.mode = mode

    def forward(self, values, grad, hess):
        chi = euler_profile(values, grad, hess, self.levels, self.eps,
                            self.volume)
        err = chi - self.targets
        if self.mode == "cap":
            err = torch.relu(err)
        return (err / (1.0 + self.targets.abs())).pow(2).mean()
