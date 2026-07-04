r"""Differentiable Kac-Rice level-crossing density estimator.

Core identity (co-area formula): for Lipschitz f: R^d -> R and a test function g,

    (1/|Omega|) \int_Omega g(f(x)) |grad f(x)| dx
        = (1/|Omega|) \int_R g(u) H^{d-1}( {f = u} ) du

With g = delta_eps centered at level u this yields the (smoothed) level-set density:
number of crossings per unit length in 1D, level-set length per unit area in 2D,
level-set area per unit volume in 3D. Monte-Carlo over sampled coordinates:

    c(u) ~= (1/N) sum_i delta_eps( f(x_i) - u ) * |grad f(x_i)|

For a stationary Gaussian process this expectation reduces to the Rice formula
mu(u) = (1/pi) sqrt(lambda_2/lambda_0) exp(-u^2 / (2 lambda_0)) (up+down crossings),
which is monotone in sqrt(lambda_2) = RMS derivative = high-frequency energy. We never
rely on Gaussianity: the training target is the *empirical* crossing density of the
ground truth evaluated at the same batch points, so the Monte-Carlo noise of estimator
and target is correlated and partially cancels.

Everything here is grid-free and FFT-free: it needs only field values and gradients at
arbitrary sample points, which is exactly what an INR provides via autograd.
"""

import math

import torch
import torch.nn as nn


def field_and_grad(model, coords, create_graph=True):
    """Evaluate model and its spatial gradient at coords.

    coords: (N, d) tensor. Returns (values (N, C), grad (N, d, C) squeezed to (N, d)
    when the output is scalar). create_graph=True keeps the graph so losses on the
    gradient backprop into model parameters (double backprop).
    """
    coords = coords.detach().requires_grad_(True)
    values = model(coords)
    if values.dim() == 1:
        values = values.unsqueeze(-1)
    grads = []
    for c in range(values.shape[-1]):
        (g,) = torch.autograd.grad(
            values[..., c].sum(), coords, create_graph=create_graph
        )
        grads.append(g)
    grad = torch.stack(grads, dim=-1)  # (N, d, C)
    if values.shape[-1] == 1:
        return values[..., 0], grad[..., 0]
    return values, grad


def gaussian_delta(z, eps):
    """Smoothed Dirac: Gaussian kernel with bandwidth (std) eps."""
    return torch.exp(-0.5 * (z / eps) ** 2) / (eps * math.sqrt(2.0 * math.pi))


def make_levels(values, n_levels=16, lo=0.02, hi=0.98):
    """Pick crossing levels at quantiles of the (ground-truth) value distribution.

    Quantile placement keeps every level populated regardless of the signal's range
    or value histogram; uniform levels can land in empty regions and contribute
    nothing but noise.
    """
    q = torch.linspace(lo, hi, n_levels, device=values.device, dtype=values.dtype)
    return torch.quantile(values.detach().flatten().float(), q.float()).to(values.dtype)


def crossing_density(values, grad_norm, levels, eps):
    """Monte-Carlo Kac-Rice integrand: c(u_j) = mean_i delta_eps(f_i - u_j) |grad f_i|.

    values: (N,), grad_norm: (N,), levels: (L,), eps: scalar bandwidth.
    Returns (L,) crossing density per level. Differentiable in values and grad_norm.
    """
    z = values.unsqueeze(0) - levels.unsqueeze(1)  # (L, N)
    return (gaussian_delta(z, eps) * grad_norm.unsqueeze(0)).mean(dim=1)


class KacRiceLoss(nn.Module):
    """Match the model's crossing-density profile to the ground truth's.

    L = mean_j w_j * ( c_theta(u_j) - c_gt(u_j) )^2

    Both densities are estimated on the SAME batch of points, so the shared
    Monte-Carlo sampling noise largely cancels in the difference.

    Parameters
    ----------
    n_levels : number of crossing levels u_j (quantiles of GT values).
    eps : smoothed-Dirac bandwidth. If None, set to eps_scale * std(gt values)
          per batch (the spec flags eps as the nuisance hyperparameter to sweep).
    eps_scale : relative bandwidth used when eps is None.
    relative : if True (default), normalize each level's squared error by
               (c_gt + mean c_gt)^2, making the loss scale-free across levels and
               across signals — crossing densities grow linearly with frequency, so
               the raw squared error would otherwise scale quadratically with the
               signal's frequency content and swamp the reconstruction MSE.
    """

    def __init__(self, n_levels=16, eps=None, eps_scale=0.15, relative=True):
        super().__init__()
        self.n_levels = n_levels
        self.eps = eps
        self.eps_scale = eps_scale
        self.relative = relative

    def forward(self, pred_values, pred_grad, gt_values, gt_grad):
        """pred_grad / gt_grad: (N, d) spatial gradients (or (N,) already-normed)."""
        pred_gn = pred_grad.norm(dim=-1) if pred_grad.dim() > 1 else pred_grad
        gt_gn = gt_grad.norm(dim=-1) if gt_grad.dim() > 1 else gt_grad

        with torch.no_grad():
            levels = make_levels(gt_values, self.n_levels)
            eps = self.eps
            if eps is None:
                eps = self.eps_scale * gt_values.float().std().clamp_min(1e-8).item()

        c_pred = crossing_density(pred_values, pred_gn, levels, eps)
        with torch.no_grad():
            c_gt = crossing_density(gt_values, gt_gn, levels, eps)

        err = (c_pred - c_gt) ** 2
        if self.relative:
            err = err / (c_gt + c_gt.mean().clamp_min(1e-8)) ** 2
        return err.mean()


class CrossingBudgetLoss(nn.Module):
    """One-sided crossing-density BUDGET (Phase 2): penalize only the EXCESS of
    the field's crossing density over a physical budget b_j per level,

        L = mean_j relu( c_theta(u_j) - b_j )^2 / (b_j + mean b)^2.

    One-sidedness is mandatory for the PINN use case: early in training the
    field is smooth (crossings *below* budget); a symmetric match would pump
    oscillation *up* -- the exact failure being prevented. Levels are FIXED
    from the amplitude range of the solution class (there is no ground-truth
    field in a PINN), and budgets come from cgle.budgets_* sources.
    """

    def __init__(self, levels, budgets, eps, relative=True):
        super().__init__()
        self.register_buffer("levels", torch.as_tensor(levels, dtype=torch.float32))
        self.register_buffer("budgets", torch.as_tensor(budgets, dtype=torch.float32))
        self.eps = float(eps)
        self.relative = relative

    def forward(self, values, grad_norm):
        c = crossing_density(values, grad_norm, self.levels, self.eps)
        over = torch.relu(c - self.budgets)
        if self.relative:
            over = over / (self.budgets + self.budgets.mean().clamp_min(1e-8))
        return over.pow(2).mean()
