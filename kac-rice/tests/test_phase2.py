"""Phase 2 correctness tests: CGLE residual, reference solver, budget loss.

Run: python tests/test_phase2.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kacrice.cgle import (B_DEFAULT, C_DEFAULT, budgets_from_physics,
                          budgets_from_reference, cgle_reference)
from kacrice.crossing import CrossingBudgetLoss
from kacrice.pinn import CGLEMlp, make_points, residual_and_grads


class PlaneWave(torch.nn.Module):
    """Exact CGLE solution A = sqrt(1-q^2) exp(i(qx - w t)), w = b q^2 + c(1-q^2)."""

    def __init__(self, q=0.3, b=B_DEFAULT, c=C_DEFAULT):
        super().__init__()
        self.q = q
        self.rho = math.sqrt(1 - q**2)
        self.w = b * q**2 + c * (1 - q**2)

    def forward(self, xt):
        phase = self.q * xt[:, 0] - self.w * xt[:, 1]
        return torch.stack([self.rho * torch.cos(phase),
                            self.rho * torch.sin(phase)], dim=1)


def test_residual_zero_on_exact_solution():
    torch.manual_seed(0)
    xt = torch.rand(2000, 2)
    xt[:, 0] = xt[:, 0] * 17.5 - 10.0
    xt[:, 1] = xt[:, 1] * 10.0
    r_u, r_v, *_ = residual_and_grads(PlaneWave(), xt)
    err = (r_u**2 + r_v**2).mean().item()
    assert err < 1e-8, f"plane-wave residual should vanish, got {err:.3e}"


def test_reference_solver_short_horizon():
    """Front stays bounded, boundary values pinned, and the solution is
    grid-converged (nx 512 vs 1024) over a short horizon."""
    x1, t1, A1 = cgle_reference(nx=512, dt=2e-4, t1=1.0, n_save=11)
    x2, t2, A2 = cgle_reference(nx=1024, dt=1e-4, t1=1.0, n_save=11)
    assert np.isfinite(A1).all() and np.isfinite(A2).all()
    assert np.abs(A2).max() < 1.6, "solution should stay O(1)"
    assert abs(A2[-1, 0] - np.tanh(10.0)) < 1e-6, "left BC pinned"
    assert abs(A2[-1, -1] - np.tanh(-7.5)) < 1e-6, "right BC pinned"
    # compare on the coarse grid (every 2nd fine point aligns for linspace)
    A2c = A2[:, ::2][:, : A1.shape[1]]
    interp = np.array([np.interp(x1, x2, A2[k].real) + 1j *
                       np.interp(x1, x2, A2[k].imag) for k in range(len(t2))])
    rel = np.linalg.norm(A1 - interp) / np.linalg.norm(interp)
    assert rel < 0.02, f"grid convergence: rel diff {rel:.4f}"


def test_budget_loss_one_sided():
    """Smooth under-budget field -> zero loss, zero gradient (no upward push).
    Oscillatory over-budget field -> positive loss, finite gradients."""
    levels = np.linspace(-0.9, 0.9, 8)
    budgets = budgets_from_physics(levels, m_crossings=4.0)["re"]
    loss_fn = CrossingBudgetLoss(levels, budgets, eps=0.1)

    n = 4000
    x = torch.linspace(-10, 7.5, n)

    smooth = torch.tanh(-x).requires_grad_(True)
    g_smooth = torch.full((n,), 1.0 / 10)  # gentle slope magnitude
    l0 = loss_fn(smooth, g_smooth)
    assert l0.item() == 0.0, f"under budget must cost nothing, got {l0.item():.2e}"
    l0.backward()
    assert smooth.grad.abs().max().item() == 0.0, "one-sidedness violated"

    wig = torch.sin(4.0 * x).requires_grad_(True)  # ~11 crossings/level per 17.5
    g_wig = (4.0 * torch.cos(4.0 * x)).abs()
    l1 = loss_fn(wig, g_wig)
    assert l1.item() > 0.01, f"over budget must be penalized, got {l1.item():.2e}"
    l1.backward()
    assert torch.isfinite(wig.grad).all() and wig.grad.abs().max() > 0


def test_budgets_from_reference_front():
    """Reference-derived budgets for the front solution should be small
    (a monotone front crosses each mid-range level ~once per domain)."""
    x, t, A = cgle_reference(nx=512, dt=2e-4, t1=1.0, n_save=11)
    levels = np.linspace(-0.9, 0.9, 8)
    b = budgets_from_reference(A, x, levels, eps=0.1)
    assert (b["re"] > 0).all()
    assert b["re"].max() < 0.6, f"front budgets should be O(0.1), got {b['re'].max():.2f}"


def test_pinn_forward_and_points():
    torch.manual_seed(0)
    model = CGLEMlp()
    pts = make_points(n_res=256, n_ic=32, n_bc=32)
    out = model(pts[0])
    assert out.shape == (256, 2) and torch.isfinite(out).all()
    ff = CGLEMlp(fourier_m=16)
    assert ff(pts[0]).shape == (256, 2)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {name}: {type(e).__name__}: {e}")
