"""GATE V: the Euler-profile estimator against exact topology.

Analytic fields whose superlevel-set topology is known by construction; all
fields decay below probed levels near the domain boundary (no Gauss-Bonnet
boundary terms). Run: python tests/test_phase3.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kacrice.minkowski import euler_profile, field_derivatives

N2, N3 = 200_000, 200_000
EPS = 0.02


class Analytic(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def chi_hat(fn, d, levels, n, box=1.0, eps=EPS, seed=0):
    torch.manual_seed(seed)
    pts = (torch.rand(n, d) * 2 - 1) * box
    v, g, h = field_derivatives(Analytic(fn), pts)
    vol = (2.0 * box) ** d
    lv = torch.as_tensor(levels, dtype=torch.float32)
    return euler_profile(v, g, h, lv, eps, vol).detach()


def gauss(x, c, s):
    return torch.exp(-((x - torch.tensor(c)) ** 2).sum(-1) / (2 * s**2))


def test_2d_one_bump():
    """Single Gaussian bump: chi(A_u) = 1 for u in (0.1, 0.9)."""
    f = lambda x: gauss(x, [0.0, 0.0], 0.35)  # noqa: E731
    chi = chi_hat(f, 2, [0.2, 0.5, 0.8], N2)
    for c in chi.tolist():
        assert abs(c - 1.0) < 0.12, f"one bump: chi={c:.3f}, want 1"


def test_2d_two_bumps_merge():
    """Two bumps (heights 1 and 0.6): chi=2 below 0.6, chi=1 above."""
    f = lambda x: gauss(x, [-0.45, 0.0], 0.22) + 0.6 * gauss(x, [0.45, 0.0], 0.22)  # noqa: E731
    chi = chi_hat(f, 2, [0.25, 0.45, 0.75], N2)
    assert abs(chi[0] - 2.0) < 0.2, f"below merge: chi={chi[0]:.3f}, want 2"
    assert abs(chi[1] - 2.0) < 0.2, f"below merge: chi={chi[1]:.3f}, want 2"
    assert abs(chi[2] - 1.0) < 0.15, f"above: chi={chi[2]:.3f}, want 1"


def test_2d_annulus_hole():
    """Ring ridge exp(-(r-0.5)^2/w^2): mid-level superlevel set is an annulus,
    chi = 0. The sign-sensitive test: components minus holes."""
    def f(x):
        r = x.norm(dim=-1)
        return torch.exp(-((r - 0.5) ** 2) / (2 * 0.1**2))

    chi = chi_hat(f, 2, [0.3, 0.5, 0.7], N2)
    for c in chi.tolist():
        assert abs(c) < 0.15, f"annulus: chi={c:.3f}, want 0"


def test_3d_one_ball():
    """Solid ball: chi = 1."""
    f = lambda x: gauss(x, [0.0, 0.0, 0.0], 0.35)  # noqa: E731
    chi = chi_hat(f, 3, [0.3, 0.6], N3)
    for c in chi.tolist():
        assert abs(c - 1.0) < 0.15, f"ball: chi={c:.3f}, want 1"


def test_3d_two_balls():
    f = lambda x: (gauss(x, [-0.45, 0, 0], 0.2)  # noqa: E731
                   + gauss(x, [0.45, 0, 0], 0.2))
    chi = chi_hat(f, 3, [0.4], N3)
    assert abs(chi[0] - 2.0) < 0.3, f"two balls: chi={chi[0]:.3f}, want 2"


def test_3d_solid_torus():
    """Tube around a circle of radius R: superlevel sets are solid tori,
    chi = 0. The decisive 3D topology test."""
    def f(x):
        rho = x[:, :2].norm(dim=-1)
        d2 = (rho - 0.55) ** 2 + x[:, 2] ** 2
        return torch.exp(-d2 / (2 * 0.12**2))

    chi = chi_hat(f, 3, [0.3, 0.6], N3)
    for c in chi.tolist():
        assert abs(c) < 0.25, f"solid torus: chi={c:.3f}, want 0"


def test_gradient_flow():
    """The chi-profile loss must backprop into network parameters."""
    from kacrice.minkowski import EulerProfileLoss
    from kacrice.models import PEMLP

    torch.manual_seed(0)
    net = PEMLP(2, 1, 64, 2, n_freqs=4)
    pts = torch.rand(4096, 2) * 2 - 1
    v, g, h = field_derivatives(net, pts)
    loss = EulerProfileLoss([0.0], [1.0], eps=0.05, volume=4.0,
                            mode="match")(v, g, h)
    loss.backward()
    gn = sum(p.grad.norm().item() for p in net.parameters()
             if p.grad is not None)
    assert torch.isfinite(loss).item() and gn > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {name}: {type(e).__name__}: {e}")
