"""Correctness tests for the Kac-Rice estimator and baseline losses.

Run:  python -m pytest tests -q   (or plain `python tests/test_correctness.py`)
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kacrice.crossing import crossing_density, field_and_grad, KacRiceLoss
from kacrice.losses import FocalFrequencyLoss
from kacrice.models import SIREN


def test_1d_sine_crossing_count():
    """f(x) = sin(pi k x) on [-1,1] has 2k zeros over length 2 -> density k."""
    k = 7
    n = 200_000
    x = torch.rand(n).mul(2).sub(1)
    f = torch.sin(math.pi * k * x)
    df = math.pi * k * torch.cos(math.pi * k * x)
    levels = torch.tensor([0.0])
    c = crossing_density(f, df.abs(), levels, eps=0.02)
    assert abs(c.item() - k) / k < 0.05, f"expected ~{k}, got {c.item():.3f}"


def test_1d_nonzero_level():
    """Crossings of sin(pi k x) at level u: density k * (2/pi) * ... actually
    for u in (-1,1) each period crosses level u exactly twice -> density still k."""
    k = 5
    n = 200_000
    x = torch.rand(n).mul(2).sub(1)
    f = torch.sin(math.pi * k * x)
    df = math.pi * k * torch.cos(math.pi * k * x)
    for u in (0.3, -0.6):
        c = crossing_density(f, df.abs(), torch.tensor([u]), eps=0.02)
        assert abs(c.item() - k) / k < 0.07, f"level {u}: expected ~{k}, got {c.item():.3f}"


def _random_field(seed=0, m=60):
    """Approximate stationary GP: many random sinusoids. Frequencies are kept
    high (>= 20 rad/unit) so the [-1,1] window holds many cycles per component;
    with low frequencies the window average diverges from ensemble moments
    (ergodicity gap) and Rice cannot match."""
    torch.manual_seed(seed)
    freqs = torch.rand(m) * 100 + 20
    amps = torch.randn(m).abs() * 0.3
    phases = torch.rand(m) * 2 * math.pi

    def f(x):
        return sum(a * torch.sin(w * x + p) for a, w, p in zip(amps, freqs, phases))

    def df(x):
        return sum(
            a * w * torch.cos(w * x + p) for a, w, p in zip(amps, freqs, phases)
        )

    return f, df


def test_estimator_matches_direct_crossing_count():
    """Ground truth independent of any Gaussian assumption: count sign changes
    of f - u on a dense grid and compare with the smoothed MC estimator."""
    f, df = _random_field(seed=1)
    xg = torch.linspace(-1, 1, 2_000_000)
    fg = f(xg)
    x = torch.rand(400_000).mul(2).sub(1)
    for u in (0.0, 0.4):
        true_density = ((fg[:-1] - u) * (fg[1:] - u) < 0).sum().item() / 2.0
        c = crossing_density(f(x), df(x).abs(), torch.tensor([u]), eps=0.05).item()
        assert abs(c - true_density) / true_density < 0.05, (
            f"level {u}: counted {true_density:.2f}/unit vs MC {c:.2f}/unit"
        )


def test_rice_formula_gaussian_process():
    """MC crossing density matches Rice with *empirical* spectral moments:
    mu(u) = (1/pi) sqrt(l2/l0) exp(-u^2/(2 l0)), l0 = Var f, l2 = Var f'."""
    f, df = _random_field(seed=0)
    x = torch.rand(400_000).mul(2).sub(1)
    fv, dv = f(x), df(x)
    l0 = fv.var().item()
    l2 = dv.var().item()

    for u in (0.0, 0.5):
        rice = (1 / math.pi) * math.sqrt(l2 / l0) * math.exp(-(u**2) / (2 * l0))
        c = crossing_density(fv, dv.abs(), torch.tensor([u]), eps=0.05).item()
        assert abs(c - rice) / rice < 0.1, f"level {u}: Rice {rice:.3f} vs MC {c:.3f}"


def test_2d_coarea_line_density():
    """f(x,y) = sin(pi k x) on [-1,1]^2: level set {f=0} is 2k vertical lines of
    length 2 in area 4 -> length density k. Checks the 2D co-area generalization."""
    k = 6
    n = 400_000
    pts = torch.rand(n, 2).mul(2).sub(1)
    f = torch.sin(math.pi * k * pts[:, 0])
    grad = torch.stack(
        [math.pi * k * torch.cos(math.pi * k * pts[:, 0]), torch.zeros(n)], dim=-1
    )
    c = crossing_density(f, grad.norm(dim=-1), torch.tensor([0.0]), eps=0.02)
    assert abs(c.item() - k) / k < 0.05, f"expected ~{k}, got {c.item():.3f}"


def test_kacrice_loss_grad_flow():
    """Loss must backprop finite, nonzero gradients into INR parameters."""
    torch.manual_seed(0)
    model = SIREN(in_features=1, hidden_features=32, hidden_layers=1)
    pts = torch.rand(512, 1).mul(2).sub(1)
    gt = torch.sin(math.pi * 9 * pts[:, 0])
    gt_grad = (math.pi * 9 * torch.cos(math.pi * 9 * pts[:, 0])).unsqueeze(-1)

    pred, pgrad = field_and_grad(model, pts)
    loss = KacRiceLoss(n_levels=8)(pred, pgrad, gt, gt_grad)
    loss.backward()
    gnorm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    assert math.isfinite(loss.item()) and gnorm > 0


def test_ffl_matches_official_package():
    """Local FFL must match the official focal-frequency-loss implementation."""
    try:
        from focal_frequency_loss import FocalFrequencyLoss as OfficialFFL
    except ImportError:
        import pytest

        pytest.skip("official focal-frequency-loss package not installed")

    torch.manual_seed(0)
    a, b = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
    ours = FocalFrequencyLoss()(a, b)
    official = OfficialFFL()(a, b)
    assert torch.allclose(ours, official, rtol=1e-4), f"{ours} vs {official}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {name}: {e}")
