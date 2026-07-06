"""BSS-2 device-noise emulation properties and the honest hardware guard."""
import numpy as np
import pytest
import torch

from doobsyn.bss2 import (Bss2NoiseModel, Bss2NoiseParams, Bss2Backend,
                          make_noise_model)
from doobsyn.models import MLP


def test_colored_noise_has_temporal_autocorrelation():
    shapes = [(2000,)]
    p = Bss2NoiseParams(color=0.7, multiplicative=0.0, fixed_pattern=0.0)
    nm = Bss2NoiseModel(shapes, p, "cpu", seed=0)
    w = [torch.zeros(2000)]
    series = np.stack([nm.sample(w, 1.0)[0].numpy() for _ in range(400)])  # (T, N)
    # lag-1 autocorrelation over time, averaged across the 2000 units
    x = series - series.mean(0, keepdims=True)
    num = (x[1:] * x[:-1]).mean()
    den = (x * x).mean()
    ac1 = num / den
    assert ac1 > 0.4                              # clearly colored (rho=0.7)


def test_white_limit_has_no_autocorrelation():
    shapes = [(4000,)]
    p = Bss2NoiseParams(color=0.0, multiplicative=0.0, fixed_pattern=0.0)
    nm = Bss2NoiseModel(shapes, p, "cpu", seed=0)
    w = [torch.zeros(4000)]
    series = np.stack([nm.sample(w, 1.0)[0].numpy() for _ in range(300)])
    x = series - series.mean(0, keepdims=True)
    ac1 = (x[1:] * x[:-1]).mean() / (x * x).mean()
    assert abs(ac1) < 0.1


def test_quantize_snaps_to_grid():
    p = Bss2NoiseParams(weight_bits=6, weight_range=1.0)
    nm = Bss2NoiseModel([(1000,)], p, "cpu", seed=0)
    w = torch.linspace(-1.5, 1.5, 1000)
    q = nm.quantize(w)
    assert torch.all(q.abs() <= 1.0 + 1e-6)       # clamped to range
    assert torch.unique(q).numel() <= 2 ** 6      # at most 64 levels


def test_fixed_pattern_is_static():
    # fixed-pattern var 1.0 vs temporal var 1.0 -> theoretical corr ~0.5
    p = Bss2NoiseParams(color=0.0, multiplicative=0.0, fixed_pattern=1.0)
    nm = Bss2NoiseModel([(3000,)], p, "cpu", seed=3)
    w = [torch.zeros(3000)]
    a = nm.sample(w, 1.0)[0]
    b = nm.sample(w, 1.0)[0]
    # the fixed-pattern component is common to both draws -> positive correlation
    assert torch.corrcoef(torch.stack([a, b]))[0, 1] > 0.3


def test_hardware_backend_raises_without_stack():
    be = Bss2Backend()
    if not be.available:                          # this environment has no stack
        with pytest.raises(RuntimeError, match="pre-registered remaining step"):
            be.measure_retention_and_energy(None)


def test_make_noise_model_shapes():
    m = MLP(4, 8, 3, n_layers=1)
    nm = make_noise_model(m, Bss2NoiseParams(), "cpu")
    incs = nm.sample(list(m.parameters()), 0.1)
    for inc, p in zip(incs, m.parameters()):
        assert inc.shape == p.shape
