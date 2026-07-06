"""The forward-noise (hardware-faithful) realization and the stability clamp."""
import numpy as np

from doobsyn.data import yin_yang
from doobsyn.hwloop import run_forward_sequence, NoisyMLP
import torch


def test_forward_noise_only_when_training():
    m = NoisyMLP(4, 8, 3)
    m.fwd_noise = 0.5
    x = torch.randn(16, 4)
    m.train()
    a, b = m(x), m(x)
    assert not torch.allclose(a, b)          # noisy in train mode
    m.eval()
    assert torch.allclose(m(x), m(x))        # deterministic in eval mode


def test_forward_sequence_runs_and_is_bounded():
    tasks = yin_yang(n_tasks=3, n_train=300, n_test=200, seed=0)
    r, avg, plas = run_forward_sequence("yin_yang", tasks, method="doob", sigma=0.3,
                                        seed=0, epochs=1, device="cpu")
    for v in (r, avg, plas):
        assert 0.0 <= v <= 1.0


def test_clamp_prevents_blowup():
    """The importance clamp keeps the run finite; an unclamped heavy-tailed Fisher
    is the porting failure mode (weights blow up)."""
    tasks = yin_yang(n_tasks=3, n_train=300, n_test=200, seed=0)
    r, _, _ = run_forward_sequence("yin_yang", tasks, method="doob", sigma=0.8,
                                   seed=0, epochs=1, imp_clip=10.0, device="cpu")
    assert np.isfinite(r)


def test_doob_equals_ou_at_zero_noise_forward():
    tasks = yin_yang(n_tasks=3, n_train=300, n_test=200, seed=0)
    kw = dict(sigma=0.0, seed=0, epochs=1, device="cpu")
    rd = run_forward_sequence("yin_yang", tasks, method="doob", **kw)[0]
    ru = run_forward_sequence("yin_yang", tasks, method="ou", **kw)[0]
    assert abs(rd - ru) < 1e-9               # Doob term is off at sigma=0
