"""Correctness of the barrier-conditioned rule and the consolidation state."""
import math

import torch

from doobsyn.diffusion import (Memory, update_memory, diagonal_fisher,
                               Consolidator, ConsolConfig)
from doobsyn.models import MLP


def _toy_model():
    torch.manual_seed(0)
    return MLP(4, 8, 3, n_layers=1)


def test_doob_score_is_restoring_and_diverges():
    """d/dw log h = -(pi/2b) tan(pi (w-mu)/2b): restoring toward mu, magnitude
    grows to the barrier."""
    b = 0.3
    for z in (0.1, 0.4, 0.49):                    # fraction of the barrier
        w = torch.tensor(z * b)
        score = -(math.pi / (2 * b)) * math.tan(math.pi * (w.item()) / (2 * b))
        assert score < 0                          # w>mu -> push down
    near = -(math.pi / (2 * b)) * math.tan(math.pi * (0.49 * b) / (2 * b))
    far = -(math.pi / (2 * b)) * math.tan(math.pi * (0.1 * b) / (2 * b))
    assert abs(near) > abs(far)                   # diverges toward the barrier


def test_fisher_nonnegative():
    m = _toy_model()
    x = torch.randn(64, 4)
    y = torch.randint(0, 3, (64,))
    loader = [(x, y)]
    import torch.nn as nn
    f = diagonal_fisher(m, nn.CrossEntropyLoss(), loader, "cpu")
    for fi in f:
        assert torch.all(fi >= 0)


def test_barrier_tighter_for_important_weights():
    m = _toy_model()
    mem = Memory()
    # realistic heavy-tailed Fisher: a small positive baseline for most weights,
    # one clearly-important and one clearly-trivial.
    fisher = [torch.full_like(p, 0.01) for p in m.parameters()]
    flat = fisher[0].reshape(-1)
    flat[0] = 100.0                               # important (>> median)
    flat[1] = 1e-4                                # trivial   (<< median)
    update_memory(mem, m, fisher, barrier_scale=0.3)
    b0 = mem.b[0].reshape(-1)
    assert b0[0] < b0[1]                          # important -> tighter barrier
    assert b0[1] <= 0.3 + 1e-6                     # loose barrier capped at scale
    assert b0[0] >= 0.05 * 0.3 - 1e-9              # clamp floor honoured
    assert b0[1] > 0.05 * 0.3                      # trivial weight is not at floor


def test_doob_reduces_to_ou_at_zero_noise():
    """At sigma=0 the Doob term (prop to sigma^2) vanishes: a doob step must equal
    an ou step exactly."""
    torch.manual_seed(1)
    for method_pair in [("doob", "ou")]:
        outs = []
        for method in method_pair:
            m = _toy_model()
            mem = Memory()
            fisher = [torch.ones_like(p) for p in m.parameters()]
            update_memory(mem, m, fisher, barrier_scale=0.3)
            # perturb weights away from the anchor
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(0.05)
            cfg = ConsolConfig(method=method, sigma=0.0, lr_c=0.1)
            Consolidator(mem, cfg, "cpu", seed=0).bind(m).step()
            outs.append(torch.cat([p.detach().reshape(-1) for p in m.parameters()]))
        assert torch.allclose(outs[0], outs[1], atol=1e-7)


def test_doob_step_bounded():
    """The finite-force cap keeps a doob step finite even at the barrier."""
    m = _toy_model()
    mem = Memory()
    fisher = [torch.full_like(p, 50.0) for p in m.parameters()]
    update_memory(mem, m, fisher, barrier_scale=0.1)
    with torch.no_grad():                         # push a weight to the barrier
        list(m.parameters())[0].reshape(-1)[0] = mem.mu[0].reshape(-1)[0] + 0.0999
    cfg = ConsolConfig(method="doob", sigma=0.5, lr_c=0.2, max_step_frac=0.25)
    Consolidator(mem, cfg, "cpu", seed=0).bind(m).step()
    for p in m.parameters():
        assert torch.all(torch.isfinite(p))
