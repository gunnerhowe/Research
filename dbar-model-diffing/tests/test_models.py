"""Model mechanics: manual GRU step = torch fused GRU; recoding preserves function;
pruning masks hold; generation is reproducible."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dbar_diff import models as M
from dbar_diff.tasks import golden_mean


def _model(width=32, seed=0):
    torch.manual_seed(seed)
    return M.GRULM(2, width).to(M.DEVICE)


def test_gru_step_matches_fused():
    model = _model()
    x = torch.randint(0, 2, (8, 20), device=M.DEVICE)
    logits_fused, states_fused = model(x)
    _, states_manual, _ = M.collect_states(model, x, sigma=0.0)
    assert torch.allclose(states_fused, states_manual, atol=1e-5)


def test_recode_permute_preserves_function():
    model = _model(seed=1)
    twin = M.recode_permute(model, seed=7)
    x = torch.randint(0, 2, (8, 50), device=M.DEVICE)
    la, _ = model(x)
    lb, _ = twin(x)
    assert torch.allclose(la, lb, atol=1e-4)
    # but the hidden states themselves are permuted, not equal
    _, sa, _ = M.collect_states(model, x)
    _, sb, _ = M.collect_states(twin, x)
    assert not torch.allclose(sa, sb, atol=1e-3)


def test_generate_reproducible_and_shapes():
    model = _model(seed=2)
    s1, b1 = M.generate(model, B=4, T=100, burn=10, seed=5)
    s2, _ = M.generate(model, B=4, T=100, burn=10, seed=5)
    s3, _ = M.generate(model, B=4, T=100, burn=10, seed=6)
    assert s1.shape == (4, 100) and b1.shape == (4, 100, 2)
    assert (s1 == s2).all()
    assert not (s1 == s3).all()


def test_noise_changes_process():
    model = _model(seed=3)
    s0, _ = M.generate(model, B=4, T=200, burn=10, seed=5, record_belief=False)
    s1, _ = M.generate(model, B=4, T=200, burn=10, seed=5, sigma=0.5,
                       record_belief=False)
    assert not (s0 == s1).all()


def test_prune_masks_hold():
    gm = golden_mean()
    model = _model(seed=4)
    twin = M.prune_finetune(model, gm, frac=0.5, steps=20)
    zeros = sum((p == 0).sum().item() for p in
                (twin.rnn.weight_ih_l0, twin.rnn.weight_hh_l0, twin.head.weight))
    total = sum(p.numel() for p in
                (twin.rnn.weight_ih_l0, twin.rnn.weight_hh_l0, twin.head.weight))
    assert zeros / total > 0.45


def test_transformer_forward_and_generate():
    gm = golden_mean()
    torch.manual_seed(0)
    tf = M.TransformerLM(2, d=32, n_layers=1, heads=2, ctx=64).to(M.DEVICE)
    x = torch.randint(0, 2, (4, 63), device=M.DEVICE)
    logits, states = tf(x)
    assert logits.shape == (4, 63, 2) and states.shape == (4, 63, 32)
    syms, beliefs = M.generate_transformer(tf, gm, B=2, T=32, burn=4, window=63)
    assert syms.shape == (2, 32) and beliefs.shape == (2, 32, 2)
