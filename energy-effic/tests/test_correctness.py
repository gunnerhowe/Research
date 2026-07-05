"""Correctness tests: estimator vs closed forms, Rice on known spectra,
sign/normalization conventions, gradient flow, and delta-cell faithfulness
to the Neil et al. 2016 mechanism.

Run: python -m pytest tests/ -q
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eventrice.budget import TemporalCrossingBudget
from eventrice.delta import (DeltaGRU, GRUClassifier, TransformerLM,
                             delta_encode_trace)
from eventrice.energy import gru_stats_dense_energy_pj, gru_stats_energy_pj
from eventrice.estimator import (crossing_count_hard, crossing_rate_midpoint,
                                 crossing_rate_segment, make_levels)
from eventrice.rice import (gaussian_discrete_rate, iid_rate, rice_rate,
                            spectral_moments)


# ---------------------------------------------------------------- estimator


def test_sine_crossing_rate_convention():
    """a(t) = sin(2 pi f t): rate at level 0 is 2f per unit time (up+down)."""
    f = 3.0
    t = torch.linspace(0, 10, 4001)
    tr = torch.sin(2 * math.pi * f * t).unsqueeze(0)
    for fn in (crossing_count_hard, lambda a, l, dt: crossing_rate_segment(a, l, 0.01, dt)):
        c = fn(tr, torch.tensor([0.0]), t[1].item() - t[0].item())
        assert abs(c.item() - 2 * f) / (2 * f) < 0.02, c


def test_multisine_exact_counts():
    """Deterministic multisine: segment estimator (small eps) and hard counts
    agree with a dense-grid reference count at several levels."""
    t = torch.linspace(0, 20, 20001)
    dt = t[1].item() - t[0].item()
    tr = (torch.sin(2 * math.pi * 1.3 * t) + 0.6 * torch.sin(2 * math.pi * 3.7 * t + 1.0)
          + 0.3 * torch.sin(2 * math.pi * 0.4 * t + 2.0)).unsqueeze(0)
    levels = torch.tensor([-0.5, 0.0, 0.7])
    hard = crossing_count_hard(tr, levels, dt)
    seg = crossing_rate_segment(tr, levels, eps=0.005, dt=dt)
    for h, s in zip(hard, seg):
        assert abs(h - s) / h < 0.03, (hard, seg)


def test_segment_converges_to_hard_count():
    """eps -> 0: smoothed segment rate converges to the sign-change count.
    (Exact up to segments whose endpoints lie within O(eps) of a level —
    quantile levels sit ON data values, so a residual of ~0.5 counts per
    coinciding sample is expected and bounded here.)"""
    torch.manual_seed(0)
    tr = torch.randn(8, 400).cumsum(dim=1) * 0.1 + torch.randn(8, 400)
    levels = make_levels(tr, 5)
    hard = crossing_count_hard(tr, levels)
    err_prev = None
    for eps in (1e-2, 1e-3, 1e-4):
        seg = crossing_rate_segment(tr, levels, eps=eps)
        err = (seg - hard).abs().max().item()
        assert err < 5e-3, (eps, seg, hard)
        if err_prev is not None:
            assert err <= err_prev + 1e-4
        err_prev = err


def test_rice_random_phase_sinusoids():
    """Sum of random-phase sinusoids: lambda0 = sum a^2/2, lambda2 = sum a^2 w^2/2.
    Rice from THEORETICAL moments vs dense hard counts, <=3% at several levels."""
    rng = np.random.default_rng(0)
    m = 80
    w = rng.uniform(4.0, 40.0, m)
    a = rng.normal(0, 1, m) * 0.2
    ph = rng.uniform(0, 2 * math.pi, m)
    t = np.linspace(0, 400, 1_600_001)
    tr = sum(ai * np.sin(wi * t + pi) for ai, wi, pi in zip(a, w, ph))
    lam0 = float(np.sum(a ** 2) / 2)
    lam2 = float(np.sum(a ** 2 * w ** 2) / 2)
    sig = math.sqrt(lam0)
    levels = np.array([0.0, 0.5 * sig, 1.0 * sig])
    pred = rice_rate(levels, 0.0, lam0, lam2)
    hard = crossing_count_hard(torch.from_numpy(tr).unsqueeze(0),
                               torch.from_numpy(levels), dt=t[1] - t[0]).numpy()
    rel = np.abs(pred - hard) / hard
    assert rel.max() < 0.03, (pred, hard, rel)


def test_spectral_moment_fitting():
    """Fitted (FD) moments recover theoretical ones on a densely sampled GP."""
    rng = np.random.default_rng(1)
    m = 60
    w = rng.uniform(2.0, 20.0, m)
    a = np.abs(rng.normal(0, 0.3, m))
    ph = rng.uniform(0, 2 * math.pi, m)
    t = np.linspace(0, 600, 2_400_001)
    tr = sum(ai * np.sin(wi * t + pi) for ai, wi, pi in zip(a, w, ph))
    mom = spectral_moments(tr[None, :], dt=t[1] - t[0])
    lam0_th = np.sum(a ** 2) / 2
    lam2_th = np.sum(a ** 2 * w ** 2) / 2
    assert abs(mom["lam0"][0] - lam0_th) / lam0_th < 0.05
    assert abs(mom["lam2"][0] - lam2_th) / lam2_th < 0.05


def test_discrete_gaussian_rate_identities():
    """u=0: arccos(rho)/pi. rho=0: matches the iid baseline formula."""
    for rho in (0.0, 0.5, 0.9):
        r = gaussian_discrete_rate(np.array([0.0]), 0.0, 1.0, rho)
        assert abs(r[0] - math.acos(rho) / math.pi) < 1e-9
    lv = np.array([-1.0, 0.3, 2.0])
    assert np.allclose(gaussian_discrete_rate(lv, 0.0, 1.0, 0.0),
                       iid_rate(lv, 0.0, 1.0), atol=1e-9)


def test_discrete_gaussian_rate_vs_ar1_simulation():
    """AR(1) sequence: measured crossing rates match the Owen's-T closed form
    within MC error, at nonzero levels (where continuous Rice underpredicts)."""
    rng = np.random.default_rng(2)
    rho = 0.8
    n = 4_000_000
    e = rng.normal(0, math.sqrt(1 - rho ** 2), n)
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    levels = np.array([0.0, 1.0])
    meas = crossing_count_hard(torch.from_numpy(x).unsqueeze(0),
                               torch.from_numpy(levels)).numpy()
    pred = gaussian_discrete_rate(levels, 0.0, 1.0, rho)
    assert np.abs(pred - meas).max() / meas.max() < 0.03, (pred, meas)


# ------------------------------------------------------------- grad flow


def test_budget_gradient_flow_and_one_sidedness():
    """Budget backprops finite nonzero grads into GRU params when over budget,
    and contributes EXACTLY zero loss and grad when within budget."""
    torch.manual_seed(0)
    model = GRUClassifier(8, 16, 1, 4)
    x = torch.randn(4, 50, 8)
    _, traces = model(x, return_traces=True)
    tr = traces[0]
    levels = make_levels(tr.detach(), 8)
    # over budget: budgets at 10% of measured profile
    with torch.no_grad():
        prof = crossing_count_hard(tr.transpose(1, 2).detach(), levels)
    b_over = TemporalCrossingBudget(levels, 0.1 * prof, eps=0.1)
    loss = b_over(tr.transpose(1, 2))
    loss.backward(retain_graph=True)
    gnorm = sum(p.grad.norm().item() for p in model.parameters()
                if p.grad is not None)
    assert math.isfinite(loss.item()) and loss.item() > 0 and gnorm > 0
    # within budget: budgets at 10x measured -> zero loss, zero grad
    model.zero_grad()
    b_under = TemporalCrossingBudget(levels, 10.0 * prof + 1.0, eps=0.1)
    loss2 = b_under(tr.transpose(1, 2))
    loss2.backward()
    gnorm2 = sum(p.grad.norm().item() for p in model.parameters()
                 if p.grad is not None)
    assert loss2.item() == 0.0 and gnorm2 == 0.0


# ------------------------------------------------------- delta faithfulness


def test_delta_encoder_hand_example():
    """Neil et al. semantics on a hand-crafted trace: threshold on
    |a_t - a_hat_last|, anchor updates to the CURRENT value on event."""
    tr = torch.tensor([[0.0, 0.3, 0.5, 1.2, 1.0, 1.0]]).unsqueeze(-1)
    anchors, events = delta_encode_trace(tr, theta=0.4)
    assert events.squeeze().tolist() == [False, False, True, True, False, False]
    assert torch.allclose(anchors.squeeze(),
                          torch.tensor([0.0, 0.0, 0.5, 1.2, 1.2, 1.2]))


def test_delta_gru_theta_zero_matches_dense_gru():
    """theta = 0 must reproduce torch.nn.GRU exactly (same weights)."""
    torch.manual_seed(0)
    model = GRUClassifier(12, 24, 2, 5)
    x = torch.randn(3, 40, 12)
    _, traces = model(x, return_traces=True)
    delta = model.as_delta(0.0)
    out, stats = delta(x)
    assert torch.allclose(out, traces[-1], atol=1e-5), (
        (out - traces[-1]).abs().max())
    # dense mode: every component fires every step
    for s in stats:
        assert s["events_x"] == s["dense_x"] and s["events_h"] == s["dense_h"]


def test_delta_gru_layer0_input_events_match_encoder():
    """Layer-0 INPUT events have no feedback: they must equal the standalone
    send-on-delta encoder run on the raw input trace."""
    torch.manual_seed(1)
    model = GRUClassifier(6, 12, 1, 3)
    x = torch.randn(4, 30, 6).cumsum(dim=1) * 0.3
    theta = 0.5
    delta = model.as_delta(theta)
    _, stats = delta(x)
    _, ev = delta_encode_trace(x, theta)
    assert stats[0]["events_x"] == ev.float().sum().item()


def test_delta_gru_events_decrease_with_theta():
    torch.manual_seed(2)
    model = GRUClassifier(6, 12, 2, 3)
    x = torch.randn(4, 30, 6)
    ev = []
    for th in (0.0, 0.1, 0.5, 2.0):
        _, stats = model.as_delta(th)(x)
        ev.append(sum(s["events_x"] + s["events_h"] for s in stats))
    assert ev[0] > ev[1] > ev[2] > ev[3] >= 0


def test_transformer_delta_theta_zero_and_events():
    torch.manual_seed(0)
    lm = TransformerLM(50, dim=32, n_layers=1, n_heads=2, ffn=64, seq_len=16)
    toks = torch.randint(0, 50, (2, 16))
    lm.set_thetas(0.0)
    with torch.no_grad():
        y0 = lm(toks)
    lm.set_thetas(1e-9)
    lm.reset_event_counts()
    with torch.no_grad():
        y1 = lm(toks)
    assert torch.allclose(y0, y1, atol=1e-4)
    lm.set_thetas(0.5)
    lm.reset_event_counts()
    with torch.no_grad():
        lm(toks)
    st = lm.event_stats()
    assert all(s["events_x"] <= s["dense_x"] for s in st)
    assert sum(s["events_x"] for s in st) < sum(s["dense_x"] for s in st)


def test_energy_accounting():
    torch.manual_seed(3)
    model = GRUClassifier(6, 12, 2, 3)
    x = torch.randn(4, 30, 6)
    _, stats = model.as_delta(0.4)(x)
    assert 0 < gru_stats_energy_pj(stats) < gru_stats_dense_energy_pj(stats)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {name}: {e}")
