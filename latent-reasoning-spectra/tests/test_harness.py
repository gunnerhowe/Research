"""GPU tests of the model harness (skipped if no CUDA)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrspec.paths import DATA  # noqa: E402
from lrspec.prosqa import load_problems  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def problems():
    return load_problems(DATA / "prosqa_test.json")


@pytest.fixture(scope="module")
def m2():
    from lrspec.harness import Harness

    return Harness("M2")


@cuda
def test_step_self_consistency(m2, problems):
    run = m2.run_latent(problems[1])
    for t in (1, 3, 5):
        f = m2.step_function(run, t)
        err = (f(m2.fed_vector(run, t)) - run.hs[t]).norm() / run.hs[t].norm()
        assert err.item() < 1e-5


@cuda
def test_jacobian_fd(m2, problems):
    run = m2.run_latent(problems[2])
    J = m2.jacobian(run, 2)
    f = m2.step_function(run, 2)
    c = m2.fed_vector(run, 2)
    eps = 1e-2
    rng = torch.Generator(device="cpu").manual_seed(0)
    d = torch.randn(768, generator=rng).to(c.device)
    d = d / d.norm()
    fd = (f(c + eps * d) - f(c - eps * d)) / (2 * eps)
    jvp = J @ d
    assert (fd - jvp).norm() / jvp.norm() < 0.05


@cuda
def test_unrolled_consistency(m2, problems):
    """g(c_t) with k steps must reproduce c_{t+k} exactly on the realized orbit."""
    run = m2.run_latent(problems[3])
    g = m2.unrolled_function(run, 2, 3)
    out = g(m2.fed_vector(run, 2))
    err = (out - run.hs[4]).norm() / run.hs[4].norm()
    assert err.item() < 1e-4


@cuda
def test_rerun_identity(m2, problems):
    """rerun_from with the unmodified thought must reproduce the base margin."""
    p = problems[4]
    run = m2.run_latent(p)
    base = m2.readout(run, p)
    r = m2.rerun_from(run, p, 3, m2.fed_vector(run, 3))
    assert abs(r["margin"] - base["margin"]) < 1e-3


@cuda
def test_pause_multipass_equals_single_pass(problems):
    """M4's multi-pass inference is numerically identical to a single causal pass,
    so running M4 through the same incremental harness as M3 is exact."""
    from lrspec.harness import Harness

    h4 = Harness("M4")
    p = problems[5]
    ids, slot_pos, start_pos, end_pos = h4.encode(p)
    # single full pass with pause embeddings at latent slots
    emb = h4.emb(torch.tensor([ids], device=h4.device)).squeeze(0)
    for sp in slot_pos:
        emb[sp] = h4.pause_embedding
    with torch.no_grad():
        out = h4.model(
            inputs_embeds=emb.unsqueeze(0),
            position_ids=torch.arange(len(ids), device=h4.device).view(1, -1),
            output_hidden_states=True,
        )
    h_single = out.hidden_states[-1][0]
    run = h4.run_latent(p)
    for i, sp in enumerate(slot_pos):
        err = (h_single[sp] - run.hs[i + 1]).norm() / run.hs[i + 1].norm()
        assert err.item() < 1e-4, (i, err.item())
