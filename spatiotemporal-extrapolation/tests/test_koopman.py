"""Learned-operator plumbing: equivariance, frequency response, spectrum extraction."""
import numpy as np
import torch

from specext.koopman import ConvKoopman, freq_response, model_spectrum, DT_STEP


def test_freq_response_identity_and_shift():
    M, kw = 4, 33
    w = np.zeros((M, M, kw))
    w[np.arange(M), np.arange(M), kw // 2] = 1.0     # identity
    kap = np.array([0.0, 0.3, 1.0])
    K = freq_response(w, kap)
    for i in range(len(kap)):
        np.testing.assert_allclose(K[i], np.eye(M), atol=1e-12)
    w2 = np.zeros((1, 1, kw))
    w2[0, 0, kw // 2 + 1] = 1.0                       # shift by one site
    K2 = freq_response(w2, kap)
    np.testing.assert_allclose(K2[:, 0, 0], np.exp(1j * kap), atol=1e-12)


def test_encoder_equivariance():
    torch.manual_seed(0)
    model = ConvKoopman().eval()
    u = torch.randn(1, 64)
    z = model.encode(u)
    z_shift = model.encode(torch.roll(u, 5, dims=-1))
    np.testing.assert_allclose(z_shift.detach().numpy(),
                               torch.roll(z, 5, dims=-1).detach().numpy(),
                               atol=1e-5)


def test_model_spectrum_runs_and_selects_stable():
    torch.manual_seed(1)
    model = ConvKoopman().eval()
    m_idx = np.array([1, 2, 5])
    out = model_spectrum(model, ell=1.0, m_idx=m_idx, N=64, device="cpu")
    assert out["mu"].shape == (3, 16)
    lam = out["lam"]
    ok = np.isfinite(lam.real)
    # every selected eigenvalue obeys the stability cap used in the selector
    assert (np.abs(out["mu_sel"][ok]) <= 1.005 + 1e-9).all()
    assert np.isfinite(out["weights"]).all()


def test_propagator_flow_conditioning():
    torch.manual_seed(2)
    model = ConvKoopman(flow=True)
    with torch.no_grad():
        model.w1[:] = torch.randn_like(model.w1)
    k1 = model.kernel(1.0)
    k2 = model.kernel(0.25)
    assert not torch.allclose(k1, k2)
    np.testing.assert_allclose((k1 - k2).detach().numpy(),
                               (0.75 * model.w1).detach().numpy(), atol=1e-6)
