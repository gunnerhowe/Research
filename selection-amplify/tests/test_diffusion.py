import numpy as np

from selamp.diffusion import Diffusion


def _gaussian_blobs(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    mu = np.where(y[:, None] == 0, np.array([-2.0, 0]), np.array([2.0, 0]))
    X = mu + 0.3 * rng.standard_normal((n, 2))
    return X.astype(np.float64), y


def test_samples_match_moments():
    X, y = _gaussian_blobs(3000, 0)
    dm = Diffusion(T=100).fit(X, y, epochs=1500, seed=0)
    s0 = dm.sample(1500, 0, seed=1)
    # class-0 samples cluster near (-2, 0)
    assert np.linalg.norm(s0.mean(0) - np.array([-2.0, 0.0])) < 0.5


def test_class_conditioning_separates():
    X, y = _gaussian_blobs(3000, 0)
    dm = Diffusion(T=100).fit(X, y, epochs=1500, seed=0)
    s0 = dm.sample(1000, 0, seed=1).mean(0)
    s1 = dm.sample(1000, 1, seed=1).mean(0)
    assert s1[0] - s0[0] > 2.0             # class 1 to the right of class 0


def test_tweedie_denoises_at_low_noise():
    import torch
    X, y = _gaussian_blobs(2000, 0)
    dm = Diffusion(T=100).fit(X, y, epochs=1000, seed=0)
    z0 = dm.norm(torch.tensor(X[:200], dtype=torch.float32, device=dm.device))
    yb = torch.zeros(200, dtype=torch.long, device=dm.device)
    i = 3                                   # low-noise index
    abar = dm.abar[i]
    eps = torch.randn_like(z0)
    zt = torch.sqrt(abar) * z0 + torch.sqrt(1 - abar) * eps
    with torch.no_grad():
        x0 = dm.tweedie_x0(zt, i, dm.eps(zt, i, yb))
    assert (x0 - z0).abs().mean().item() < 0.3
