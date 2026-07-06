"""Base generator: a small class-conditional DDPM (discrete VP diffusion)
trained on the curated corpus D_obs. eps-prediction, cosine schedule; the
model works in a normalized coordinate frame, and exposes the Tweedie denoiser
x_hat_0 and the schedule so bridge.py can run reconstruction-guided sampling.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _cosine_abar(T, s=0.008):
    t = np.linspace(0, T, T + 1)
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    abar = f / f[0]
    betas = np.clip(1 - abar[1:] / abar[:-1], 1e-5, 0.999)
    return betas


def _time_embed(t, dim, device):
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, device=device) / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class _ScoreMLP(nn.Module):
    def __init__(self, d=2, n_classes=2, tdim=64, cdim=32, hidden=128, depth=3):
        super().__init__()
        self.tdim = tdim
        self.cls = nn.Embedding(n_classes, cdim)
        layers, di = [], d + tdim + cdim
        for _ in range(depth):
            layers += [nn.Linear(di, hidden), nn.SiLU()]
            di = hidden
        layers += [nn.Linear(di, d)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, y):
        te = _time_embed(t, self.tdim, x.device)
        ce = self.cls(y)
        return self.net(torch.cat([x, te, ce], -1))


class Diffusion:
    def __init__(self, T=200, n_classes=2, hidden=128, depth=3, device=DEVICE):
        self.T = T
        self.device = device
        self.n_classes = n_classes
        betas = torch.tensor(_cosine_abar(T), dtype=torch.float32, device=device)
        self.betas = betas
        self.alphas = 1 - betas
        self.abar = torch.cumprod(self.alphas, 0)
        self.model = _ScoreMLP(2, n_classes, hidden=hidden, depth=depth).to(device)
        self._mu = None
        self._sd = None

    # ------------------------------------------------------ normalization
    def _fit_norm(self, X):
        self._mu = torch.tensor(X.mean(0), dtype=torch.float32, device=self.device)
        self._sd = torch.tensor(X.std(0) + 1e-6, dtype=torch.float32,
                                device=self.device)

    def norm(self, X):
        return (X - self._mu) / self._sd

    def denorm(self, Z):
        return Z * self._sd + self._mu

    # -------------------------------------------------------------- train
    def fit(self, X, y, epochs=2000, batch=256, lr=2e-3, seed=0):
        torch.manual_seed(seed)
        self._fit_norm(X)
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        Zt = self.norm(Xt)
        yt = torch.tensor(y, dtype=torch.long, device=self.device)
        n = len(Zt)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.model.train()
        for _ in range(epochs):
            idx = torch.randint(n, (min(batch, n),), generator=g)
            z0 = Zt[idx.to(self.device)]
            yb = yt[idx.to(self.device)]
            t = torch.randint(0, self.T, (len(z0),), device=self.device)
            abar_t = self.abar[t][:, None]
            eps = torch.randn_like(z0)
            zt = torch.sqrt(abar_t) * z0 + torch.sqrt(1 - abar_t) * eps
            pred = self.model(zt, t, yb)
            loss = ((pred - eps) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        self.model.eval()
        return self

    # -------------------------------------------------- eps / score / tweedie
    def eps(self, z, t_idx, y):
        t = torch.full((len(z),), t_idx, device=self.device, dtype=torch.long)
        return self.model(z, t, y)

    def tweedie_x0(self, z, t_idx, eps):
        abar_t = self.abar[t_idx]
        return (z - torch.sqrt(1 - abar_t) * eps) / torch.sqrt(abar_t)

    # ----------------------------------------------------- plain sampling
    @torch.no_grad()
    def sample(self, n, y_class, temperature=1.0, seed=0):
        """Unconditional (unguided) ancestral sampling for class y_class.
        temperature>1 broadens the reverse noise (the B2-diversity knob)."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        z = torch.randn(n, 2, device=self.device, generator=g)
        y = torch.full((n,), y_class, device=self.device, dtype=torch.long)
        for i in reversed(range(self.T)):
            eps = self.eps(z, i, y)
            z = self._ddpm_step(z, i, eps, g, temperature)
        return self.denorm(z).cpu().numpy()

    def _ddpm_step(self, z, i, eps, g, temperature=1.0):
        beta, alpha, abar = self.betas[i], self.alphas[i], self.abar[i]
        mean = (z - beta / torch.sqrt(1 - abar) * eps) / torch.sqrt(alpha)
        if i > 0:
            noise = torch.randn(z.shape, device=self.device, generator=g)
            return mean + temperature * torch.sqrt(beta) * noise
        return mean
