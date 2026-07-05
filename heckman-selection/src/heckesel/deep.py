"""Heckman selection models with neural-network feature maps (Paper A).

Architecture (per info.txt): deep outcome net f_theta(x), shallow/LINEAR
selection head g(w) on w = [x, z], plus two global parameters (log_sigma,
atanh_rho). Joint MLE minimizes the same bivariate-normal negative
log-likelihood as the classic estimator (selection.py), summed over ALL
units (selected and unselected).

Training detail (stability): atanh_rho is frozen at 0 for the first
`warmup` epochs so the outcome net first fits the selected-sample mean;
the correction term then separates the population function from the
selection effect. Characterized in the paper per the pre-registered
methods-contingency.

Two-step variant: classic probit on w, then the outcome net is trained on
selected units with mean f(x) + c * inverse_mills(w'gamma_hat), c a free
scalar. Population prediction is f(x) alone.

Epistemic uncertainty: ensembles of independently initialized models;
predictive variance = Var_ensemble[f(x)] + E_ensemble[sigma^2].
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .selection import probit_fit, inverse_mills

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG2PI = math.log(2.0 * math.pi)


def make_mlp(d_in: int, hidden=(64, 64), d_out: int = 1,
             dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = d_in
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last = h
    layers.append(nn.Linear(last, d_out))
    return nn.Sequential(*layers)


class DeepHeckman(nn.Module):
    """Joint-MLE Heckman with a deep outcome net and linear selection head."""

    def __init__(self, d_x: int, d_w: int, hidden=(64, 64)):
        super().__init__()
        self.f = make_mlp(d_x, hidden, 1)
        self.g = nn.Linear(d_w, 1)
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self.atanh_rho = nn.Parameter(torch.tensor(0.0))

    def nll(self, x, w, y, s, freeze_rho: bool = False):
        f = self.f(x).squeeze(-1)
        g = self.g(w).squeeze(-1)
        sigma = torch.exp(self.log_sigma)
        rho = torch.tanh(self.atanh_rho if not freeze_rho
                         else self.atanh_rho.detach() * 0.0)
        sel = s > 0.5
        ll0 = torch.special.log_ndtr(-g[~sel]).sum()
        e = (y[sel] - f[sel]) / sigma
        ll1 = (-0.5 * e**2 - self.log_sigma - 0.5 * LOG2PI).sum()
        arg = (g[sel] + rho * e) / torch.sqrt(1.0 - rho**2 + 1e-12)
        ll1 = ll1 + torch.special.log_ndtr(arg).sum()
        return -(ll0 + ll1) / len(s)

    @torch.no_grad()
    def predict_f(self, x):
        return self.f(x).squeeze(-1)

    @property
    def sigma(self) -> float:
        return float(torch.exp(self.log_sigma))

    @property
    def rho(self) -> float:
        return float(torch.tanh(self.atanh_rho))


def fit_deep_heckman(x, w, y, s, hidden=(64, 64), epochs: int = 1500,
                     warmup: int = 100, lr: float = 1e-2, seed: int = 0,
                     weight_decay: float = 1e-5) -> DeepHeckman:
    torch.manual_seed(seed)
    x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    w = torch.as_tensor(w, dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(np.nan_to_num(y), dtype=torch.float32, device=DEVICE)
    st = torch.as_tensor(s, dtype=torch.float32, device=DEVICE)
    model = DeepHeckman(x.shape[1], w.shape[1], hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        opt.zero_grad()
        loss = model.nll(x, w, yt, st, freeze_rho=(ep < warmup))
        loss.backward()
        opt.step()
        sched.step()
    return model


class DeepHeckmanTwoStep:
    """Two-step variant: classic probit, then NN outcome with an IMR feature."""

    def __init__(self, net: nn.Module, c: float, gamma: np.ndarray,
                 sigma: float):
        self.net, self.c, self.gamma, self.sigma = net, c, gamma, sigma

    @torch.no_grad()
    def predict_f(self, x):
        return self.net(x).squeeze(-1)


def fit_deep_heckman_two_step(x, w, y, s, hidden=(64, 64), epochs: int = 1500,
                              lr: float = 1e-2, seed: int = 0,
                              weight_decay: float = 1e-5):
    torch.manual_seed(seed)
    pr = probit_fit(np.asarray(w, dtype=float), np.asarray(s, dtype=float))
    sel = np.asarray(s) > 0.5
    lam = inverse_mills(np.asarray(w, dtype=float)[sel] @ pr.params)

    xs = torch.as_tensor(np.asarray(x)[sel], dtype=torch.float32,
                         device=DEVICE)
    ys = torch.as_tensor(np.asarray(y)[sel], dtype=torch.float32,
                         device=DEVICE)
    lt = torch.as_tensor(lam, dtype=torch.float32, device=DEVICE)

    net = make_mlp(xs.shape[1], hidden, 1).to(DEVICE)
    c = torch.zeros((), device=DEVICE, requires_grad=True)
    log_sigma = torch.zeros((), device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam(list(net.parameters()) + [c, log_sigma], lr=lr,
                           weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        mu = net(xs).squeeze(-1) + c * lt
        nll = (0.5 * ((ys - mu) / torch.exp(log_sigma))**2 + log_sigma
               + 0.5 * LOG2PI).mean()
        nll.backward()
        opt.step()
        sched.step()
    # residual-based sigma consistent with the classic two-step recovery
    with torch.no_grad():
        resid = (ys - net(xs).squeeze(-1) - c * lt).cpu().numpy()
    eta = np.asarray(w, dtype=float)[sel] @ pr.params
    delta = lam * (lam + eta)
    sigma2 = float(np.mean(resid**2)) + float(np.mean(delta)) * float(c)**2
    return DeepHeckmanTwoStep(net, float(c), pr.params,
                              float(np.sqrt(max(sigma2, 1e-12))))


class HeckmanEnsemble:
    """Ensemble of joint-MLE deep Heckman models -> corrected predictive
    distribution: mean_k f_k(x), Var_k[f_k(x)] + mean_k sigma_k^2."""

    def __init__(self, members: list[DeepHeckman]):
        self.members = members

    @classmethod
    def fit(cls, x, w, y, s, k: int = 5, seed: int = 0, **kw):
        return cls([fit_deep_heckman(x, w, y, s, seed=seed * 1000 + i, **kw)
                    for i in range(k)])

    @torch.no_grad()
    def predict(self, x):
        xt = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
        fs = torch.stack([m.predict_f(xt) for m in self.members])
        mu = fs.mean(0).cpu().numpy()
        epi = fs.var(0, unbiased=False).cpu().numpy()
        ale = float(np.mean([m.sigma**2 for m in self.members]))
        return mu, epi + ale

    @property
    def rho(self) -> float:
        return float(np.mean([m.rho for m in self.members]))

    @property
    def sigma(self) -> float:
        return float(np.mean([m.sigma for m in self.members]))


class HeckmanTwoStepEnsemble:
    def __init__(self, members: list[DeepHeckmanTwoStep]):
        self.members = members

    @classmethod
    def fit(cls, x, w, y, s, k: int = 5, seed: int = 0, **kw):
        return cls([fit_deep_heckman_two_step(x, w, y, s,
                                              seed=seed * 1000 + i, **kw)
                    for i in range(k)])

    @torch.no_grad()
    def predict(self, x):
        xt = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
        fs = torch.stack([m.predict_f(xt) for m in self.members])
        mu = fs.mean(0).cpu().numpy()
        epi = fs.var(0, unbiased=False).cpu().numpy()
        ale = float(np.mean([m.sigma**2 for m in self.members]))
        return mu, epi + ale
