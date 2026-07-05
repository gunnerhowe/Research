"""Uncertainty-quantification baselines for Paper A (A-E3).

All baselines are trained on the SELECTED sample only -- that is the point:
they never see the selection mechanism. The importance-weighted variant gets
ORACLE propensities P(s=1 | x, z) from the generator, i.e., exactly the
quantity covariate-shift theory asks for; under selection on unobservables
it is structurally unable to remove the conditional bias
E[y | x, s=1] - E[y | x], which is what the experiments demonstrate.

- DeepEnsembleUQ: K heteroscedastic Gaussian-NLL MLPs (Lakshminarayanan et
  al. 2017, without adversarial training), mixture predictive.
- MCDropoutUQ: heteroscedastic MLP with dropout kept on at test time
  (Gal & Ghahramani 2016), T stochastic passes.
- GPUQ: exact GP regression (RBF + white noise, tuned by marginal
  likelihood) via sklearn.
- IWDeepEnsembleUQ: deep ensemble with per-sample oracle inverse-propensity
  weights (normalized to mean 1).
- BlindTwoHeadUQ: ablation -- same two-head architecture as DeepHeckman
  (outcome net + probit selection head trained on s) but rho frozen at 0:
  the extra head without the joint error model.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .deep import DEVICE, make_mlp, fit_deep_heckman

LOG2PI = math.log(2.0 * math.pi)


class HeteroscedasticMLP(nn.Module):
    def __init__(self, d_in: int, hidden=(64, 64), dropout: float = 0.0):
        super().__init__()
        self.body = make_mlp(d_in, hidden, 2, dropout=dropout)

    def forward(self, x):
        out = self.body(x)
        mu, log_var = out[..., 0], out[..., 1]
        return mu, torch.clamp(log_var, -10.0, 6.0)


def _fit_gaussian_mlp(x, y, weights=None, hidden=(64, 64), dropout=0.0,
                      epochs=1500, lr=5e-3, seed=0, weight_decay=1e-5):
    torch.manual_seed(seed)
    xt = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
    wt = (torch.as_tensor(weights, dtype=torch.float32, device=DEVICE)
          if weights is not None else None)
    model = HeteroscedasticMLP(xt.shape[1], hidden, dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        mu, log_var = model(xt)
        nll = 0.5 * ((yt - mu)**2 / torch.exp(log_var) + log_var + LOG2PI)
        loss = (nll * wt).mean() if wt is not None else nll.mean()
        loss.backward()
        opt.step()
        sched.step()
    return model


class DeepEnsembleUQ:
    name = "deep_ensemble"

    def __init__(self, members):
        self.members = members

    @classmethod
    def fit(cls, x, y, k=5, seed=0, weights=None, **kw):
        return cls([_fit_gaussian_mlp(x, y, weights=weights,
                                      seed=seed * 1000 + i, **kw)
                    for i in range(k)])

    @torch.no_grad()
    def predict(self, x):
        xt = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
        mus, vs = [], []
        for m in self.members:
            m.eval()
            mu, log_var = m(xt)
            mus.append(mu)
            vs.append(torch.exp(log_var))
        mus, vs = torch.stack(mus), torch.stack(vs)
        mu = mus.mean(0)
        var = (vs + mus**2).mean(0) - mu**2
        return mu.cpu().numpy(), var.cpu().numpy()


class IWDeepEnsembleUQ(DeepEnsembleUQ):
    """Oracle-inverse-propensity-weighted deep ensemble."""
    name = "iw_oracle_ensemble"

    @classmethod
    def fit_iw(cls, x, y, propensity, k=5, seed=0, **kw):
        wts = 1.0 / np.clip(np.asarray(propensity, dtype=float), 1e-3, None)
        wts = wts / wts.mean()
        return cls.fit(x, y, k=k, seed=seed, weights=wts, **kw)


class MCDropoutUQ:
    name = "mc_dropout"

    def __init__(self, model, t=64):
        self.model, self.t = model, t

    @classmethod
    def fit(cls, x, y, dropout=0.1, t=64, seed=0, **kw):
        return cls(_fit_gaussian_mlp(x, y, dropout=dropout, seed=seed, **kw),
                   t=t)

    @torch.no_grad()
    def predict(self, x):
        xt = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
        self.model.train()  # dropout active
        mus, vs = [], []
        for _ in range(self.t):
            mu, log_var = self.model(xt)
            mus.append(mu)
            vs.append(torch.exp(log_var))
        mus, vs = torch.stack(mus), torch.stack(vs)
        mu = mus.mean(0)
        var = (vs + mus**2).mean(0) - mu**2
        return mu.cpu().numpy(), var.cpu().numpy()


class GPUQ:
    name = "gp"

    def __init__(self, gp, max_train=2000):
        self.gp = gp

    @classmethod
    def fit(cls, x, y, seed=0, max_train=2000, **kw):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, \
            ConstantKernel
        rng = np.random.default_rng(seed)
        x, y = np.asarray(x), np.asarray(y)
        if len(x) > max_train:
            idx = rng.choice(len(x), max_train, replace=False)
            x, y = x[idx], y[idx]
        kern = (ConstantKernel(1.0) * RBF(length_scale=np.ones(x.shape[1]))
                + WhiteKernel(noise_level=0.1))
        gp = GaussianProcessRegressor(kernel=kern, normalize_y=True,
                                      n_restarts_optimizer=2,
                                      random_state=seed)
        gp.fit(x, y)
        return cls(gp)

    def predict(self, x):
        mu, sd = self.gp.predict(np.asarray(x), return_std=True)
        return mu, sd**2


class BlindTwoHeadUQ:
    """Ablation: DeepHeckman architecture with rho permanently frozen at 0."""
    name = "blind_two_head"

    def __init__(self, members):
        self.members = members

    @classmethod
    def fit(cls, x, w, y, s, k=5, seed=0, epochs=1500, **kw):
        members = [fit_deep_heckman(x, w, y, s, seed=seed * 1000 + i,
                                    epochs=epochs, warmup=epochs, **kw)
                   for i in range(k)]  # warmup == epochs: rho never unfrozen
        return cls(members)

    @torch.no_grad()
    def predict(self, x):
        xt = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
        fs = torch.stack([m.predict_f(xt) for m in self.members])
        mu = fs.mean(0).cpu().numpy()
        epi = fs.var(0, unbiased=False).cpu().numpy()
        ale = float(np.mean([m.sigma**2 for m in self.members]))
        return mu, epi + ale
