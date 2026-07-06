"""Off-manifold guards and independent validation.

Two roles, deliberately kept independent so the validator is not the gate:

1. GateKDE -- a differentiable torch Gaussian KDE on the reference pool, used
   INSIDE the reward's density gate and hard veto during sampling.

2. IndependentValidator -- a SEPARATE density estimator (sklearn KDE, different
   bandwidth) fit on D_full plus a nearest-neighbour-in-D_full distance. Every
   synthesized batch is scored by it; rejections (below a D_full-self
   percentile of log-density, or beyond a D_full-self percentile of NN
   distance) are counted and reported as the K3 diagnostic. Because it is fit
   on different data with a different estimator than the gate, passing it is
   evidence the samples are on-manifold, not a tautology.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import KernelDensity, NearestNeighbors


class GateKDE:
    """Differentiable Gaussian KDE (log-density) on a reference pool."""

    def __init__(self, X_ref, bandwidth=0.3, max_points=1500, device=None,
                 seed=0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        rng = np.random.default_rng(seed)
        if len(X_ref) > max_points:
            X_ref = X_ref[rng.choice(len(X_ref), max_points, replace=False)]
        self.pts = torch.tensor(X_ref, dtype=torch.float32, device=self.device)
        self.h = float(bandwidth)
        self.d = X_ref.shape[1]
        self._norm = -0.5 * self.d * np.log(2 * np.pi * self.h ** 2)

    def log_p(self, X):
        # X: (n,d) torch; returns (n,) log-density, differentiable in X
        d2 = ((X[:, None, :] - self.pts[None, :, :]) ** 2).sum(-1)
        logk = self._norm - 0.5 * d2 / self.h ** 2
        return torch.logsumexp(logk, dim=1) - np.log(len(self.pts))

    @torch.no_grad()
    def log_p_np(self, X):
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        return self.log_p(Xt).cpu().numpy()

    def quantile_threshold(self, X_eval, q):
        """log-density level at the q-quantile of X_eval's own densities;
        used to pre-register tau as a population-coverage quantile."""
        lp = self.log_p_np(X_eval)
        return float(np.quantile(lp, q))


class IndependentValidator:
    """Independent density + NN-distance validator fit on D_full."""

    def __init__(self, X_full, bandwidth=0.4, logp_q=0.01, nn_q=0.99):
        self.kde = KernelDensity(bandwidth=bandwidth).fit(X_full)
        self.nn = NearestNeighbors(n_neighbors=2).fit(X_full)
        self_lp = self.kde.score_samples(X_full)
        # exclude self (column 0 is the point itself at distance 0)
        self_nn = self.nn.kneighbors(X_full)[0][:, 1]
        # thresholds are D_full's OWN percentiles (pre-registered as quantiles)
        self.logp_thresh = float(np.quantile(self_lp, logp_q))
        self.nn_thresh = float(np.quantile(self_nn, nn_q))

    def score(self, X):
        lp = self.kde.score_samples(X)
        # X are external points: column 0 is the genuine nearest D_full neighbour
        nnd = self.nn.kneighbors(X)[0][:, 0]
        reject = (lp < self.logp_thresh) | (nnd > self.nn_thresh)
        return {
            "reject_rate": float(reject.mean()),
            "logp_reject_rate": float((lp < self.logp_thresh).mean()),
            "nn_reject_rate": float((nnd > self.nn_thresh).mean()),
            "mean_logp": float(lp.mean()),
            "mean_nn": float(nnd.mean()),
            "reject_mask": reject,
        }
