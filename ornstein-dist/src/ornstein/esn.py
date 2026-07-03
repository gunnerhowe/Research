"""Echo State Network surrogate of a multivariate time series (standard reservoir
computing setup: tanh reservoir, [r, r^2] readout features, ridge regression).

Used as the "real learned surrogate" for the metric study: sweeping the spectral
radius produces a family of surrogates ranging from dynamically faithful to subtly
(or grossly) wrong, on which the metrics can disagree.
"""
from __future__ import annotations

import numpy as np


class ESN:
    def __init__(self, dim_in, n_res=400, density=0.02, rho_spec=0.9,
                 sigma_in=0.5, ridge=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.n_res = n_res
        W = rng.normal(0, 1, (n_res, n_res))
        W[rng.random((n_res, n_res)) > density] = 0.0
        eig = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (rho_spec / eig)
        self.W_in = rng.uniform(-sigma_in, sigma_in, (n_res, dim_in))
        self.b = rng.uniform(-0.2, 0.2, n_res)
        self.ridge = ridge
        self.rho_spec = rho_spec

    def _features(self, r):
        return np.concatenate([r, r * r])

    def fit(self, U, washout=1000):
        """Teacher-forced training to predict U[t+1] from U[t]. U: (T, d) normalized."""
        T, d = U.shape
        r = np.zeros(self.n_res)
        feats = np.empty((T - 1 - washout, 2 * self.n_res))
        targets = U[washout + 1:]
        for t in range(T - 1):
            r = np.tanh(self.W @ r + self.W_in @ U[t] + self.b)
            if t >= washout:
                feats[t - washout] = self._features(r)
        A = feats.T @ feats + self.ridge * np.eye(2 * self.n_res)
        self.W_out = np.linalg.solve(A, feats.T @ targets)
        self.r_end = r
        self.u_end = U[-1]
        pred = feats @ self.W_out
        self.train_nrmse = float(np.sqrt(np.mean((pred - targets) ** 2)))
        return self

    def rollout(self, n_steps, blowup=1e3):
        """Autonomous prediction from the post-training state. Returns (traj, status)."""
        r = self.r_end.copy()
        u = self.u_end.copy()
        out = np.empty((n_steps, len(u)))
        for t in range(n_steps):
            r = np.tanh(self.W @ r + self.W_in @ u + self.b)
            u = self._features(r) @ self.W_out
            out[t] = u
            if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > blowup:
                return out[:t], "diverged"
        if np.std(out[-n_steps // 4:], axis=0).max() < 1e-3:
            return out, "collapsed"
        return out, "ok"
