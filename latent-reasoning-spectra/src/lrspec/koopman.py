"""E2: routed-vs-local dynamics and orbit-level (Koopman/EDMD) operators.

Routedness index (PLAN.md E2):
  R_{t,k} = ||G_{t->t+k} - J_{t+k-1} ... J_t||_F / ||G_{t->t+k}||_F
where G is the full influence Jacobian (all paths, incl. KV written at intermediate
latent slots) and the product chains the local step Jacobians.

Pooled EDMD: least-squares Koopman estimate over all (c_t -> c_{t+1}) transitions in a
problem set, in a PCA-reduced coordinate system (r dims), optionally with delay
embedding 2.  Per-step orbit-level quantities:
  - koopman_residual(t)  = ||z_{t+1} - A z_t|| / ||z_{t+1}||   (deviation from the
    pooled linear evolution at step t; a routedness/nonlinearity signature)
  - unit_participation(t) = ||P_unit z_t|| / ||z_t||  with P_unit the spectral projector
    onto Koopman modes with |lambda| in [0.9, 1.1]  (orbit-level anchor predictor)
"""

from __future__ import annotations

import numpy as np


def routedness(G: np.ndarray, J_chain: list[np.ndarray]) -> dict:
    P = J_chain[0]
    for J in J_chain[1:]:
        P = J @ P
    num = float(np.linalg.norm(G - P))
    den = float(np.linalg.norm(G))
    return {
        "R": num / den if den > 0 else np.nan,
        "norm_G": den,
        "norm_chain": float(np.linalg.norm(P)),
    }


class PooledEDMD:
    def __init__(self, r: int = 128, ridge: float = 1e-6, delay: int = 1):
        self.r = r
        self.ridge = ridge
        self.delay = delay  # 1 = plain linear dictionary; 2 = delay embedding of 2

    def fit(self, trajectories: list[np.ndarray]) -> "PooledEDMD":
        """trajectories: list of (T, 768) arrays (c_1..c_6 per problem)."""
        X_all = np.concatenate(trajectories, axis=0)
        self.mean_ = X_all.mean(axis=0)
        Xc = X_all - self.mean_
        # PCA basis
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.r = min(self.r, Vt.shape[0])
        self.basis_ = Vt[: self.r].T  # (768, r)
        self.explained_ = float((S[: self.r] ** 2).sum() / (S ** 2).sum())

        pairs_x, pairs_y = [], []
        for tr in trajectories:
            z = (tr - self.mean_) @ self.basis_  # (T, r)
            if self.delay == 1:
                pairs_x.append(z[:-1])
                pairs_y.append(z[1:])
            else:
                zz = np.concatenate([z[1:], z[:-1]], axis=1)  # (T-1, 2r)
                pairs_x.append(zz[:-1])
                pairs_y.append(zz[1:])
        X = np.concatenate(pairs_x, axis=0)
        Y = np.concatenate(pairs_y, axis=0)
        d = X.shape[1]
        A = np.linalg.solve(X.T @ X + self.ridge * np.eye(d), X.T @ Y).T  # (d, d)
        self.A_ = A
        self.eigvals_, self.eigvecs_ = np.linalg.eig(A)
        self.resid_ = float(np.linalg.norm(Y - X @ A.T) / np.linalg.norm(Y))
        return self

    def spectrum(self) -> np.ndarray:
        return self.eigvals_

    def _proj(self, tr: np.ndarray) -> np.ndarray:
        return (tr - self.mean_) @ self.basis_

    def unit_projector(self, band=(0.9, 1.1)) -> np.ndarray:
        """Oblique spectral projector onto modes with |lambda| in band."""
        sel = (np.abs(self.eigvals_) >= band[0]) & (np.abs(self.eigvals_) <= band[1])
        V = self.eigvecs_
        try:
            W = np.linalg.inv(V)  # rows: left eigenvectors
        except np.linalg.LinAlgError:
            W = np.linalg.pinv(V)
        P = (V[:, sel] @ W[sel, :]).real
        return P

    def per_step(self, tr: np.ndarray, band=(0.9, 1.1)) -> dict:
        """Per-step orbit-level quantities for one trajectory (T, 768), delay=1 only."""
        assert self.delay == 1
        z = self._proj(tr)
        P = self.unit_projector(band)
        resid, part = [], []
        for t in range(z.shape[0] - 1):
            pred = self.A_ @ z[t]
            resid.append(float(np.linalg.norm(z[t + 1] - pred) /
                               max(np.linalg.norm(z[t + 1]), 1e-9)))
            part.append(float(np.linalg.norm(P @ z[t]) /
                              max(np.linalg.norm(z[t]), 1e-9)))
        return {"koopman_residual": resid, "unit_participation": part}
