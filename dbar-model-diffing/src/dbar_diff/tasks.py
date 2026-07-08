"""Stationary symbolic task generators: edge-emitting HMMs (epsilon-machine style).

An HMM here is T[a, s, s'] = P(emit a, go to s' | in s). The three named tasks are
fixed by PLAN.md:

- GM     golden-mean process (m=2, h = 2/3 bits, P(1) = 1/3, forbids "11")
- FEVEN  symbol-flipped even process (m=2, h = 2/3 bits, P(1) = 1/3, even 0-runs) —
         the different-task control with IDENTICAL entropy rate and unigram marginal
         to GM, so distances to it cannot be explained by entropy or marginals.
- MESS3  Marzen–Crutchfield mess3 (m=3), the Shai-et-al. belief-geometry generator.
"""
from __future__ import annotations

import numpy as np


class HMM:
    def __init__(self, T, name=""):
        T = np.asarray(T, dtype=np.float64)
        assert T.ndim == 3
        self.T = T
        self.m, self.S = T.shape[0], T.shape[1]
        self.name = name
        row = T.sum(axis=0)          # (S, S') combined transition matrix
        assert np.allclose(row.sum(axis=1), 1.0), "rows must sum to 1"
        # stationary state distribution
        w, v = np.linalg.eig(row.T)
        i = np.argmin(np.abs(w - 1.0))
        pi = np.real(v[:, i])
        self.pi = pi / pi.sum()
        # flat (a, s') distribution per state, for vectorized sampling
        self._flat = T.transpose(1, 0, 2).reshape(self.S, self.m * self.S)
        self._cum = np.cumsum(self._flat, axis=1)

    def is_unifilar(self):
        """Unifilar: given (s, a), at most one successor state has positive prob."""
        return bool(np.all((self.T > 1e-12).sum(axis=2) <= 1))

    def entropy_rate(self):
        """Exact for unifilar HMMs: h = Σ_s π_s H(emission dist at s) (bits)."""
        if not self.is_unifilar():
            raise ValueError("closed-form entropy rate requires unifilarity")
        em = self.T.sum(axis=2).T          # (S, m) emission probs per state
        h_s = np.array([-np.sum(p[p > 0] * np.log2(p[p > 0])) for p in em])
        return float(self.pi @ h_s)

    def marginal(self):
        """Stationary symbol marginal P(a)."""
        return self.T.sum(axis=2) @ self.pi

    def sample(self, n_chains, length, seed=0, burn=64):
        """Sample (n_chains, length) symbols, stationary start + burn-in."""
        rng = np.random.default_rng(seed)
        s = rng.choice(self.S, size=n_chains, p=self.pi)
        out = np.empty((n_chains, length), dtype=np.int8)
        for t in range(burn + length):
            u = rng.random(n_chains)
            idx = (self._cum[s] < u[:, None]).sum(axis=1)
            idx = np.minimum(idx, self.m * self.S - 1)
            a, s = idx // self.S, idx % self.S
            if t >= burn:
                out[:, t - burn] = a
        return out


def golden_mean():
    T = np.zeros((2, 2, 2))
    T[0, 0, 0] = 0.5   # A --0--> A
    T[1, 0, 1] = 0.5   # A --1--> B
    T[0, 1, 0] = 1.0   # B --0--> A
    return HMM(T, "GM")


def flipped_even():
    T = np.zeros((2, 2, 2))
    T[1, 0, 0] = 0.5   # A --1--> A
    T[0, 0, 1] = 0.5   # A --0--> B
    T[0, 1, 0] = 1.0   # B --0--> A  (0-runs between 1s have even length)
    return HMM(T, "FEVEN")


def mess3(x=0.15, a=0.6):
    b = (1 - a) / 2
    y = 1 - 2 * x
    ay, ax, by, bx = a * y, a * x, b * y, b * x
    T = np.zeros((3, 3, 3))
    T[0] = [[ay, bx, bx], [ax, by, bx], [ax, bx, by]]
    T[1] = [[by, ax, bx], [bx, ay, bx], [bx, ax, by]]
    T[2] = [[by, bx, ax], [bx, by, ax], [bx, bx, ay]]
    return HMM(T, "MESS3")


TASKS = {"gm": golden_mean, "feven": flipped_even, "mess3": mess3}
