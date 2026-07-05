"""Streaming spectral estimators for KS mode time series.

All estimators consume chunks of real-space fields (n_t, S, N) and accumulate:
  - stationary spatial power spectrum (per seed),
  - Welch temporal cross-power per retained mode (zero-padded segments, so the
    windowed-autocorrelation bias is exactly invertible via the window's linear
    autocorrelation R_w),
  - per-sector Hankel-EDMD Gram matrices at several delay strides.

Conventions (PLAN.md): modes are u_hat_m / N; one-sided power P_m = 2|u_hat_m/N|^2;
spectral density S(k) = P_m / (2 pi / L); retained sectors 0 < k <= k_max.
Per-sector dictionary: d delays of the complex mode at stride s samples; the EDMD
one-step map advances by s samples, so lambda = log(mu) / (s * dt_s).
"""
from __future__ import annotations

import numpy as np


class ModeGrid:
    def __init__(self, L, N, k_max=3.0):
        self.L, self.N, self.k_max = float(L), int(N), float(k_max)
        k_all = 2 * np.pi * np.arange(N // 2 + 1) / L
        self.m_idx = np.nonzero((k_all > 0) & (k_all <= k_max + 1e-12))[0]
        self.k = k_all[self.m_idx]
        self.k_all = k_all
        self.dk = 2 * np.pi / L


class SpectrumAccum:
    """Mean one-sided power P_m per seed over the full m range."""

    def __init__(self, n_seeds, N):
        self.p_sum = np.zeros((n_seeds, N // 2 + 1))
        self.n = 0
        self.N = N

    def add_modes(self, modes_full):
        # modes_full: (n_t, S, N//2+1) = rfft(field)/N
        self.p_sum += 2.0 * (np.abs(modes_full) ** 2).sum(axis=0)
        self.n += modes_full.shape[0]

    def power(self):
        p = self.p_sum / self.n
        p[:, 0] /= 2.0  # m=0 is not doubled (kept for completeness; excluded anyway)
        if self.N % 2 == 0:
            p[:, -1] /= 2.0  # Nyquist not doubled
        return p


class WelchAccum:
    """Welch cross-power of complex mode series, zero-padded x2, Hann window.

    Autocorrelation up to lag block//2 is recovered exactly (in expectation) via
    division by the window's linear autocorrelation.
    """

    def __init__(self, block, n_seeds, n_modes, mode_sel=None):
        self.block = int(block)
        self.hop = self.block // 2
        self.win = np.hanning(self.block)
        self.mode_sel = mode_sel  # indices into retained modes, or None = all
        nm = n_modes if mode_sel is None else len(mode_sel)
        self.p_sum = np.zeros((2 * self.block, n_seeds, nm))
        self.n_blocks = 0
        self._buf = []
        self._buf_len = 0

    def add(self, modes):  # modes: (n_t, S, M_retained) complex
        if self.mode_sel is not None:
            modes = modes[:, :, self.mode_sel]
        self._buf.append(modes)
        self._buf_len += modes.shape[0]
        while self._buf_len >= self.block:
            arr = np.concatenate(self._buf, axis=0)
            seg = arr[:self.block] * self.win[:, None, None]
            F = np.fft.fft(seg, n=2 * self.block, axis=0)
            self.p_sum += np.abs(F) ** 2
            self.n_blocks += 1
            rest = arr[self.hop:]
            self._buf = [rest]
            self._buf_len = rest.shape[0]

    def autocorr(self, max_lag=None):
        """C(tau) complex, shape (n_lags, S, nm), tau = 0..max_lag samples."""
        if self.n_blocks == 0:
            raise RuntimeError("no full Welch block accumulated")
        if max_lag is None:
            max_lag = self.block // 2
        c_raw = np.fft.ifft(self.p_sum / self.n_blocks, axis=0)[:max_lag + 1]
        rw = np.correlate(self.win, self.win, mode="full")[self.block - 1:
                                                           self.block - 1 + max_lag + 1]
        return c_raw / rw[:, None, None]


class HankelAccum:
    """Streaming per-sector Hankel-EDMD Grams at several strides.

    For stride s and d delays, accumulates Gz[a, b] = sum_t conj(Z_a) Z_b with
    Z_j(t) = x(t + s - j*s), j = 0..d. Then G = Gz[1:, 1:], A = Gz[1:, :d].
    Pairs are anchored at their future point so each is counted exactly once
    across chunk boundaries; large strides are t-subsampled by s//4 (redundant
    within an autocorrelation time; deterministic phase from the global index).
    """

    def __init__(self, strides, d, n_seeds, n_modes):
        self.strides = list(strides)
        self.d = int(d)
        self.gz = {s: np.zeros((n_seeds, n_modes, d + 1, d + 1), dtype=np.complex128)
                   for s in self.strides}
        self.npairs = {s: 0 for s in self.strides}
        self._hist = None  # (H, S, M)
        self._g0 = 0       # global index of _hist[0]
        self.hmax = (self.d + 1) * max(self.strides)

    def add(self, modes):
        arr = modes if self._hist is None else np.concatenate([self._hist, modes], axis=0)
        g0 = self._g0
        chunk_start = arr.shape[0] - modes.shape[0]
        for s in self.strides:
            step = max(1, s // 4)
            d = self.d
            # future anchors idx_y in the new-chunk region with full history
            lo = max(chunk_start, d * s)
            idx_y = np.arange(lo, arr.shape[0])
            if idx_y.size:
                idx_y = idx_y[(g0 + idx_y) % step == 0]
            if idx_y.size == 0:
                continue
            for blk in range(0, idx_y.size, 1024):  # bound temporaries
                iy = idx_y[blk:blk + 1024]
                # Z[i, :, :, j] = arr[iy[i] - j*s]
                gather = iy[:, None] - s * np.arange(d + 1)[None, :]
                Z = arr[gather]                      # (nt, d+1, S, M)
                Z = np.ascontiguousarray(np.moveaxis(Z, 1, 3))   # (nt, S, M, d+1)
                nt, S, M, D = Z.shape
                Zr = Z.reshape(nt, S * M, D).transpose(1, 2, 0)  # (S*M, D, nt)
                gz = Zr.conj() @ Zr.transpose(0, 2, 1)           # (S*M, D, D)
                self.gz[s] += gz.reshape(S, M, D, D)
                self.npairs[s] += nt
        keep = min(self.hmax, arr.shape[0])
        self._hist = arr[-keep:].copy()
        self._g0 = g0 + arr.shape[0] - keep


def edmd_eig(gz, npairs, reg=1e-10):
    """Eigen-decomposition of the per-sector EDMD operator from a (d+1,d+1) Gram.

    Returns (mu, w): eigenvalues and spectral weights of the delay-0 observable,
    so that C(n) ~ sum_j w_j mu_j^n (autocovariance of x at multiples of the
    stride). Returns NaNs if the Gram is empty/degenerate (too little data).
    """
    d = gz.shape[0] - 1
    if npairs < d + 2 or not np.isfinite(gz).all():
        return np.full(d, np.nan, complex), np.full(d, np.nan, complex)
    G = gz[1:, 1:] / npairs
    A = gz[1:, :d] / npairs
    tr = np.trace(G).real
    if not np.isfinite(tr) or tr <= 0:
        return np.full(d, np.nan, complex), np.full(d, np.nan, complex)
    G = G + reg * tr / d * np.eye(d)
    try:
        K = np.linalg.solve(G, A)
        mu, V = np.linalg.eig(K)
        Vinv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        return np.full(d, np.nan, complex), np.full(d, np.nan, complex)
    w = (G @ V)[0, :] * Vinv[:, 0]
    return mu, w


def leading_resonance(mu, w, dt_eff, mu_cap=1.005, mu_floor=1e-3, half_lag=4):
    """Continuous-time leading resonance lambda = log(mu)/dt_eff.

    Leading = largest mid-window contribution |w_j mu_j^half_lag| among stable,
    non-negligible mu (the delay-stack dictionary makes the non-leading cluster
    near-defective at mu ~ 0; raw |w_j| there is numerically pathological, but its
    contribution at moderate lag is negligible — weight by |mu|^half_lag).
    Returns (lam, weight_fraction).
    """
    contrib = np.abs(w) * np.abs(mu) ** half_lag
    ok = (np.abs(mu) <= mu_cap) & (np.abs(mu) >= mu_floor)
    if not ok.any():
        return np.nan + 0j, 0.0
    j = np.argmax(np.where(ok, contrib, -np.inf))
    lam = np.log(mu[j]) / dt_eff
    frac = contrib[j] / max(contrib[ok].sum(), 1e-300)
    return lam, float(frac)


def tau_e_from_autocorr(c, dt_s):
    """First time |rho(t)| <= 1/e; c: (n_lags, ...) complex, lag 0 first."""
    rho = np.abs(c) / np.maximum(np.abs(c[0]), 1e-300)
    below = rho <= np.exp(-1.0)
    below[0] = False
    idx = below.argmax(axis=0)          # 0 where never below
    never = ~below.any(axis=0)
    tau = idx.astype(float) * dt_s
    tau[never] = (rho.shape[0] - 1) * dt_s  # censored at max lag
    return tau, never


def omega_peak_from_welch(p_sum, block, dt_s):
    """|peak frequency| (rad/time) of the two-sided Welch PSD per (seed, mode).
    p_sum: (2*block, S, M) accumulated padded periodograms."""
    nfft = p_sum.shape[0]
    f = np.fft.fftfreq(nfft, d=dt_s) * 2 * np.pi
    idx = p_sum.argmax(axis=0)
    return np.abs(f[idx])


def pick_stride(tau_e, omega_pk, strides, d, dt_s, target_mult=3.0, alias_frac=0.8):
    """PLAN rule + anti-aliasing guard: among strides with omega_pk*s*dt_s <=
    alias_frac*pi (so log(mu) stays on the principal branch), pick the one whose
    window (d-1)*s*dt_s is closest to target_mult*tau_e; if none admissible, the
    smallest stride."""
    strides = np.asarray(strides)
    tau_e = np.atleast_1d(np.asarray(tau_e, float))
    omega_pk = np.atleast_1d(np.asarray(omega_pk, float))
    windows = (d - 1) * strides * dt_s
    dist = np.abs(windows[None, :] - target_mult * tau_e[:, None])
    admissible = omega_pk[:, None] * strides[None, :] * dt_s <= alias_frac * np.pi
    dist = np.where(admissible, dist, np.inf)
    out = strides[np.argmin(dist, axis=1)]
    out[~admissible.any(axis=1)] = strides.min()
    return out


def edmd_autocorr_r2(gz, npairs, stride, c_meas, dt_s, t_max, reg=1e-10):
    """R^2 of the EDMD-implied autocovariance C(n) = (G_hat K^n)_00 against the
    measured one, compared at lags that are multiples of the stride, up to t_max.
    Matrix powers (not eigenvectors): stable for the near-defective delay cluster."""
    d = gz.shape[0] - 1
    if npairs < d + 2 or not np.isfinite(gz).all():
        return np.nan
    G = gz[1:, 1:] / npairs
    A = gz[1:, :d] / npairs
    tr = np.trace(G).real
    if not np.isfinite(tr) or tr <= 0:
        return np.nan
    Greg = G + reg * tr / d * np.eye(d)
    try:
        K = np.linalg.solve(Greg, A)
    except np.linalg.LinAlgError:
        return np.nan
    n_max = int(t_max / (stride * dt_s))
    lag_idx = np.arange(n_max + 1) * stride
    lag_idx = lag_idx[lag_idx < c_meas.shape[0]]
    row = G[0].copy()          # row 0 of G_hat K^n, updated by right-multiplication
    pred = [row[0]]
    for _ in range(len(lag_idx) - 1):
        row = row @ K
        pred.append(row[0])
    pred = np.asarray(pred)
    meas = c_meas[lag_idx]
    if np.abs(meas[0]) < 1e-300:
        return np.nan
    resid = np.abs(pred - meas) ** 2
    var = np.abs(meas - meas.mean()) ** 2
    return float(1.0 - resid.sum() / max(var.sum(), 1e-300))
