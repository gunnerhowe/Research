"""Finite-size scaling fits of spectral quantities.

Two fitters (PLAN.md):
  1. pointwise: y(k; L) = y_inf(k) + c1(k)/L (or + c2(k)/L^2) at fixed k on the
     common grid (k a multiple of 2 pi / 22, present at every training size),
     weighted by seed SE; model form chosen by AICc across all k jointly.
  2. smooth ("fitted FSS flow"): y(k, ell) = f0(k) + ell f1(k), ell = L_base/L,
     with f0, f1 cubic B-splines in k on a fixed knot grid, linear least squares
     over ALL modes of ALL training sizes. Evaluating at ell_target on the target
     k-grid is the domain-extension prediction.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline, PchipInterpolator

L_BASE = 22.0


def pointwise_fit(L, y, se=None, power=1):
    """Weighted LS fit y = y_inf + c/L^power. Returns dict with params, R2, AICc."""
    L = np.asarray(L, float)
    y = np.asarray(y, float)
    w = np.ones_like(y) if se is None else 1.0 / np.maximum(np.asarray(se, float), 1e-12)
    X = np.stack([np.ones_like(L), (L_BASE / L) ** power], axis=1)
    A = X * w[:, None]
    b = y * w
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = y - X @ coef
    ss_res = float(np.sum((w * resid) ** 2))
    ss_tot = float(np.sum((w * (y - np.average(y, weights=w**2))) ** 2))
    n, p = len(y), 2
    r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
    # Gaussian AICc from weighted residuals
    aic = n * np.log(max(ss_res / n, 1e-300)) + 2 * p
    aicc = aic + (2 * p * (p + 1)) / max(n - p - 1, 1e-12)
    return {"y_inf": float(coef[0]), "c": float(coef[1]), "r2": float(r2),
            "aicc": float(aicc), "power": power}


def pointwise_predict(fit, L):
    return fit["y_inf"] + fit["c"] * (L_BASE / np.asarray(L, float)) ** fit["power"]


def _bspline_design(k, knots_interior, k_lo, k_hi, degree=3):
    t = np.concatenate([[k_lo] * (degree + 1), knots_interior, [k_hi] * (degree + 1)])
    n_basis = len(t) - degree - 1
    X = np.zeros((len(k), n_basis))
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        X[:, j] = BSpline(t, c, degree, extrapolate=True)(k)
    return X, t


class SmoothFlow:
    """y(k, ell) = f0(k) + ell f1(k) with B-spline f0, f1; ell = L_BASE / L."""

    def __init__(self, k_lo=0.05, k_hi=3.05, knot_spacing=0.2, degree=3,
                 log_y=False):
        self.k_lo, self.k_hi, self.degree = k_lo, k_hi, degree
        self.knots = np.arange(k_lo + knot_spacing, k_hi - knot_spacing / 2,
                               knot_spacing)
        self.log_y = log_y
        self.coef = None

    def fit(self, k, L, y, se=None):
        k = np.asarray(k, float)
        L = np.asarray(L, float)
        y = np.asarray(y, float)
        if self.log_y:
            ok = y > 0
            k, L, y = k[ok], L[ok], y[ok]
            se = None if se is None else np.asarray(se, float)[ok] / y  # dlog ~ rel
            y = np.log(y)
        ell = L_BASE / L
        X0, self.t = _bspline_design(k, self.knots, self.k_lo, self.k_hi, self.degree)
        X = np.concatenate([X0, ell[:, None] * X0], axis=1)
        w = np.ones_like(y) if se is None else 1.0 / np.maximum(se, 1e-12)
        # mild ridge for stability at sparsely-populated low-k knots
        lam = 1e-6 * np.trace((X * w[:, None]).T @ (X * w[:, None])) / X.shape[1]
        XtX = (X * w[:, None] ** 2).T @ X + lam * np.eye(X.shape[1])
        Xty = (X * w[:, None] ** 2).T @ y
        self.coef = np.linalg.solve(XtX, Xty)
        self.n_basis = X0.shape[1]
        return self

    def predict(self, k, L):
        k = np.asarray(k, float)
        ell = L_BASE / float(L)
        X0, _ = _bspline_design(k, self.knots, self.k_lo, self.k_hi, self.degree)
        y = X0 @ self.coef[:self.n_basis] + ell * (X0 @ self.coef[self.n_basis:])
        return np.exp(y) if self.log_y else y


class PointwiseFlow:
    """Nearest-neighbour-in-k pointwise 1/L flow (the GATE-S recipe generalized to
    arbitrary target k). For each target wavenumber, gather the value at the
    nearest mode of each training size, fit y = y_inf + c*(L_BASE/L)^p with p
    chosen by AICc (p in {1,2}), and predict at the target L. No smoothness is
    imposed across k, so the real per-mode structure is preserved; the only
    requirement is that every training size has a mode near the target k (true by
    construction for the full-support band). Optionally fits in log y (for
    positive quantities: gamma, S)."""

    def __init__(self, log_y=False, min_sizes=2):
        self.log_y = log_y
        self.min_sizes = min_sizes

    def fit(self, size_curves):
        # size_curves: {L: (k_array, y_array[, se_array])} seed-reduced
        self.sizes = sorted(size_curves)
        self.ks = {L: np.asarray(size_curves[L][0], float) for L in self.sizes}
        self.ys = {L: np.asarray(size_curves[L][1], float) for L in self.sizes}
        self.se = {L: (np.asarray(size_curves[L][2], float)
                       if len(size_curves[L]) > 2 else None) for L in self.sizes}
        return self

    def predict(self, k_target, L_target):
        k_target = np.atleast_1d(np.asarray(k_target, float))
        out = np.full(len(k_target), np.nan)
        for i, kt in enumerate(k_target):
            Ls, ys, kdev, ses = [], [], [], []
            for L in self.sizes:
                j = int(np.argmin(np.abs(self.ks[L] - kt)))
                yv = self.ys[L][j]
                if not np.isfinite(yv):
                    continue
                if self.log_y and yv <= 0:
                    continue
                Ls.append(L)
                ys.append(np.log(yv) if self.log_y else yv)
                kdev.append(abs(self.ks[L][j] - kt))
                if self.se[L] is not None:
                    s = self.se[L][j]
                    ses.append(s / max(abs(yv), 1e-12) if self.log_y else s)
                else:
                    ses.append(0.0)
            if len(Ls) < self.min_sizes:
                continue
            Ls = np.asarray(Ls, float)
            ys = np.asarray(ys, float)
            # weight: k-proximity x inverse-SE (downweight noisy per-mode estimates)
            w = np.exp(-(np.asarray(kdev) / (2 * np.pi / L_BASE)) ** 2)
            ses = np.asarray(ses)
            if np.any(ses > 0):
                floor = np.median(ses[ses > 0])
                w = w / np.maximum(ses, 0.25 * floor)
            best = None
            for p in (1, 2):
                X = np.stack([np.ones_like(Ls), (L_BASE / Ls) ** p], axis=1)
                A = X * w[:, None]
                coef, *_ = np.linalg.lstsq(A, ys * w, rcond=None)
                resid = ys - X @ coef
                ss = float(np.sum((w * resid) ** 2))
                n, kp = len(ys), 2
                aicc = n * np.log(max(ss / n, 1e-300)) + 2 * kp + \
                    (2 * kp * (kp + 1)) / max(n - kp - 1, 1e-9)
                pred = coef[0] + coef[1] * (L_BASE / L_target) ** p
                if best is None or aicc < best[0]:
                    best = (aicc, pred)
            out[i] = np.exp(best[1]) if self.log_y else best[1]
        return out


def interp_null(k_src, y_src, k_target, log_y=False):
    """No-flow null: PCHIP interpolation of a single-size measured curve onto the
    target grid; linear extrapolation from the two lowest points below k_src.min()."""
    k_src = np.asarray(k_src, float)
    y = np.asarray(y_src, float)
    if log_y:
        ok = np.isfinite(y) & (y > 0)
    else:
        ok = np.isfinite(y)
    k_src, y = k_src[ok], y[ok]
    yy = np.log(y) if log_y else y
    f = PchipInterpolator(k_src, yy, extrapolate=False)
    out = f(np.asarray(k_target, float))
    lo = np.asarray(k_target) < k_src[0]
    if lo.any():
        slope = (yy[1] - yy[0]) / (k_src[1] - k_src[0])
        out[lo] = yy[0] + slope * (np.asarray(k_target)[lo] - k_src[0])
    hi = np.asarray(k_target) > k_src[-1]
    if hi.any():
        slope = (yy[-1] - yy[-2]) / (k_src[-1] - k_src[-2])
        out[hi] = yy[-1] + slope * (np.asarray(k_target)[hi] - k_src[-1])
    return np.exp(out) if log_y else out
