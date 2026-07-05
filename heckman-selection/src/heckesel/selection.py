"""Two-equation Heckman selection models (classic linear versions).

Model (Heckman 1979):
    selection:  s_i = 1[ w_i' gamma + u_i > 0 ],  u_i ~ N(0, 1)
    outcome:    y_i = x_i' beta + eps_i,          eps_i ~ N(0, sigma^2)
                y_i observed iff s_i = 1;  Corr(u_i, eps_i) = rho.

Estimators:
- probit_fit:        Newton-Raphson probit MLE (selection equation alone).
- heckman_two_step:  Heckman (1979) two-step: probit, then OLS of y on
                     [x, inverse Mills ratio]; sigma and rho recovered from
                     the residuals and the Mills coefficient.
- heckman_mle:       joint MLE of the bivariate-normal likelihood. Exact
                     gradients via torch (float64); scipy BFGS; started at
                     the two-step estimate (falls back to zeros).

Faithfulness is gated by tests/ (A-E0): statsmodels Probit agreement, a
statsmodels-composed two-step on Mroz87, published sampleSelection (R)
reference values, and large-n synthetic parameter recovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import optimize
from scipy.special import log_ndtr, ndtr


# --------------------------------------------------------------------- utils


def inverse_mills(z: np.ndarray) -> np.ndarray:
    """phi(z) / Phi(z), computed in log space for numerical stability."""
    z = np.asarray(z, dtype=float)
    log_pdf = -0.5 * z**2 - 0.5 * np.log(2.0 * np.pi)
    return np.exp(log_pdf - log_ndtr(z))


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


# --------------------------------------------------------------------- probit


@dataclass
class ProbitResult:
    params: np.ndarray
    loglik: float
    cov: np.ndarray
    converged: bool
    n_iter: int


def probit_fit(W: np.ndarray, s: np.ndarray, max_iter: int = 100,
               tol: float = 1e-10) -> ProbitResult:
    """Probit MLE by Newton-Raphson with analytic Hessian.

    W must already contain a constant column if one is wanted.
    """
    W = _as_2d(W)
    s = np.asarray(s, dtype=float)
    n, k = W.shape
    gamma = np.zeros(k)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        eta = W @ gamma
        # score: sum_i q_i * lambda(q_i eta_i) * w_i with q_i = 2 s_i - 1
        q = 2.0 * s - 1.0
        lam = inverse_mills(q * eta) * q          # generalized residual
        score = W.T @ lam
        # Hessian: -sum_i lam_i (lam_i + eta_i) w_i w_i'
        d = lam * (lam + eta)
        H = -(W * d[:, None]).T @ W
        step = np.linalg.solve(H, score)
        gamma_new = gamma - step
        if np.max(np.abs(gamma_new - gamma)) < tol:
            gamma = gamma_new
            converged = True
            break
        gamma = gamma_new
    eta = W @ gamma
    ll = float(np.sum(s * log_ndtr(eta) + (1 - s) * log_ndtr(-eta)))
    lam = inverse_mills((2 * s - 1) * eta) * (2 * s - 1)
    d = lam * (lam + eta)
    H = -(W * d[:, None]).T @ W
    cov = np.linalg.inv(-H)
    return ProbitResult(params=gamma, loglik=ll, cov=cov, converged=converged,
                        n_iter=it)


# ------------------------------------------------------------------ two-step


@dataclass
class HeckmanResult:
    beta: np.ndarray            # outcome coefficients
    gamma: np.ndarray           # selection coefficients
    sigma: float                # outcome error sd
    rho: float                  # Corr(u, eps)
    beta_lambda: float          # coefficient on the inverse Mills ratio
                                # (= rho * sigma in the model)
    loglik: float | None        # joint log-likelihood (MLE only)
    method: str
    converged: bool
    extra: dict = field(default_factory=dict)

    def predict_outcome(self, X: np.ndarray) -> np.ndarray:
        """Population (selection-corrected) conditional mean E[y | x]."""
        return _as_2d(X) @ self.beta

    def predict_observed(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Selected-sample conditional mean E[y | x, s=1]."""
        return (_as_2d(X) @ self.beta
                + self.rho * self.sigma * inverse_mills(_as_2d(W) @ self.gamma))


def heckman_two_step(y: np.ndarray, X: np.ndarray, s: np.ndarray,
                     W: np.ndarray) -> HeckmanResult:
    """Heckman (1979) two-step estimator.

    Parameters
    ----------
    y : outcomes, length n; entries where s == 0 are ignored (may be NaN).
    X : outcome design (n, k_o), constant included by caller.
    s : selection indicator, length n.
    W : selection design (n, k_s), constant included by caller.
    """
    X, W = _as_2d(X), _as_2d(W)
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    sel = s > 0.5

    pr = probit_fit(W, s)
    eta1 = W[sel] @ pr.params
    lam = inverse_mills(eta1)
    delta = lam * (lam + eta1)

    Z = np.column_stack([X[sel], lam])
    coef, *_ = np.linalg.lstsq(Z, y[sel], rcond=None)
    beta, beta_lam = coef[:-1], float(coef[-1])
    resid = y[sel] - Z @ coef
    n1 = int(sel.sum())
    sigma2 = float(resid @ resid) / n1 + float(np.mean(delta)) * beta_lam**2
    sigma = float(np.sqrt(max(sigma2, 1e-12)))
    rho = float(np.clip(beta_lam / sigma, -0.9999, 0.9999))
    return HeckmanResult(beta=beta, gamma=pr.params, sigma=sigma, rho=rho,
                         beta_lambda=beta_lam, loglik=None, method="two-step",
                         converged=pr.converged,
                         extra={"probit_loglik": pr.loglik,
                                "resid_var": float(resid @ resid) / n1,
                                "mean_delta": float(np.mean(delta))})


# ----------------------------------------------------------------- joint MLE


def _heckman_nll_torch(theta: torch.Tensor, y1: torch.Tensor, X1: torch.Tensor,
                       X0w: torch.Tensor, W1: torch.Tensor,
                       ko: int, ks: int) -> torch.Tensor:
    """Negative joint log-likelihood (Heckman 1979, eq. for bivariate normal).

    theta = [beta (ko), gamma (ks), log_sigma, atanh_rho].
    y1/X1/W1: selected observations; X0w: selection design of non-selected.
    """
    beta = theta[:ko]
    gamma = theta[ko:ko + ks]
    sigma = torch.exp(theta[ko + ks])
    rho = torch.tanh(theta[ko + ks + 1])

    # non-selected: log Phi(-w'gamma)
    ll0 = torch.special.log_ndtr(-(X0w @ gamma)).sum()

    # selected: log phi_sigma(y - x'beta) + log Phi( (w'gamma + rho e / sigma)
    #           / sqrt(1 - rho^2) )
    e = (y1 - X1 @ beta) / sigma
    ll1 = (-0.5 * e**2 - torch.log(sigma)
           - 0.5 * float(np.log(2.0 * np.pi))).sum()
    arg = (W1 @ gamma + rho * e) / torch.sqrt(1.0 - rho**2 + 1e-14)
    ll1 = ll1 + torch.special.log_ndtr(arg).sum()
    return -(ll0 + ll1)


def heckman_mle(y: np.ndarray, X: np.ndarray, s: np.ndarray, W: np.ndarray,
                start: HeckmanResult | None = None,
                gtol: float = 1e-9) -> HeckmanResult:
    """Joint MLE of the Heckman model (exact torch gradients + scipy BFGS)."""
    X, W = _as_2d(X), _as_2d(W)
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    sel = s > 0.5
    ko, ks = X.shape[1], W.shape[1]

    if start is None:
        start = heckman_two_step(y, X, s, W)
    theta0 = np.concatenate([
        start.beta, start.gamma,
        [np.log(max(start.sigma, 1e-6))],
        [np.arctanh(np.clip(start.rho, -0.99, 0.99))],
    ])

    y1 = torch.tensor(y[sel], dtype=torch.float64)
    X1 = torch.tensor(X[sel], dtype=torch.float64)
    W1 = torch.tensor(W[sel], dtype=torch.float64)
    X0w = torch.tensor(W[~sel], dtype=torch.float64)

    def fun_and_grad(th_np: np.ndarray):
        th = torch.tensor(th_np, dtype=torch.float64, requires_grad=True)
        nll = _heckman_nll_torch(th, y1, X1, X0w, W1, ko, ks)
        (g,) = torch.autograd.grad(nll, th)
        return float(nll.item()), g.numpy()

    res = optimize.minimize(fun_and_grad, theta0, jac=True, method="BFGS",
                            options={"gtol": gtol, "maxiter": 2000})

    # Newton polish with the exact autograd Hessian: BFGS routinely stops on
    # "precision loss" a hair from the optimum; a few damped Newton steps land
    # it, and the observed information matrix is the covariance by-product.
    th = torch.tensor(res.x, dtype=torch.float64)

    def nll_fn(t):
        return _heckman_nll_torch(t, y1, X1, X0w, W1, ko, ks)

    grad_norm = np.inf
    H = None
    for _ in range(50):
        t_req = th.clone().requires_grad_(True)
        nll = nll_fn(t_req)
        (g,) = torch.autograd.grad(nll, t_req)
        grad_norm = float(g.abs().max())
        if grad_norm < 1e-6 * (1.0 + float(nll.abs())):
            break
        H = torch.autograd.functional.hessian(nll_fn, th)
        Hn = H.numpy()
        gn = g.numpy()
        lam_damp = 0.0
        for _try in range(12):
            try:
                step = np.linalg.solve(
                    Hn + lam_damp * np.eye(len(gn)), gn)
                cand = th - torch.tensor(step)
                if nll_fn(cand).item() <= nll.item() + 1e-12:
                    th = cand
                    break
            except np.linalg.LinAlgError:
                pass
            lam_damp = max(2.0 * lam_damp, 1e-6)
        else:
            break

    th_np = th.numpy()
    final_nll = float(nll_fn(th).item())
    if H is None:
        H = torch.autograd.functional.hessian(nll_fn, th)
    try:
        cov_theta = np.linalg.inv(H.numpy())
    except np.linalg.LinAlgError:
        cov_theta = np.full((len(th_np), len(th_np)), np.nan)

    beta, gamma = th_np[:ko], th_np[ko:ko + ks]
    sigma = float(np.exp(th_np[ko + ks]))
    rho = float(np.tanh(th_np[ko + ks + 1]))
    converged = grad_norm < 1e-4 * (1.0 + abs(final_nll))
    return HeckmanResult(beta=beta, gamma=gamma, sigma=sigma, rho=rho,
                         beta_lambda=rho * sigma, loglik=-final_nll,
                         method="mle", converged=converged,
                         extra={"n_iter": int(res.nit),
                                "grad_norm": grad_norm,
                                "cov_theta": cov_theta})


def heckman_loglik(y, X, s, W, beta, gamma, sigma, rho) -> float:
    """Joint log-likelihood at given parameters (for tests/model comparison)."""
    X, W = _as_2d(X), _as_2d(W)
    s = np.asarray(s, dtype=float)
    sel = s > 0.5
    ll0 = log_ndtr(-(W[~sel] @ gamma)).sum()
    e = (np.asarray(y, dtype=float)[sel] - X[sel] @ beta) / sigma
    ll1 = np.sum(-0.5 * e**2 - np.log(sigma) - 0.5 * np.log(2 * np.pi))
    arg = (W[sel] @ gamma + rho * e) / np.sqrt(1 - rho**2)
    ll1 += log_ndtr(arg).sum()
    return float(ll0 + ll1)


def selection_probability(W: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """P(s=1 | w) = Phi(w'gamma)."""
    return ndtr(_as_2d(W) @ np.asarray(gamma, dtype=float))
