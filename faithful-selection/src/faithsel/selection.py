"""Two-equation Heckman selection models for the verbalization confound.

Model (Heckman 1979), instantiated for CoT faithfulness:
    selection:  V_i = 1[ w_i' gamma + u_i > 0 ],   u_i ~ N(0, 1)
        V_i = "the CoT verbalizes the hint" (the only channel through which
        reliance can be read off observation-only deployments);
    outcome:    R_i = x_i' beta + eps_i,           eps_i ~ N(0, sigma^2)
        R_i = latent reliance on the hint, observed iff V_i = 1;
    Corr(u_i, eps_i) = rho.  rho != 0 <=> verbalization is endogenously
    selected on the latent computation => naive faithfulness metrics that
    read reliance off the verbalized channel are confounded.

Estimators:
- probit_fit:            Newton-Raphson probit MLE (selection equation alone).
- heckman_two_step:      Heckman (1979) two-step (probit -> OLS with IMR).
- heckman_mle:           joint MLE, exact torch gradients, Newton polish.
- heckman_mle_fixed_rho: restricted MLE with rho frozen (rho-sensitivity +
                         the LR test's null model at rho = 0).
- heckprob_mle:          binary-outcome probit with selection (van de Ven &
                         van Praag 1981), for observation-only deployments
                         where the reliance proxy is an indicator.

Tests / corrections:
- rho_lr_test, rho_wald_test:  H0: rho = 0.
- estimands / heckprob_estimands: population mean reliance, hidden reliance
  E[R | V=0] (what the naive metric silently sets to "no reliance"), and the
  naive selected-sample estimators they correct.
- rho_sensitivity: corrected estimands across a fixed-rho grid (identification
  fallback when the exclusion restriction is contested).
- bootstrap_fit: nonparametric bootstrap CIs over instances.

The linear core (probit/two-step/MLE) is vendored from the author's
heckman-selection project (src/heckesel/selection.py), where it is gated by
tests against statsmodels Probit, a statsmodels-composed two-step on Mroz87,
published R sampleSelection reference values, and large-n synthetic recovery.
tests/test_selection.py re-gates the vendored copy plus everything added here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import optimize, stats
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

    def loglik_at(g):
        eta = W @ g
        return float(np.sum(s * log_ndtr(eta) + (1 - s) * log_ndtr(-eta)))

    ll_cur = loglik_at(gamma)
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
        # step-halving line search: guards against (quasi-)separation, where
        # an undamped Newton step overshoots and the iteration diverges
        scale = 1.0
        for _half in range(30):
            gamma_new = gamma - scale * step
            ll_new = loglik_at(gamma_new)
            if np.isfinite(ll_new) and ll_new >= ll_cur - 1e-12:
                break
            scale *= 0.5
        else:
            gamma_new = gamma
            ll_new = ll_cur
        if np.max(np.abs(gamma_new - gamma)) < tol:
            gamma = gamma_new
            converged = True
            break
        gamma = gamma_new
        ll_cur = ll_new
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
        """Selected-sample conditional mean E[y | x, V=1]."""
        return (_as_2d(X) @ self.beta
                + self.rho * self.sigma * inverse_mills(_as_2d(W) @ self.gamma))

    def predict_hidden(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Non-selected conditional mean E[y | x, V=0]: the reliance the
        naive verbalization count silently codes as zero."""
        eta = _as_2d(W) @ self.gamma
        return _as_2d(X) @ self.beta - self.rho * self.sigma * inverse_mills(-eta)


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
                       ko: int, ks: int,
                       fixed_rho: float | None = None) -> torch.Tensor:
    """Negative joint log-likelihood (Heckman 1979, bivariate normal).

    theta = [beta (ko), gamma (ks), log_sigma, atanh_rho]; when fixed_rho is
    given theta = [beta, gamma, log_sigma] and rho is frozen at fixed_rho.
    y1/X1/W1: selected observations; X0w: selection design of non-selected.
    """
    beta = theta[:ko]
    gamma = theta[ko:ko + ks]
    sigma = torch.exp(theta[ko + ks])
    if fixed_rho is None:
        rho = torch.tanh(theta[ko + ks + 1])
    else:
        rho = torch.tensor(float(fixed_rho), dtype=theta.dtype)

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


def _mle_optimize(nll_fn, theta0: np.ndarray, gtol: float = 1e-9):
    """BFGS + damped-Newton polish; returns (theta, nll, grad_norm, cov)."""

    def fun_and_grad(th_np: np.ndarray):
        th = torch.tensor(th_np, dtype=torch.float64, requires_grad=True)
        nll = nll_fn(th)
        (g,) = torch.autograd.grad(nll, th)
        return float(nll.item()), g.numpy()

    res = optimize.minimize(fun_and_grad, theta0, jac=True, method="BFGS",
                            options={"gtol": gtol, "maxiter": 2000})

    th = torch.tensor(res.x, dtype=torch.float64)
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
                step = np.linalg.solve(Hn + lam_damp * np.eye(len(gn)), gn)
                cand = th - torch.tensor(step)
                if nll_fn(cand).item() <= nll.item() + 1e-12:
                    th = cand
                    break
            except np.linalg.LinAlgError:
                pass
            lam_damp = max(2.0 * lam_damp, 1e-6)
        else:
            break

    final_nll = float(nll_fn(th).item())
    if H is None:
        H = torch.autograd.functional.hessian(nll_fn, th)
    try:
        cov_theta = np.linalg.inv(H.numpy())
    except np.linalg.LinAlgError:
        cov_theta = np.full((len(th), len(th)), np.nan)
    return th.numpy(), final_nll, grad_norm, cov_theta, int(res.nit)


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

    def nll_fn(t):
        return _heckman_nll_torch(t, y1, X1, X0w, W1, ko, ks)

    th_np, final_nll, grad_norm, cov_theta, nit = _mle_optimize(nll_fn, theta0,
                                                                gtol)
    beta, gamma = th_np[:ko], th_np[ko:ko + ks]
    sigma = float(np.exp(th_np[ko + ks]))
    rho = float(np.tanh(th_np[ko + ks + 1]))
    converged = grad_norm < 1e-4 * (1.0 + abs(final_nll))
    return HeckmanResult(beta=beta, gamma=gamma, sigma=sigma, rho=rho,
                         beta_lambda=rho * sigma, loglik=-final_nll,
                         method="mle", converged=converged,
                         extra={"n_iter": nit, "grad_norm": grad_norm,
                                "cov_theta": cov_theta})


def heckman_mle_fixed_rho(y: np.ndarray, X: np.ndarray, s: np.ndarray,
                          W: np.ndarray, rho: float,
                          gtol: float = 1e-9) -> HeckmanResult:
    """Restricted MLE with rho frozen (LR-test null model; rho-sensitivity)."""
    X, W = _as_2d(X), _as_2d(W)
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    sel = s > 0.5
    ko, ks = X.shape[1], W.shape[1]

    start = heckman_two_step(y, X, s, W)
    theta0 = np.concatenate([start.beta, start.gamma,
                             [np.log(max(start.sigma, 1e-6))]])

    y1 = torch.tensor(y[sel], dtype=torch.float64)
    X1 = torch.tensor(X[sel], dtype=torch.float64)
    W1 = torch.tensor(W[sel], dtype=torch.float64)
    X0w = torch.tensor(W[~sel], dtype=torch.float64)

    def nll_fn(t):
        return _heckman_nll_torch(t, y1, X1, X0w, W1, ko, ks, fixed_rho=rho)

    th_np, final_nll, grad_norm, cov_theta, nit = _mle_optimize(nll_fn, theta0,
                                                                gtol)
    beta, gamma = th_np[:ko], th_np[ko:ko + ks]
    sigma = float(np.exp(th_np[ko + ks]))
    converged = grad_norm < 1e-4 * (1.0 + abs(final_nll))
    return HeckmanResult(beta=beta, gamma=gamma, sigma=sigma, rho=float(rho),
                         beta_lambda=float(rho) * sigma, loglik=-final_nll,
                         method=f"mle-fixed-rho({rho:g})", converged=converged,
                         extra={"n_iter": nit, "grad_norm": grad_norm,
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
    """P(V=1 | w) = Phi(w'gamma)."""
    return ndtr(_as_2d(W) @ np.asarray(gamma, dtype=float))


# ------------------------------------------------------------- rho tests


def rho_wald_test(fit: HeckmanResult) -> dict:
    """Wald test of H0: rho = 0 from the MLE observed information.

    Works on the atanh(rho) parametrization (last coordinate of theta), which
    is the parametrization the covariance is computed in; atanh(0) = 0 so the
    null is unchanged.
    """
    if fit.method != "mle":
        raise ValueError("Wald test needs an unrestricted MLE fit")
    cov = fit.extra["cov_theta"]
    a = np.arctanh(np.clip(fit.rho, -0.999999, 0.999999))
    se_a = float(np.sqrt(cov[-1, -1]))
    z = a / se_a
    p = 2.0 * (1.0 - ndtr(abs(z)))
    # delta-method SE for rho itself: d tanh(a)/da = 1 - rho^2
    se_rho = se_a * (1.0 - fit.rho**2)
    return {"stat": float(z), "p": float(p), "se_atanh_rho": se_a,
            "se_rho": float(se_rho),
            "rho_ci95": [float(np.tanh(a - 1.96 * se_a)),
                         float(np.tanh(a + 1.96 * se_a))]}


def rho_lr_test(y, X, s, W, fit_mle: HeckmanResult | None = None) -> dict:
    """Likelihood-ratio test of H0: rho = 0 (chi2, 1 df)."""
    if fit_mle is None:
        fit_mle = heckman_mle(y, X, s, W)
    fit0 = heckman_mle_fixed_rho(y, X, s, W, rho=0.0)
    lr = 2.0 * (fit_mle.loglik - fit0.loglik)
    lr = max(lr, 0.0)
    p = float(stats.chi2.sf(lr, df=1))
    return {"stat": float(lr), "p": p, "loglik_unrestricted": fit_mle.loglik,
            "loglik_rho0": fit0.loglik}


# ----------------------------------------------- binary-outcome selection


def _phi2_torch(a: torch.Tensor, b: torch.Tensor, rho: torch.Tensor,
                n_nodes: int = 48) -> torch.Tensor:
    """Bivariate standard-normal CDF Phi2(a, b; rho), torch-differentiable.

    Uses the tetrachoric integral Phi2 = Phi(a)Phi(b) +
    int_0^rho phi2(a, b; r) dr with Gauss-Legendre nodes rescaled to
    [0, rho], so the result is smooth in a, b and rho. Accurate to ~1e-10
    for |rho| <= 0.99 at 48 nodes. rho may be a scalar tensor or a
    per-observation vector matching a and b.
    """
    t, wq = np.polynomial.legendre.leggauss(n_nodes)
    t = torch.tensor(0.5 * (t + 1.0), dtype=a.dtype)          # [0, 1]
    wq = torch.tensor(0.5 * wq, dtype=a.dtype)
    rho_col = (rho if rho.dim() > 0 else rho[None]).reshape(-1, 1)
    r = rho_col * t[None, :]                       # (n or 1, n_nodes)
    one_m_r2 = 1.0 - r**2
    aa = a[:, None]
    bb = b[:, None]
    dens = torch.exp(-(aa**2 - 2.0 * r * aa * bb + bb**2)
                     / (2.0 * one_m_r2)) / (2.0 * np.pi * torch.sqrt(one_m_r2))
    integral = rho_col.reshape(-1) * (dens * wq).sum(dim=1)
    base = torch.exp(torch.special.log_ndtr(a) + torch.special.log_ndtr(b))
    return torch.clamp(base + integral, min=1e-300)


def _heckprob_nll_torch(theta: torch.Tensor,
                        y1: torch.Tensor, X1: torch.Tensor, W1: torch.Tensor,
                        X0w: torch.Tensor, ko: int, ks: int) -> torch.Tensor:
    """Negative log-likelihood of probit-with-selection (heckprob).

    s=0:        log Phi(-w'gamma)
    s=1, y=1:   log Phi2( x'beta,  w'gamma;  rho)
    s=1, y=0:   log Phi2(-x'beta,  w'gamma; -rho)
    """
    beta = theta[:ko]
    gamma = theta[ko:ko + ks]
    rho = torch.tanh(theta[ko + ks])

    ll0 = torch.special.log_ndtr(-(X0w @ gamma)).sum()

    xb = X1 @ beta
    wg = W1 @ gamma
    q = 2.0 * y1 - 1.0                        # +1 / -1
    p = _phi2_torch(q * xb, wg, q * rho)
    ll1 = torch.log(p).sum()
    return -(ll0 + ll1)


@dataclass
class HeckprobResult:
    beta: np.ndarray
    gamma: np.ndarray
    rho: float
    loglik: float
    converged: bool
    extra: dict = field(default_factory=dict)

    def predict_population(self, X: np.ndarray) -> np.ndarray:
        """P(y=1 | x), selection-corrected."""
        return ndtr(_as_2d(X) @ self.beta)

    def predict_hidden(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """P(y=1 | x, V=0): reliance probability among non-verbalizers."""
        X, W = _as_2d(X), _as_2d(W)
        a = torch.tensor(X @ self.beta, dtype=torch.float64)
        b = torch.tensor(W @ self.gamma, dtype=torch.float64)
        rho = torch.tensor(self.rho, dtype=torch.float64)
        # P(y=1, V=0) = Phi(a) - Phi2(a, b; rho)
        p11 = _phi2_torch(a, b, rho).numpy()
        p_y1 = ndtr(X @ self.beta)
        p_v0 = ndtr(-(W @ self.gamma))
        return np.clip((p_y1 - p11) / np.maximum(p_v0, 1e-12), 0.0, 1.0)

    def predict_observed(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """P(y=1 | x, V=1)."""
        X, W = _as_2d(X), _as_2d(W)
        a = torch.tensor(X @ self.beta, dtype=torch.float64)
        b = torch.tensor(W @ self.gamma, dtype=torch.float64)
        rho = torch.tensor(self.rho, dtype=torch.float64)
        p11 = _phi2_torch(a, b, rho).numpy()
        p_v1 = ndtr(W @ self.gamma)
        return np.clip(p11 / np.maximum(p_v1, 1e-12), 0.0, 1.0)


def heckprob_mle(y: np.ndarray, X: np.ndarray, s: np.ndarray, W: np.ndarray,
                 fixed_rho: float | None = None,
                 gtol: float = 1e-8) -> HeckprobResult:
    """Probit-with-selection MLE (binary reliance proxy, e.g. followed-hint).

    y observed (0/1) iff s == 1. When fixed_rho is given, rho is frozen
    (rho-sensitivity / LR null); theta then omits the atanh-rho coordinate.
    """
    X, W = _as_2d(X), _as_2d(W)
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    sel = s > 0.5
    ko, ks = X.shape[1], W.shape[1]

    # start: separate probits (outcome probit on selected sample)
    pr_sel = probit_fit(W, s)
    pr_out = probit_fit(X[sel], y[sel])
    if fixed_rho is None:
        theta0 = np.concatenate([pr_out.params, pr_sel.params, [0.0]])
    else:
        theta0 = np.concatenate([pr_out.params, pr_sel.params])

    y1 = torch.tensor(y[sel], dtype=torch.float64)
    X1 = torch.tensor(X[sel], dtype=torch.float64)
    W1 = torch.tensor(W[sel], dtype=torch.float64)
    X0w = torch.tensor(W[~sel], dtype=torch.float64)

    if fixed_rho is None:
        def nll_fn(t):
            return _heckprob_nll_torch(t, y1, X1, W1, X0w, ko, ks)
    else:
        ar = float(np.arctanh(np.clip(fixed_rho, -0.999, 0.999)))

        def nll_fn(t):
            t_full = torch.cat([t, torch.tensor([ar], dtype=t.dtype)])
            return _heckprob_nll_torch(t_full, y1, X1, W1, X0w, ko, ks)

    th_np, final_nll, grad_norm, cov_theta, nit = _mle_optimize(nll_fn, theta0,
                                                                gtol)
    beta, gamma = th_np[:ko], th_np[ko:ko + ks]
    rho = float(np.tanh(th_np[ko + ks])) if fixed_rho is None else float(fixed_rho)
    converged = grad_norm < 1e-4 * (1.0 + abs(final_nll))
    return HeckprobResult(beta=beta, gamma=gamma, rho=rho, loglik=-final_nll,
                          converged=converged,
                          extra={"n_iter": nit, "grad_norm": grad_norm,
                                 "cov_theta": cov_theta})


def heckprob_rho_tests(y, X, s, W, fit: HeckprobResult) -> dict:
    """Wald + LR tests of H0: rho = 0 for the heckprob model."""
    cov = fit.extra["cov_theta"]
    a = np.arctanh(np.clip(fit.rho, -0.999999, 0.999999))
    se_a = float(np.sqrt(cov[-1, -1]))
    z = a / se_a
    wald_p = 2.0 * (1.0 - ndtr(abs(z)))
    fit0 = heckprob_mle(y, X, s, W, fixed_rho=0.0)
    lr = max(2.0 * (fit.loglik - fit0.loglik), 0.0)
    lr_p = float(stats.chi2.sf(lr, df=1))
    return {"wald_stat": float(z), "wald_p": float(wald_p),
            "rho_ci95": [float(np.tanh(a - 1.96 * se_a)),
                         float(np.tanh(a + 1.96 * se_a))],
            "lr_stat": float(lr), "lr_p": lr_p}


# ------------------------------------------------------------- estimands


def estimands(fit: HeckmanResult, y: np.ndarray, s: np.ndarray,
              X: np.ndarray, W: np.ndarray) -> dict:
    """Naive vs corrected reliance estimands (linear outcome).

    naive_selected:  mean observed reliance among verbalizers, E_hat[R | V=1]
                     -- what you conclude reading only hint-mentioning CoTs.
    naive_zerofill:  mean of V*R -- the verbalization-count logic that codes
                     every non-verbalizing CoT as "did not rely".
    corrected_pop:   selection-corrected population mean, mean_i x_i'beta.
    corrected_hidden: selection-corrected E_hat[R | V=0] averaged over the
                     non-selected instances (the quantity naive metrics set
                     to zero / never observe).
    """
    X, W = _as_2d(X), _as_2d(W)
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    sel = s > 0.5
    out = {
        "n": int(len(s)),
        "n_selected": int(sel.sum()),
        "select_rate": float(sel.mean()),
        "naive_selected": float(np.mean(y[sel])) if sel.any() else np.nan,
        "naive_zerofill": float(np.mean(np.where(sel, y, 0.0))),
        "corrected_pop": float(np.mean(fit.predict_outcome(X))),
        "corrected_hidden": (float(np.mean(fit.predict_hidden(X, W)[~sel]))
                             if (~sel).any() else np.nan),
        "corrected_observed": (float(np.mean(fit.predict_observed(X, W)[sel]))
                               if sel.any() else np.nan),
        "rho": fit.rho, "sigma": fit.sigma, "method": fit.method,
    }
    return out


def ground_truth_targets(y_full: np.ndarray, s: np.ndarray) -> dict:
    """Oracle targets when the outcome is measured for ALL instances
    (open-model validation): population mean and hidden (V=0) mean."""
    s = np.asarray(s, dtype=float)
    sel = s > 0.5
    y_full = np.asarray(y_full, dtype=float)
    return {
        "true_pop": float(np.mean(y_full)),
        "true_hidden": float(np.mean(y_full[~sel])) if (~sel).any() else np.nan,
        "true_selected": float(np.mean(y_full[sel])) if sel.any() else np.nan,
    }


def rho_sensitivity(y, X, s, W,
                    rho_grid=(-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9)) -> list:
    """Corrected estimands across a fixed-rho grid (identification fallback)."""
    rows = []
    for r in rho_grid:
        fit = heckman_mle_fixed_rho(y, X, s, W, rho=float(r))
        row = estimands(fit, y, s, X, W)
        row["rho_fixed"] = float(r)
        row["loglik"] = fit.loglik
        rows.append(row)
    return rows


def bootstrap_fit(y, X, s, W, y_full=None, n_boot: int = 1000,
                  seed: int = 0, method: str = "two-step") -> dict:
    """Nonparametric bootstrap over instances.

    Returns percentile CIs for rho and each estimand (and, when y_full is
    given, for the estimation errors vs the oracle targets).
    """
    rng = np.random.default_rng(seed)
    X, W = _as_2d(X), _as_2d(W)
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(s)
    keys = ["rho", "naive_selected", "naive_zerofill", "corrected_pop",
            "corrected_hidden"]
    draws = {k: [] for k in keys}
    err_draws = {"err_naive_selected": [], "err_corrected_pop": [],
                 "err_naive_zerofill": [],
                 "err_hidden_naive0": [], "err_hidden_corrected": []}
    n_fail = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if s[idx].sum() < X.shape[1] + 2 or s[idx].sum() > n - 2:
            n_fail += 1
            continue
        try:
            if method == "two-step":
                fit = heckman_two_step(y[idx], X[idx], s[idx], W[idx])
            else:
                fit = heckman_mle(y[idx], X[idx], s[idx], W[idx])
            est = estimands(fit, y[idx], s[idx], X[idx], W[idx])
        except (np.linalg.LinAlgError, ValueError):
            n_fail += 1
            continue
        for k in keys:
            draws[k].append(est[k])
        if y_full is not None:
            tgt = ground_truth_targets(np.asarray(y_full)[idx], s[idx])
            err_draws["err_naive_selected"].append(
                est["naive_selected"] - tgt["true_pop"])
            err_draws["err_naive_zerofill"].append(
                est["naive_zerofill"] - tgt["true_pop"])
            err_draws["err_corrected_pop"].append(
                est["corrected_pop"] - tgt["true_pop"])
            err_draws["err_hidden_naive0"].append(0.0 - tgt["true_hidden"])
            err_draws["err_hidden_corrected"].append(
                est["corrected_hidden"] - tgt["true_hidden"])
    out = {"n_boot": n_boot, "n_fail": n_fail, "method": method}
    for k, v in list(draws.items()) + list(err_draws.items()):
        if len(v) == 0:
            continue
        arr = np.asarray(v, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        out[k] = {"mean": float(arr.mean()),
                  "lo95": float(np.percentile(arr, 2.5)),
                  "hi95": float(np.percentile(arr, 97.5))}
    return out
