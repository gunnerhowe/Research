"""Learning-curve surrogates under successive-halving censoring (Paper B).

Curve family (pow3, the standard LC-extrapolation form):
    y_i(t) = a_i - b_i * t^(-c_i),  t = 1..T
    phi_i = (a_i, log b_i, log c_i) ~ N(mu_pop, Sigma_pop)   [latent]
    yhat_it = y_i(t) + sigma_obs * eps_it,  eps iid N(0,1)   [observed]

Successive halving (SH): at rung epochs t_1 < t_2 < ..., keep the top 1/eta
fraction by the OBSERVED value at the rung epoch. Survival therefore selects
on noise: E[eps | survived] > 0 at rung epochs, so survivors' fitted curve
parameters are noise-inflated and their continuations systematically
disappoint a naive per-curve extrapolation -- a Tobit/Heckman structure on
the LATENT curve parameters.

HONEST CRUX (stated up front in the paper): selection is on OBSERVED
prefixes; if prefixes were noiseless the censoring would be ignorable (MAR).
All bias quantified here enters through sigma_obs > 0.

Estimators:
- fit_pow3_ls: naive per-curve nonlinear least squares (the standard
  extrapolating surrogate).
- fit_pow3_tobit: per-curve truncation correction -- adds the survival
  log-probit terms log Phi((y_phi(t_k) - c_k)/sigma) to the NLL, correcting
  the rung observations for having been selected (no population prior).
- EBModel: hierarchical Gaussian population model over phi, in the
  Laplace / sufficient-statistic formulation: each curve is summarized by
  its nonlinear-LS fit phi_hat_i with Gaussian observation covariance
  V_i = sigma^2 (J'J)^-1 (J = fit Jacobian), giving the CLOSED-FORM
  marginal phi_hat_i ~ N(mu, Sigma + V_i). (A naive prior-sample MC
  marginal was tried first and collapses for peaked likelihoods --
  documented failure mode, kept out of the API.) Three fitting modes:
    'all'               all curves' prefixes (killed + survivors): the
                        MAR-correct usage;
    'survivor_naive'    survivors only, no correction (the common practice:
                        train the extrapolator on completed runs);
    'survivor_heckman'  survivors only WITH a population-level survival
                        normalizer - n * log P_psi(survive all rungs) added
                        to the marginal likelihood, i.e. the population-
                        conditional form prod_i p(D_i) / P(survive)^n. This
                        is NOT the full per-curve conditional likelihood
                        (which would also weight each curve's data integral
                        by its own survival probability); the population
                        normalizer suffices for recovering mu. P_psi(survive)
                        is a smooth low-dimensional integral estimated by
                        fixed common-random-number MC over the prior
                        (well-behaved, unlike the data marginal).
  Predictions are closed-form posterior shrinkage: phi_i | phi_hat_i ~
  N(m_i, C_i) with C_i = (Sigma^-1 + V_i^-1)^-1; the predicted final is
  the posterior mean of y_T(phi) by small-sample MC from that Gaussian.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy import optimize
from scipy.special import log_ndtr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG2PI = math.log(2.0 * math.pi)

# Population defaults, loosely LCBench-like validation-accuracy curves:
# a in ~[0.6, 0.95], y(1) = a - b in ~[0.3, 0.7], c in ~[0.3, 1.2].
POP_MU = np.array([0.78, np.log(0.30), np.log(0.6)])
POP_SD = np.array([0.07, 0.35, 0.35])
POP_CORR_AB = 0.3  # better asymptote weakly correlated with larger gain


def default_pop_cov() -> np.ndarray:
    S = np.diag(POP_SD**2)
    S[0, 1] = S[1, 0] = POP_CORR_AB * POP_SD[0] * POP_SD[1]
    return S


def curve_values(phi: np.ndarray, t: np.ndarray) -> np.ndarray:
    """phi (n, 3) = (a, log b, log c); t (T,) -> (n, T) true curve values."""
    a = phi[:, [0]]
    b = np.exp(phi[:, [1]])
    c = np.exp(phi[:, [2]])
    return a - b * np.power(t[None, :], -c)


@dataclass
class SHRun:
    """Result of running SH censoring on a pool of observed curves."""
    rungs: list[int]            # rung epochs (1-based)
    eta: float
    thresholds: list[float]     # observed-value threshold at each rung
    alive: np.ndarray           # (n, K+1) alive[i, k] = alive AFTER rung k
                                # alive[:, 0] = all True
    kill_rung: np.ndarray       # (n,) index of rung where killed, -1 if never


def run_sh(y_obs: np.ndarray, rungs: list[int], eta: float = 3.0) -> SHRun:
    """Standard SH on observed curves: at each rung epoch keep the top 1/eta
    of still-alive configs by observed value at that epoch."""
    n, T = y_obs.shape
    alive = np.ones((n, len(rungs) + 1), dtype=bool)
    kill_rung = np.full(n, -1)
    thresholds = []
    cur = np.ones(n, dtype=bool)
    for k, t in enumerate(rungs):
        vals = y_obs[:, t - 1]
        n_keep = max(1, int(round(cur.sum() / eta)))
        alive_vals = np.sort(vals[cur])[::-1]
        thr = float(alive_vals[n_keep - 1])
        thresholds.append(thr)
        killed = cur & (vals < thr)
        kill_rung[killed] = k
        cur = cur & (vals >= thr)
        alive[:, k + 1] = cur
    return SHRun(rungs=list(rungs), eta=eta, thresholds=thresholds,
                 alive=alive, kill_rung=kill_rung)


# ------------------------------------------------------- per-curve fitting


def _pow3_residuals(p, t, y):
    a, logb, logc = p
    return a - np.exp(logb) * np.power(t, -np.exp(logc)) - y


def fit_pow3_ls(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Naive per-curve least-squares pow3 fit. Returns phi = (a, logb, logc)."""
    a0 = float(y[-1]) + 0.05
    b0 = max(a0 - float(y[0]), 0.02)
    p0 = np.array([a0, np.log(b0), np.log(0.6)])
    res = optimize.least_squares(
        _pow3_residuals, p0, args=(t, y),
        bounds=([0.0, np.log(1e-3), np.log(0.05)],
                [1.5, np.log(3.0), np.log(4.0)]))
    return res.x


def fit_pow3_tobit(t: np.ndarray, y: np.ndarray, rung_epochs: np.ndarray,
                   rung_thresholds: np.ndarray, sigma_obs: float
                   ) -> np.ndarray:
    """Per-curve truncation-corrected fit (Tobit-style, no population prior).

    NLL = 0.5 ||(y - y_phi)/sigma||^2 + sum_k log Phi((y_phi(t_k)-c_k)/sigma)
    The +log Phi terms subtract the log-probability of the survival events
    the naive fit implicitly conditions on, pulling the fitted curve down at
    rung epochs by exactly the selection effect.
    """
    def nll(p):
        r = _pow3_residuals(p, t, y) / sigma_obs
        a, logb, logc = p
        m = a - np.exp(logb) * np.power(rung_epochs, -np.exp(logc))
        surv = log_ndtr((m - rung_thresholds) / sigma_obs)
        return 0.5 * float(r @ r) + float(surv.sum())

    p0 = fit_pow3_ls(t, y)
    res = optimize.minimize(nll, p0, method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-9,
                                     "maxiter": 2000})
    return res.x


def phi_final(phi: np.ndarray, T: int) -> np.ndarray:
    """Predicted final value(s) at epoch T from phi (n,3) or (3,)."""
    phi = np.atleast_2d(phi)
    return curve_values(phi, np.array([float(T)]))[:, 0]


def _pow3_jacobian(phi: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Jacobian of y_phi(t) wrt (a, logb, logc); returns (T, 3)."""
    a, logb, logc = phi
    b, c = np.exp(logb), np.exp(logc)
    tc = np.power(t, -c)
    return np.column_stack([np.ones_like(t),
                            -b * tc,
                            b * tc * np.log(t) * c])


def fit_pow3_with_cov(t: np.ndarray, y: np.ndarray, sigma_obs: float,
                      ridge: float = 1e-7):
    """Naive LS fit + Laplace observation covariance V = sigma^2 (J'J)^-1."""
    phi = fit_pow3_ls(t, y)
    J = _pow3_jacobian(phi, t)
    JtJ = J.T @ J + ridge * np.eye(3)
    V = sigma_obs**2 * np.linalg.inv(JtJ)
    return phi, V




# ------------------------------------------------ hierarchical (EB) model


class EBModel:
    """Hierarchical Gaussian population model over phi = (a, log b, log c),
    Laplace / sufficient-statistic formulation (see module docstring).

    Data enters as per-curve LS fits phi_hat_i with observation covariances
    V_i (from fit_pow3_with_cov); the marginal likelihood
    phi_hat_i ~ N(mu, Sigma + V_i) is closed-form. The 'survivor_heckman'
    mode subtracts the survival log-probability log P_psi(survive all
    rungs), estimated by common-random-number MC over the prior.
    """

    def __init__(self, mu: np.ndarray, cov: np.ndarray, sigma_obs: float):
        self.mu = mu
        self.cov = cov
        self.sigma_obs = sigma_obs

    @classmethod
    def fit(cls, phi_hat: np.ndarray, V: np.ndarray, mode: str = "all",
            survival_groups: list | None = None,
            sigma_obs: float = 0.01,
            steps: int = 1500, lr: float = 0.02, n_samples: int = 8192,
            seed: int = 0) -> "EBModel":
        """survival_groups (for mode='survivor_heckman'): list of
        (rung_epochs, rung_thresholds, n_group) tuples -- one per SH
        bracket. Multiple brackets with different rung schedules act as the
        exclusion restriction (selection pressure varies, the outcome model
        does not); a single group is the weakly identified case (kept for
        the honesty experiment)."""
        assert mode in ("all", "survivor_naive", "survivor_heckman")
        torch.manual_seed(seed)
        n = phi_hat.shape[0]
        ph = torch.as_tensor(phi_hat, dtype=torch.float64, device=DEVICE)
        Vt = torch.as_tensor(V, dtype=torch.float64, device=DEVICE)

        mu = ph.mean(0).clone().requires_grad_(True)
        emp = (ph - ph.mean(0)).T @ (ph - ph.mean(0)) / max(n - 1, 1)
        d0 = torch.sqrt(torch.clamp(torch.diagonal(emp), min=1e-4))
        log_diag = torch.log(d0).clone().requires_grad_(True)
        off = torch.zeros(3, dtype=torch.float64, device=DEVICE,
                          requires_grad=True)

        heck = mode == "survivor_heckman"
        if heck:
            assert survival_groups, "survivor_heckman needs survival_groups"
            groups = [(torch.as_tensor(np.asarray(re, dtype=float),
                                       device=DEVICE),
                       torch.as_tensor(np.asarray(rt, dtype=float),
                                       device=DEVICE),
                       float(ng)) for re, rt, ng in survival_groups]
            eps_fix = torch.randn(n_samples, 3, dtype=torch.float64,
                                  device=DEVICE)

        idx = torch.tril_indices(3, 3, offset=-1)
        opt = torch.optim.Adam([mu, log_diag, off], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        eye = torch.eye(3, dtype=torch.float64, device=DEVICE)
        for _ in range(steps):
            opt.zero_grad()
            L = torch.diag(torch.exp(log_diag))
            L = L.index_put((idx[0], idx[1]), off)
            Sigma = L @ L.T + 1e-10 * eye
            S = Sigma[None] + Vt                       # (n, 3, 3)
            diff = (ph - mu)[..., None]                # (n, 3, 1)
            sol = torch.linalg.solve(S, diff)          # (n, 3, 1)
            quad = (diff * sol).sum(dim=(1, 2))
            logdet = torch.logdet(S)
            nll = 0.5 * (quad + logdet + 3 * LOG2PI).mean()
            if heck:
                phi_s = mu + eps_fix @ L.T             # (S, 3)
                for re, rt, ng in groups:
                    m_vals = phi_s[:, 0:1] - torch.exp(phi_s[:, 1:2])                         * torch.pow(re[None, :], -torch.exp(phi_s[:, 2:3]))
                    lsurv = torch.special.log_ndtr(
                        (m_vals - rt[None, :]) / sigma_obs).sum(-1)
                    log_psurv = torch.logsumexp(lsurv, 0)                         - math.log(n_samples)
                    nll = nll + (ng / n) * log_psurv
            nll.backward()
            opt.step()
            sched.step()

        with torch.no_grad():
            L = torch.diag(torch.exp(log_diag))
            L = L.index_put((idx[0], idx[1]), off)
            Sigma = (L @ L.T).cpu().numpy()
        return cls(mu.detach().cpu().numpy(), Sigma, sigma_obs)

    # -- prediction ---------------------------------------------------------

    def posterior(self, phi_hat: np.ndarray, V: np.ndarray):
        """Closed-form Gaussian posterior per curve: returns (m, C)."""
        Sig_inv = np.linalg.inv(self.cov + 1e-12 * np.eye(3))
        V_inv = np.linalg.inv(V)                        # (n, 3, 3)
        C = np.linalg.inv(Sig_inv[None] + V_inv)
        m = np.einsum("nij,nj->ni", C,
                      Sig_inv @ self.mu + np.einsum("nij,nj->ni", V_inv,
                                                    phi_hat))
        return m, C

    def predict_final(self, phi_hat: np.ndarray, V: np.ndarray, T: int,
                      n_mc: int = 512, seed: int = 0,
                      return_var: bool = False):
        """Posterior mean (and variance) of y_T by MC from N(m_i, C_i)."""
        m, C = self.posterior(phi_hat, V)
        rng = np.random.default_rng(seed)
        Lc = np.linalg.cholesky(C + 1e-12 * np.eye(3))
        eps = rng.standard_normal((phi_hat.shape[0], n_mc, 3))
        phi = m[:, None, :] + np.einsum("nij,nsj->nsi", Lc, eps)
        yT = phi[..., 0] - np.exp(phi[..., 1])             * np.power(float(T), -np.exp(phi[..., 2]))
        mean = yT.mean(axis=1)
        if return_var:
            return mean, yT.var(axis=1)
        return mean


    def predict_final_exact(self, y_prefix: np.ndarray, prefix_len,
                            phi_hat: np.ndarray, V: np.ndarray, T: int,
                            n_mc: int = 1024, seed: int = 0,
                            proposal_scale: float = 1.5,
                            return_var: bool = False):
        """Exact-likelihood posterior prediction of y_T (self-normalized IS).

        The Laplace posterior N(m_i, C_i * proposal_scale^2) is only the
        PROPOSAL; weights use the exact prefix likelihood times the prior,
        so the Laplace summary error (crude for 3-parameter fits on short
        prefixes) cancels. Selection needs no extra term: survival is a
        deterministic function of the observed prefix, so p(phi | data) is
        prior x likelihood regardless of censoring.

        y_prefix: (n, T_max) observed values (only [:prefix_len_i] used);
        prefix_len: int or (n,) array of prefix lengths.
        """
        n = y_prefix.shape[0]
        plen = (np.full(n, prefix_len, dtype=int)
                if np.isscalar(prefix_len) else np.asarray(prefix_len))
        m, C = self.posterior(phi_hat, V)
        rng = np.random.default_rng(seed)
        Lc = np.linalg.cholesky(C + 1e-12 * np.eye(3)) * proposal_scale
        eps = rng.standard_normal((n, n_mc, 3))
        phi = m[:, None, :] + np.einsum("nij,nsj->nsi", Lc, eps)  # (n,S,3)

        dev = DEVICE
        phi_t = torch.as_tensor(phi, dtype=torch.float64, device=dev)
        t_grid = torch.arange(1, y_prefix.shape[1] + 1, dtype=torch.float64,
                              device=dev)
        y_t = torch.as_tensor(np.nan_to_num(y_prefix), dtype=torch.float64,
                              device=dev)
        mask = torch.zeros_like(y_t)
        for i in range(n):
            mask[i, :plen[i]] = 1.0
        a = phi_t[..., 0:1]
        b = torch.exp(phi_t[..., 1:2])
        c = torch.exp(phi_t[..., 2:3])
        curves = a - b * torch.pow(t_grid[None, None, :], -c)   # (n,S,T)
        sig = self.sigma_obs
        loglik = (-0.5 * ((y_t[:, None, :] - curves) / sig) ** 2
                  * mask[:, None, :]).sum(-1)                    # (n,S)

        mu_t = torch.as_tensor(self.mu, dtype=torch.float64, device=dev)
        Sig_inv = torch.as_tensor(np.linalg.inv(self.cov
                                                + 1e-12 * np.eye(3)),
                                  dtype=torch.float64, device=dev)
        dphi = phi_t - mu_t
        logprior = -0.5 * torch.einsum("nsi,ij,nsj->ns", dphi, Sig_inv,
                                       dphi)
        m_t = torch.as_tensor(m, dtype=torch.float64, device=dev)
        Cp_inv = torch.as_tensor(
            np.linalg.inv(C * proposal_scale**2 + 1e-12 * np.eye(3)),
            dtype=torch.float64, device=dev)
        dq = phi_t - m_t[:, None, :]
        logq = -0.5 * torch.einsum("nsi,nij,nsj->ns", dq, Cp_inv, dq)

        logw = loglik + logprior - logq
        w = torch.softmax(logw, dim=1)                           # (n,S)
        yT = (phi_t[..., 0] - torch.exp(phi_t[..., 1])
              * torch.pow(torch.tensor(float(T), dtype=torch.float64,
                                       device=dev),
                          -torch.exp(phi_t[..., 2])))
        mean = (w * yT).sum(1)
        if return_var:
            var = (w * (yT - mean[:, None]) ** 2).sum(1)
            ess = 1.0 / (w ** 2).sum(1)
            return (mean.cpu().numpy(), var.cpu().numpy(),
                    ess.cpu().numpy())
        return mean.cpu().numpy()
