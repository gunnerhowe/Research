"""Auxiliary losses for the surrogate training conditions.

- tpp_ratio_loss: the intervention. Reverse-KL-style likelihood-ratio objective
  in TPP space (the TPP analog of Variational Score Distillation):
      L_aux = - E_rollout[ log p_gt_tpp(w) - log p_self_tpp(w) ]
  where w are SOFT events from the differentiable rollout, p_gt_tpp is the TPP
  fit to ground-truth event timings (frozen), and p_self_tpp is a small TPP
  periodically refit by MLE on the surrogate's OWN hard rollout events. Plain
  likelihood (-E[log p_gt_tpp(w)]) is mode-seeking — the empty sequence often
  has the highest density under a fitted TPP, so it collapses the event rate;
  the self-TPP term cancels that pressure and gives a zero-gradient fixed point
  exactly when the surrogate's event process matches ground truth.
  (kept as `tpp_mle` ablation via ratio=False)

- sinkhorn_stats_loss: the marginal/invariant-measure control, faithful to
  Jiang et al. 2023 (NeurIPS): debiased Sinkhorn divergence between point
  clouds of physics-informed summary statistics S(u) = {du/dt,
  (u_{k+1} - u_{k-2}) u_{k-1}, u_k} from rollouts vs. ground truth.
"""

import torch

from events import soft_events


# ---------------------------------------------------------------- TPP aux ---

def tpp_soft_event_streams(x: torch.Tensor, u: float, tau: float) -> torch.Tensor:
    """x: (B, T, K) PHYSICAL-unit states -> soft event streams (B*K, T-1)."""
    w = soft_events(x, u, tau)              # (B, T-1, K)
    return w.permute(0, 2, 1).reshape(-1, w.shape[1])


def tpp_ratio_loss(gt_tpp, self_tpp, w_streams: torch.Tensor, dt: float,
                   warmup: int, ratio: bool = True):
    """Negative (mean per-step) log-likelihood ratio of soft rollout events.
    Both TPPs are frozen here; gradients flow only into w_streams.
    Returns (loss, ll_gt_detached, ll_self_detached)."""
    ll_gt = gt_tpp.log_likelihood(w_streams, dt, warmup=warmup)
    if not ratio:
        return -ll_gt, float(ll_gt), 0.0
    ll_self = self_tpp.log_likelihood(w_streams, dt, warmup=warmup)
    return -(ll_gt - ll_self), float(ll_gt), float(ll_self)


# ------------------------------------------------------- marginal control ---

def l96_summary_stats(x: torch.Tensor, dt: float) -> torch.Tensor:
    """Jiang et al. physics-informed pointwise stats for Lorenz-96.
    x: (B, T, K) physical units -> points (N, 3): {du/dt, advection, u}.
    du/dt by central differences along time."""
    dudt = (x[:, 2:, :] - x[:, :-2, :]) / (2 * dt)          # (B, T-2, K)
    xm = x[:, 1:-1, :]
    adv = (torch.roll(xm, -1, dims=2) - torch.roll(xm, 2, dims=2)) * torch.roll(xm, 1, dims=2)
    pts = torch.stack([dudt, adv, xm], dim=-1)               # (B, T-2, K, 3)
    return pts.reshape(-1, 3)


def ks_summary_stats(x: torch.Tensor, dt: float, dx: float = 22.0 / 64) -> torch.Tensor:
    """Jiang et al. physics-informed pointwise stats for KS:
    {u_t, u_x, u_xx}. x: (B, T, K) -> (N, 3). Ring finite differences."""
    ut = (x[:, 2:, :] - x[:, :-2, :]) / (2 * dt)
    xm = x[:, 1:-1, :]
    ux = (torch.roll(xm, -1, dims=2) - torch.roll(xm, 1, dims=2)) / (2 * dx)
    uxx = (torch.roll(xm, -1, dims=2) - 2 * xm + torch.roll(xm, 1, dims=2)) / dx**2
    return torch.stack([ut, ux, uxx], dim=-1).reshape(-1, 3)


STATS_FNS = {"l96": l96_summary_stats, "ks": ks_summary_stats}


def _sinkhorn_cost(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(a, b, p=2).pow(2) / 2


def _sinkhorn_val(a: torch.Tensor, b: torch.Tensor, eps: float,
                  iters: int) -> torch.Tensor:
    """Entropic OT value <T, C> between uniform clouds a (N,d), b (M,d),
    log-domain Sinkhorn, differentiable."""
    C = _sinkhorn_cost(a, b)
    N, M = C.shape
    log_mu = torch.full((N,), -torch.log(torch.tensor(float(N))), device=C.device)
    log_nu = torch.full((M,), -torch.log(torch.tensor(float(M))), device=C.device)
    f = torch.zeros(N, device=C.device)
    g = torch.zeros(M, device=C.device)
    for _ in range(iters):
        f = -eps * torch.logsumexp((g[None, :] - C) / eps + log_nu[None, :], dim=1)
        g = -eps * torch.logsumexp((f[:, None] - C) / eps + log_mu[:, None], dim=0)
    T = torch.exp((f[:, None] + g[None, :] - C) / eps + log_mu[:, None] + log_nu[None, :])
    return (T * C).sum()


def sinkhorn_divergence(a: torch.Tensor, b: torch.Tensor, eps: float,
                        iters: int = 60) -> torch.Tensor:
    """Debiased Sinkhorn divergence S(a,b) = W(a,b) - (W(a,a)+W(b,b))/2."""
    return (_sinkhorn_val(a, b, eps, iters)
            - 0.5 * (_sinkhorn_val(a, a, eps, iters)
                     + _sinkhorn_val(b, b, eps, iters)))


def sinkhorn_stats_loss(x_roll: torch.Tensor, gt_pts_std: torch.Tensor,
                        stat_mean: torch.Tensor, stat_std: torch.Tensor,
                        dt: float, n_pts: int = 1024, eps: float = 0.1,
                        iters: int = 60,
                        generator: torch.Generator | None = None,
                        stats_fn=l96_summary_stats) -> torch.Tensor:
    """x_roll: (B, T, K) physical rollout states (attached). gt_pts_std: (M, 3)
    pre-standardized ground-truth stat cloud (subsampled per call by caller or
    fixed). stat_mean/std: (3,) standardization constants from ground truth."""
    pts = stats_fn(x_roll, dt)
    idx = torch.randint(0, pts.shape[0], (n_pts,), device=pts.device,
                        generator=generator)
    pts = (pts[idx] - stat_mean) / stat_std
    return sinkhorn_divergence(pts, gt_pts_std, eps=eps, iters=iters)
