"""Extreme-event extraction: threshold upcrossings of a site-level observable.

PRIMARY observable (see PLAN.md D3): s_k = -x_k (deep troughs), threshold
u = pooled 0.95-quantile of s over ground truth. Site-level events are
strongly sub-Poisson (quasi-periodic recurrence); this is the temporal
structure the TPP learns. Hard events (bool) for TPP fitting and evaluation;
soft events (in [0,1], differentiable) for the auxiliary training signal.

Event at step t (t >= 1): s_t > u AND s_{t-1} <= u.
"""

import torch

OBSERVABLES = {
    "neg": lambda x: -x,        # primary: deep troughs
    "value": lambda x: x,       # positive crests
    "energy": lambda x: x * x,  # info.txt's example (empirically ~Poisson)
}


def observable(x: torch.Tensor, kind: str = "neg") -> torch.Tensor:
    """x: (..., T, K) -> s: (..., T, K)."""
    return OBSERVABLES[kind](x)


def hard_events(x: torch.Tensor, u: float, kind: str = "neg") -> torch.Tensor:
    """Upcrossing indicators. x: (..., T, K) -> (..., T-1, K) bool at t=1..T-1."""
    s = observable(x, kind)
    return (s[..., 1:, :] > u) & (s[..., :-1, :] <= u)


def soft_events(x: torch.Tensor, u: float, tau: float,
                kind: str = "neg") -> torch.Tensor:
    """Differentiable upcrossing weights in [0,1]. x: (..., T, K) -> (..., T-1, K).
    tau is the sigmoid temperature in units of the observable s."""
    s = observable(x, kind)
    above = torch.sigmoid((s[..., 1:, :] - u) / tau)
    below = torch.sigmoid((u - s[..., :-1, :]) / tau)
    return above * below


def streams(ev: torch.Tensor) -> torch.Tensor:
    """(B, T, K) -> (B*K, T): one row per exchangeable site stream."""
    return ev.permute(0, 2, 1).reshape(-1, ev.shape[1])


def intervals_pooled(ev: torch.Tensor, dt: float) -> torch.Tensor:
    """Pooled inter-event times (MTU) across all streams. ev: (B, T, K) bool."""
    iets = []
    st = streams(ev)
    for row in st:
        idx = torch.nonzero(row, as_tuple=False).squeeze(-1)
        if idx.numel() >= 2:
            iets.append(torch.diff(idx).float() * dt)
    if not iets:
        return torch.empty(0)
    return torch.cat(iets)


def poissonize_events(ev: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Poissonized surrogate event sequences: iid Bernoulli per step with the
    POOLED empirical rate (exchangeable streams). Preserves the mean rate,
    destroys all temporal structure. ev: (B, T, K) bool -> (B, T, K) bool."""
    p = ev.float().mean().item()
    return torch.rand(ev.shape, generator=generator) < p


def shuffle_intervals(ev: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Interval-shuffled surrogate event sequences (per stream): preserves each
    stream's IET marginal and event count, destroys serial IET correlations.
    ev: (B, T, K) bool -> (B, T, K) bool."""
    B, T, K = ev.shape
    out = torch.zeros_like(ev)
    for b in range(B):
        for k in range(K):
            idx = torch.nonzero(ev[b, :, k], as_tuple=False).squeeze(-1)
            if idx.numel() < 2:
                out[b, idx, k] = True
                continue
            first = idx[0].item()
            gaps = torch.diff(idx)
            perm = torch.randperm(gaps.numel(), generator=generator)
            new_idx = first + torch.cumsum(
                torch.cat([torch.zeros(1, dtype=gaps.dtype), gaps[perm]]), 0)
            new_idx = new_idx[new_idx < T]
            out[b, new_idx, k] = True
    return out
