r"""Differentiable level-crossing rate estimators for TEMPORAL traces.

Vendored from the Kac-Rice program with a temporal adaptation:
    origin: E:/GitHub/Research/kac-rice/src/kacrice/crossing.py
    (howe2026levelcrossing; "Level-Crossing Density as a Mesh-Free
    High-Frequency Auxiliary Loss for Implicit Neural Representations").
The spatial functions (gaussian_delta, make_levels, crossing_density) are kept
verbatim in semantics; the temporal wrappers below adapt them to sampled
activation traces a[t], where the "gradient norm" is |da/dt| by finite
differences. The trace is data in t but differentiable in the NETWORK
PARAMETERS, which is what the training budget needs.

Two smoothed estimators are provided:

1. Midpoint (co-area Monte-Carlo, the direct vendored form):
       c(u) ~= (1/T) sum_t delta_eps( m_t - u ) * |a_{t+1} - a_t| / dt,
   with m_t the interval midpoint. Underestimates when a single step jumps
   by many eps (the bump is evaluated once on a long segment).

2. Segment (exact for piecewise-linear interpolation):
   for a linear segment a_t -> a_{t+1},
       int_seg delta_eps(a(s) - u) |a'(s)| ds
           = | Phi((a_{t+1}-u)/eps) - Phi((a_t-u)/eps) |,
   with Phi the standard normal CDF. Summed over intervals and divided by the
   window length this is a smoothed crossing RATE that converges to the hard
   sign-change count as eps -> 0, exactly, and stays differentiable in the
   trace values. This is the default for coarsely sampled activation traces.

Rates are per unit time (per step when dt=1). Both estimators count up- and
down-crossings, matching Rice's mu(u) = (1/pi) sqrt(l2/l0) exp(-u^2/(2 l0)).
"""

import math

import torch

SQRT2 = math.sqrt(2.0)


def gaussian_delta(z, eps):
    """Smoothed Dirac: Gaussian kernel with bandwidth (std) eps. [vendored]"""
    return torch.exp(-0.5 * (z / eps) ** 2) / (eps * math.sqrt(2.0 * math.pi))


def make_levels(values, n_levels=16, lo=0.02, hi=0.98):
    """Crossing levels at quantiles of the value distribution. [vendored]

    Quantile placement keeps every level populated regardless of range or
    histogram (paper3 Rule 1: crossing objectives only receive gradient near
    active levels).
    """
    q = torch.linspace(lo, hi, n_levels, device=values.device, dtype=values.dtype)
    flat = values.detach().flatten().float()
    if flat.numel() > 1_000_000:  # torch.quantile has an input-size limit
        idx = torch.randint(0, flat.numel(), (1_000_000,), device=flat.device)
        flat = flat[idx]
    return torch.quantile(flat, q.float()).to(values.dtype)


def crossing_density(values, grad_norm, levels, eps):
    """Monte-Carlo Kac-Rice integrand (spatial form). [vendored]

    values: (N,), grad_norm: (N,), levels: (L,) -> (L,) densities.
    """
    z = values.unsqueeze(0) - levels.unsqueeze(1)  # (L, N)
    return (gaussian_delta(z, eps) * grad_norm.unsqueeze(0)).mean(dim=1)


# --------------------------------------------------------------------------
# Temporal adaptation
# --------------------------------------------------------------------------


def _norm_cdf(z):
    return 0.5 * (1.0 + torch.erf(z / SQRT2))


def crossing_rate_midpoint(traces, levels, eps, dt=1.0):
    """Midpoint co-area estimator on traces (..., T) -> (L,) rates/unit time.

    All leading dims are pooled (channels/batch): the returned rate is the
    mean over traces of each trace's crossing rate.
    """
    mid = 0.5 * (traces[..., 1:] + traces[..., :-1])       # (..., T-1)
    slope = (traces[..., 1:] - traces[..., :-1]).abs() / dt
    z = mid.reshape(1, -1) - levels.reshape(-1, 1)          # (L, N)
    return (gaussian_delta(z, eps) * slope.reshape(1, -1)).mean(dim=1)


def crossing_rate_segment(traces, levels, eps, dt=1.0):
    """Segment (piecewise-linear-exact) smoothed crossing rate.

    traces: (..., T); levels: (L,). Returns (L,) rates per unit time, pooled
    over leading dims. eps -> 0 recovers the hard sign-change count exactly.
    Differentiable in `traces`.
    """
    a0 = traces[..., :-1].reshape(1, -1)                    # (1, N)
    a1 = traces[..., 1:].reshape(1, -1)
    u = levels.reshape(-1, 1)                               # (L, 1)
    seg = (_norm_cdf((a1 - u) / eps) - _norm_cdf((a0 - u) / eps)).abs()
    return seg.mean(dim=1) / dt


def crossing_rate_segment_per_channel(traces, levels, eps, dt=1.0):
    """Per-channel segment rates. traces: (B, T, C) -> (C, L)."""
    B, T, C = traces.shape
    a = traces.permute(2, 0, 1).reshape(C, -1, T)           # (C, B, T)
    a0 = a[..., :-1].reshape(C, 1, -1)                      # (C, 1, N)
    a1 = a[..., 1:].reshape(C, 1, -1)
    u = levels.reshape(1, -1, 1)                            # (1, L, 1)
    seg = (_norm_cdf((a1 - u) / eps) - _norm_cdf((a0 - u) / eps)).abs()
    return seg.mean(dim=2) / dt                             # (C, L)


def crossing_count_hard(traces, levels, dt=1.0):
    """Hard sign-change crossing rate (non-differentiable ground truth).

    traces: (..., T); levels: (L,) -> (L,) mean rates per unit time. A sample
    exactly equal to a level counts via the strict-inequality product rule
    (measure-zero for continuous data).
    """
    a0 = traces[..., :-1].reshape(1, -1)
    a1 = traces[..., 1:].reshape(1, -1)
    u = levels.reshape(-1, 1)
    crossed = ((a0 - u) * (a1 - u) < 0).float()
    return crossed.mean(dim=1) / dt


def crossing_count_hard_per_channel(traces, levels, dt=1.0):
    """Hard crossing rates per channel. traces: (B, T, C) -> (C, L)."""
    a = traces.permute(2, 0, 1)                             # (C, B, T)
    a0 = a[..., :-1].reshape(a.shape[0], 1, -1)
    a1 = a[..., 1:].reshape(a.shape[0], 1, -1)
    u = levels.reshape(1, -1, 1)
    crossed = ((a0 - u) * (a1 - u) < 0).float()
    return crossed.mean(dim=2) / dt
