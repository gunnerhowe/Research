r"""One-sided crossing budget for temporal traces, plus E4 control penalties.

The budget is the howe2026budget mechanism transplanted to its native domain:
delta events ARE level crossings, so excess crossings are excess energy by
construction. Per layer,

    L = mean_j relu( c_hat(u_j) - b_j )^2 / (b_j + mean_k b_k)^2,

with c_hat the SEGMENT smoothed crossing rate of the layer's hidden traces
(differentiable in the network parameters), levels u_j on a quantile ladder of
the layer's calibration activation distribution (paper3 Rule 1: crossing
objectives only receive gradient near populated levels), and budgets b_j a
scaled copy of the calibration profile, b_j = rho * c_cal(u_j) (rho < 1 =
demanded event reduction). Properties inherited from howe2026budget: the loss
and its gradient are exactly zero while the profile is within budget
(one-sidedness), so within-budget dynamics are untouched.
"""

import torch
import torch.nn as nn

from .estimator import crossing_count_hard, crossing_rate_segment, make_levels


class TemporalCrossingBudget(nn.Module):
    """One-sided budget on the crossing-rate profile of (B, T[, C]) traces."""

    def __init__(self, levels, budgets, eps, dt=1.0, relative=True):
        super().__init__()
        self.register_buffer("levels", torch.as_tensor(levels, dtype=torch.float32))
        self.register_buffer("budgets", torch.as_tensor(budgets, dtype=torch.float32))
        self.eps = float(eps)
        self.dt = float(dt)
        self.relative = relative

    def profile(self, traces):
        return crossing_rate_segment(traces, self.levels, self.eps, self.dt)

    def forward(self, traces):
        c = self.profile(traces)
        over = torch.relu(c - self.budgets)
        if self.relative:
            over = over / (self.budgets + self.budgets.mean().clamp_min(1e-8))
        return over.pow(2).mean()

    @classmethod
    def from_calibration(cls, traces, rho, n_levels=16, eps_scale=0.15, dt=1.0):
        """Levels at quantiles of calibration traces; budgets = rho * measured
        hard profile; eps = eps_scale * std(traces)."""
        levels = make_levels(traces, n_levels=n_levels)
        eps = eps_scale * traces.float().std().clamp_min(1e-8).item()
        with torch.no_grad():
            c_cal = crossing_count_hard(traces, levels, dt)
        return cls(levels, rho * c_cal, eps, dt)


class L1DeltaPenalty(nn.Module):
    """E4(i) control: L1 on one-step activation deltas, mean |a_t - a_{t-1}|.
    Shrinks ALL deltas (including the information-carrying large ones);
    contrast with the budget, which prices only crossings and is silent
    within budget."""

    def forward(self, traces):
        return (traces[..., 1:, :] - traces[..., :-1, :]).abs().mean()


class RatePenalty(nn.Module):
    """E4(ii) control: global activity/rate regularizer (spiking-style),
    mean squared activation."""

    def forward(self, traces):
        return traces.pow(2).mean()
