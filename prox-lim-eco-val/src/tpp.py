"""Discrete-time recurrent-intensity temporal point process.

GRU consumes the event-indicator sequence of a single stream; a head maps the
hidden state to a per-step conditional intensity lambda_t (events / MTU).
Log-likelihood (grid discretization of the continuous TPP likelihood, exact as
dt -> 0, and LINEAR in the event weights w so soft differentiable events from
surrogate rollouts drop in natively):

    L = sum_t [ w_t * log(lambda_t) - lambda_t * dt ]

lambda_t is predicted from history strictly before t (h_{t-1}), so the model
is a proper conditional-intensity process. A learned initial hidden state
handles the empty-history start; likelihood over an initial warmup window can
be masked out during fitting.
"""

import torch
import torch.nn as nn


class RecurrentTPP(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, num_layers=layers,
                          batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(layers, 1, hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        # bias init so softplus(head) starts near a plausible rate (~0.1 /MTU)
        nn.init.constant_(self.head[-1].bias, -2.0)

    def intensities(self, w: torch.Tensor) -> torch.Tensor:
        """w: (B, T) event weights in [0,1]. Returns lambda: (B, T), events/MTU.

        lambda_t depends only on w_{<t}: we feed [0, w_1..w_{T-1}] shifted so the
        GRU state at position t encodes history up to t-1.
        """
        B, T = w.shape
        w_in = torch.cat([torch.zeros(B, 1, device=w.device, dtype=w.dtype),
                          w[:, :-1]], dim=1).unsqueeze(-1)  # (B, T, 1)
        h0 = self.h0.expand(-1, B, -1).contiguous()
        out, _ = self.gru(w_in, h0)
        lam = nn.functional.softplus(self.head(out).squeeze(-1))
        return lam + 1e-6

    def log_likelihood(self, w: torch.Tensor, dt: float,
                       warmup: int = 0) -> torch.Tensor:
        """Mean per-step log-likelihood over the batch (masking first `warmup`
        steps). w: (B, T). Returns scalar."""
        lam = self.intensities(w)
        ll = w * torch.log(lam) - lam * dt
        if warmup > 0:
            ll = ll[:, warmup:]
        return ll.mean()
