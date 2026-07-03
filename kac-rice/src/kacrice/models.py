"""INR backbones: SIREN, FINER, and a positional-encoding ReLU MLP.

SIREN: Sitzmann et al., "Implicit Neural Representations with Periodic Activation
Functions", NeurIPS 2020. sin(omega_0 * Wx + b) with the standard init: first layer
U(-1/fan_in, 1/fan_in), hidden layers U(-sqrt(6/fan_in)/omega_0, +...).

FINER: Liu et al., CVPR 2024. Variable-periodic activation sin(omega * (|z| + 1) * z)
with the (|z| + 1) factor detached from the graph (as in the official FINER++ code),
SIREN init, and the first layer's bias drawn from U(-k, k) (first_bias_scale) to widen
the supported frequency set.

PEMLP: ReLU MLP on Fourier positional encodings (NeRF-style), the classic backbone
whose spectral bias auxiliary losses are supposed to fight.
"""

import math

import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(self, in_f, out_f, omega=30.0, is_first=False, finer=False,
                 first_bias_scale=None):
        super().__init__()
        self.omega = omega
        self.finer = finer
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_f
            else:
                bound = math.sqrt(6.0 / in_f) / omega
            self.linear.weight.uniform_(-bound, bound)
            if is_first and first_bias_scale is not None:
                self.linear.bias.uniform_(-first_bias_scale, first_bias_scale)

    def forward(self, x):
        z = self.linear(x)
        if self.finer:
            scale = (z.abs() + 1.0).detach()
            return torch.sin(self.omega * scale * z)
        return torch.sin(self.omega * z)


class SIREN(nn.Module):
    def __init__(self, in_features=2, out_features=1, hidden_features=256,
                 hidden_layers=3, omega=30.0, finer=False, first_bias_scale=None):
        super().__init__()
        layers = [SineLayer(in_features, hidden_features, omega, is_first=True,
                            finer=finer, first_bias_scale=first_bias_scale)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, omega,
                                    finer=finer))
        final = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features) / omega
            final.weight.uniform_(-bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def FINER(in_features=2, out_features=1, hidden_features=256, hidden_layers=3,
          omega=30.0, first_bias_scale=None):
    """FINER = SIREN with variable-periodic activation. first_bias_scale=None
    keeps the default bias init, matching the official image-fitting scripts."""
    return SIREN(in_features, out_features, hidden_features, hidden_layers,
                 omega=omega, finer=True, first_bias_scale=first_bias_scale)


class PEMLP(nn.Module):
    """ReLU MLP on Fourier features: [sin(2^k pi x), cos(2^k pi x)] for k < n_freqs."""

    def __init__(self, in_features=2, out_features=1, hidden_features=256,
                 hidden_layers=3, n_freqs=8, include_input=True):
        super().__init__()
        self.n_freqs = n_freqs
        self.include_input = include_input
        enc_dim = in_features * (2 * n_freqs + (1 if include_input else 0))
        layers = [nn.Linear(enc_dim, hidden_features), nn.ReLU(inplace=True)]
        for _ in range(hidden_layers):
            layers += [nn.Linear(hidden_features, hidden_features), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)
        self.register_buffer(
            "freqs", (2.0 ** torch.arange(n_freqs)) * math.pi, persistent=False
        )

    def forward(self, x):
        xb = x.unsqueeze(-1) * self.freqs  # (N, d, K)
        enc = torch.cat([xb.sin(), xb.cos()], dim=-1).flatten(-2)
        if self.include_input:
            enc = torch.cat([x, enc], dim=-1)
        return self.net(enc)


def make_model(name, in_features, out_features=1, **kw):
    name = name.lower()
    if name == "siren":
        return SIREN(in_features, out_features, **kw)
    if name == "finer":
        return FINER(in_features, out_features, **kw)
    if name == "pemlp":
        return PEMLP(in_features, out_features, **kw)
    raise ValueError(f"unknown model {name!r}")
