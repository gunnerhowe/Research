"""Small MLPs for the continual-learning testbeds. Kept deliberately small so
the per-weight SDE sweep (methods x noise levels x seeds) runs in an afternoon on
one GPU."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, n_layers=2, act="relu"):
        super().__init__()
        A = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[act]
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), A()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(testbed, hidden=None, n_layers=2, seed=0, device="cpu"):
    from .data import input_dim, n_head
    torch.manual_seed(seed)
    if hidden is None:
        hidden = 100 if testbed == "split_mnist" else 30
    m = MLP(input_dim(testbed), hidden, n_head(testbed), n_layers=n_layers)
    return m.to(device)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
