"""Hardware-faithful noise realization: the intrinsic noise enters the FORWARD
pass (activation / MAC noise), as it does on analog silicon, rather than being
injected on the weights.

This is the realization a neuromorphic chip actually implements: the analog
multiply-accumulate is noisy, so the forward pass -- and hence the gradient -- is
noisy. We show the Doob barrier-conditioning mechanism SURVIVES this realization,
provided the importance is normalized+CLAMPED (an unclamped heavy-tailed Fisher
makes the anchored drift blow the weights up -- the failure mode we hit porting to
hardware). The Doob-steering coupling strength `sig_doob_k` tunes the retention
optimum to any device's intrinsic-noise level, so the mechanism is portable to a
given chip's noise (a few % CV) rather than requiring a specific noise amplitude.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import input_dim, n_head


class NoisyMLP(nn.Module):
    """MLP whose pre-activations carry Gaussian noise during training -- a stand-in
    for the analog MAC's trial-to-trial noise. Noise is off during evaluation."""

    def __init__(self, din, h, dout, n_hidden=2):
        super().__init__()
        dims = [din] + [h] * n_hidden + [dout]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                     for i in range(len(dims) - 1)])
        self.fwd_noise = 0.0

    def _noise(self, z):
        if self.fwd_noise > 0 and self.training:
            z = z + self.fwd_noise * torch.randn_like(z)
        return z

    def forward(self, x):
        for i, lin in enumerate(self.layers):
            z = self._noise(lin(x))
            x = F.relu(z) if i < len(self.layers) - 1 else z
        return x


@torch.no_grad()
def accuracy(model, X, y, bs=4096):
    model.eval(); correct = 0
    for i in range(0, len(X), bs):
        correct += (model(X[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(X)


def diagonal_fisher(model, X, y, bs=256):
    """Model-sampled diagonal Fisher; skips non-finite batches (an unstable,
    unclamped run) so it degrades gracefully instead of crashing."""
    model.eval()
    fish = [torch.zeros_like(p) for p in model.parameters()]
    n = 0
    for i in range(0, len(X), bs):
        xb = X[i:i + bs]
        model.zero_grad()
        logits = model(xb)
        if not torch.isfinite(logits).all():
            continue
        lp = F.log_softmax(logits, 1)
        probs = torch.nan_to_num(lp.exp(), nan=0.0).clamp_min(0.0)
        if (probs.sum(1) <= 0).any():
            continue
        samp = torch.multinomial(probs, 1).squeeze(1)
        lp[torch.arange(len(xb)), samp].sum().backward()
        for f, p in zip(fish, model.parameters()):
            if p.grad is not None:
                f += p.grad.detach() ** 2
        n += len(xb)
    model.zero_grad()
    return [f / max(n, 1) for f in fish]


def run_forward_sequence(testbed, tasks, *, method="doob", sigma=0.0, seed=0,
                         realization="forward", sig_doob_k=1.0, lr=0.1, lr_c=0.1,
                         epochs=2, bs=128, hidden=100, barrier=0.2, kappa=1.0,
                         imp_clip=10.0, max_step_frac=0.25, device="cpu"):
    """Continual-learning sequence with the noise realization selectable:
      'forward' -- noise on the activations (analog MAC noise; the hardware case),
      'weight'  -- noise injected on the weights (the original simulation).
    Importance is normalized to its median and CLAMPED to `imp_clip` (the fix that
    makes the anchored drift stable). Returns (retention, avg_acc, plasticity)."""
    dev = torch.device(device)
    torch.manual_seed(seed)
    m = NoisyMLP(input_dim(testbed), hidden, n_head(testbed)).to(dev)
    tk = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in t.items()} for t in tasks]
    T = len(tk)
    A = np.full((T, T), np.nan)
    mu = s = b = None
    gen = torch.Generator(device=dev); gen.manual_seed(1000 + seed)
    for t in range(T):
        task = tk[t]
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        m.fwd_noise = sigma if (t > 0 and realization == "forward") else 0.0
        sd = sig_doob_k * sigma            # Doob coupling, in the noise units
        for _ in range(epochs):
            idx = torch.randperm(len(task["Xtr"]), generator=gen, device=dev)
            for i in range(0, len(idx), bs):
                j = idx[i:i + bs]
                m.train(); opt.zero_grad()
                F.cross_entropy(m(task["Xtr"][j]), task["ytr"][j]).backward()
                opt.step()
                if t > 0 and mu is not None:
                    with torch.no_grad():
                        for p, mui, si, bi in zip(m.parameters(), mu, s, b):
                            p.add_(-lr_c * si * (p - mui))
                            if method == "doob" and sd > 0 and kappa > 0:
                                z = (p - mui) / bi
                                arg = torch.clamp(0.5 * math.pi * z, -0.5 * math.pi + 1e-4,
                                                  0.5 * math.pi - 1e-4)
                                score = -(math.pi / (2 * bi)) * torch.tan(arg)
                                p.add_(torch.clamp(lr_c * kappa * (sd ** 2) * score,
                                                   -max_step_frac * bi, max_step_frac * bi))
                        if realization == "weight" and sigma > 0:
                            for p in m.parameters():
                                p.add_(sigma * math.sqrt(lr_c) *
                                       torch.randn(p.shape, generator=gen, device=dev))
        m.fwd_noise = 0.0
        for jt in range(t + 1):
            A[t, jt] = accuracy(m, tk[jt]["Xte"], tk[jt]["yte"])
        f = diagonal_fisher(m, task["Xtr"], task["ytr"])
        mu = [p.detach().clone() for p in m.parameters()]
        s = f if s is None else [a + bb for a, bb in zip(s, f)]
        sref = torch.median(torch.cat([si.reshape(-1) for si in s])) + 1e-12
        s = [torch.clamp(si / sref, 0, imp_clip) for si in s]
        b = [torch.clamp(barrier / torch.sqrt(1 + si), 0.05 * barrier, barrier) for si in s]
    final = A[T - 1]
    return (float(np.nanmean(final[:T - 1])), float(np.nanmean(final)),
            float(np.nanmean([A[j, j] for j in range(T)])))
