"""Models and constructed twins.

GRULM: 1-layer GRU language model over a small symbol alphabet. Training and clean
teacher-forced passes use the fused nn.GRU; noisy and free-running passes use a manual
GRU step that reproduces torch's gate math exactly (verified in tests), so that noise
can be injected into the hidden state at every step with identical semantics in
generation and in state collection.

Twins (PLAN.md):
- noise-injected: h_t <- h_t + sigma * N(0, I) after every step
- recoded: random permutation of hidden units, function-preserving (the null)
- pruned: global magnitude pruning + masked fine-tuning
- distilled: fresh student trained by KL to the teacher's conditionals
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LN2 = float(np.log(2.0))


# --------------------------------------------------------------------------- GRU

class GRULM(nn.Module):
    def __init__(self, m, width=64):
        super().__init__()
        self.m, self.width = m, width
        self.emb = nn.Embedding(m, width)
        self.rnn = nn.GRU(width, width, batch_first=True)
        self.head = nn.Linear(width, m)

    def forward(self, x, h0=None):
        """Teacher-forced clean pass (fused). x: (B, L) long. Returns logits, states."""
        out, _ = self.rnn(self.emb(x), h0)
        return self.head(out), out


def gru_step(model: GRULM, x_emb, h, sigma=0.0, gen=None):
    """One manual GRU step matching torch semantics; optional hidden-state noise."""
    rnn = model.rnn
    gi = x_emb @ rnn.weight_ih_l0.T + rnn.bias_ih_l0
    gh = h @ rnn.weight_hh_l0.T + rnn.bias_hh_l0
    i_r, i_z, i_n = gi.chunk(3, 1)
    h_r, h_z, h_n = gh.chunk(3, 1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    h_new = (1 - z) * n + z * h
    if sigma > 0:
        h_new = h_new + sigma * torch.randn(h_new.shape, generator=gen,
                                            device=h_new.device)
    return h_new


# ----------------------------------------------------------------- training loop

def _batch(task, B, L, seed):
    return torch.from_numpy(task.sample(B, L, seed=seed).astype(np.int64))

def val_ce_bits(model, task, n_seq=64, L=256, seed=123456, skip=16, sigma=0.0):
    """Held-out next-symbol cross-entropy in bits/symbol (positions >= skip)."""
    model.eval()
    x = _batch(task, n_seq, L + 1, seed).to(DEVICE)
    with torch.no_grad():
        if sigma == 0.0:
            logits, _ = model(x[:, :-1])
        else:
            logits, _, _ = collect_states(model, x[:, :-1], sigma=sigma, seed=seed)
        ce = F.cross_entropy(logits[:, skip:].reshape(-1, model.m),
                             x[:, skip + 1:].reshape(-1))
    return float(ce) / _LN2


def train_lm(model, task, steps=4000, B=64, L=256, lr=1e-3, seed=0, skip=16,
             target_bits=None, check_every=250, masks=None, loss_fn=None):
    """Train by next-symbol CE (or a custom loss_fn(logits, x) for distillation).
    Early-stops when val CE <= target_bits. masks: dict param->0/1 mask re-applied
    after every step (pruned fine-tuning)."""
    torch.manual_seed(seed)
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        x = _batch(task, B, L + 1, seed=seed * 1_000_003 + step).to(DEVICE)
        logits, _ = model(x[:, :-1])
        if loss_fn is None:
            loss = F.cross_entropy(logits[:, skip:].reshape(-1, model.m),
                                   x[:, skip + 1:].reshape(-1))
        else:
            loss = loss_fn(logits, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if masks is not None:
            with torch.no_grad():
                for p, msk in masks.items():
                    p.mul_(msk)
        if target_bits is not None and (step + 1) % check_every == 0:
            if val_ce_bits(model, task) <= target_bits:
                break
            model.train()
    model.eval()
    return model


def train_base(task, width=64, seed=0, rel_tol=0.01, steps=6000):
    """Base model trained to within rel_tol of the task entropy rate."""
    model = _seeded_model(GRULM, task.m, width, seed)
    h = task.entropy_rate()
    train_lm(model, task, steps=steps, seed=seed, target_bits=h * (1 + rel_tol))
    return model


def _seeded_model(cls, *args, **_):
    *cargs, seed = args
    torch.manual_seed(seed)
    return cls(*cargs).to(DEVICE)


# ----------------------------------------------------- generation & state readout

@torch.no_grad()
def generate(model: GRULM, B=64, T=32768, burn=512, sigma=0.0, seed=0,
             record_belief=True):
    """Free-running autoregressive sampling: the process the model defines.

    Returns (symbols (B,T) int8, beliefs (B,T,m) float16 or None).
    """
    model.eval().to(DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    h = torch.zeros(B, model.width, device=DEVICE)
    s = torch.zeros(B, dtype=torch.long, device=DEVICE)
    syms = torch.empty(B, T, dtype=torch.int8, device=DEVICE)
    beliefs = (torch.empty(B, T, model.m, dtype=torch.float16, device=DEVICE)
               if record_belief else None)
    for t in range(burn + T):
        h = gru_step(model, model.emb(s), h, sigma=sigma, gen=gen)
        p = torch.softmax(model.head(h), dim=1)
        s = torch.multinomial(p, 1, generator=gen).squeeze(1)
        if t >= burn:
            syms[:, t - burn] = s.to(torch.int8)
            if record_belief:
                beliefs[:, t - burn] = p.to(torch.float16)
    return syms.cpu().numpy(), (beliefs.cpu().numpy() if record_belief else None)


@torch.no_grad()
def collect_states(model: GRULM, inputs, sigma=0.0, seed=0):
    """Teacher-forced pass with the twin's noise semantics active.

    inputs: (B, L) long on DEVICE. Returns (logits (B,L,m), states (B,L,W), beliefs).
    """
    model.eval()
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    B, L = inputs.shape
    h = torch.zeros(B, model.width, device=DEVICE)
    states = torch.empty(B, L, model.width, device=DEVICE)
    for t in range(L):
        h = gru_step(model, model.emb(inputs[:, t]), h, sigma=sigma, gen=gen)
        states[:, t] = h
    logits = model.head(states)
    return logits, states, torch.softmax(logits, dim=-1)


# ------------------------------------------------------------------------- twins

def recode_permute(model: GRULM, seed=0):
    """Function-preserving twin: random permutation of the hidden units."""
    twin = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(model.width, generator=g).to(DEVICE)
    W = model.width
    with torch.no_grad():
        for blk in range(3):
            rows = slice(blk * W, (blk + 1) * W)
            twin.rnn.weight_ih_l0[rows] = model.rnn.weight_ih_l0[rows][perm]
            twin.rnn.weight_hh_l0[rows] = model.rnn.weight_hh_l0[rows][perm][:, perm]
            twin.rnn.bias_ih_l0[rows] = model.rnn.bias_ih_l0[rows][perm]
            twin.rnn.bias_hh_l0[rows] = model.rnn.bias_hh_l0[rows][perm]
        twin.head.weight.copy_(model.head.weight[:, perm])
    return twin


def prune_finetune(model: GRULM, task, frac=0.5, seed=0, rel_tol=0.005,
                   steps=2000):
    """Global magnitude pruning on {W_ih, W_hh, head.W} + masked fine-tuning."""
    twin = copy.deepcopy(model)
    params = [twin.rnn.weight_ih_l0, twin.rnn.weight_hh_l0, twin.head.weight]
    allw = torch.cat([p.detach().abs().flatten() for p in params])
    thresh = torch.quantile(allw, frac)
    masks = {p: (p.detach().abs() > thresh).float() for p in params}
    with torch.no_grad():
        for p, msk in masks.items():
            p.mul_(msk)
    target = val_ce_bits(model, task) * (1 + rel_tol)
    train_lm(twin, task, steps=steps, seed=seed + 77, target_bits=target,
             masks=masks)
    return twin


def distill(teacher: GRULM, task, width=64, seed=0, rel_tol=0.005, steps=8000,
            B=64, L=256, skip=16):
    """Fresh student trained by KL to the teacher's next-symbol conditionals."""
    student = _seeded_model(GRULM, teacher.m, width, seed + 31)
    teacher.eval()

    def kl_loss(logits, x):
        with torch.no_grad():
            t_logits, _ = teacher(x[:, :-1])
            t_logp = F.log_softmax(t_logits[:, skip:], dim=-1)
        s_logp = F.log_softmax(logits[:, skip:], dim=-1)
        return F.kl_div(s_logp.reshape(-1, teacher.m),
                        t_logp.reshape(-1, teacher.m),
                        log_target=True, reduction="batchmean")

    target = val_ce_bits(teacher, task) * (1 + rel_tol)
    train_lm(student, task, steps=steps, seed=seed + 13, target_bits=target,
             loss_fn=kl_loss, B=B, L=L)
    return student


# ------------------------------------------------------------------- transformer

class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.heads = heads

    def forward(self, x):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = (t.view(B, L, self.heads, -1).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, d))
        return x + self.mlp(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(self, m, d=64, n_layers=2, heads=4, ctx=256):
        super().__init__()
        self.m, self.width, self.ctx = m, d, ctx
        self.emb = nn.Embedding(m, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, m)
        self._sigma = 0.0
        self._gen = None

    def forward(self, x, h0=None):
        B, L = x.shape
        z = self.emb(x) + self.pos(torch.arange(L, device=x.device))[None]
        for blk in self.blocks:
            z = blk(z)
            if self._sigma > 0:
                z = z + self._sigma * torch.randn(z.shape, generator=self._gen,
                                                  device=z.device)
        z = self.ln_f(z)
        return self.head(z), z          # states = final residual stream (post-LN)


@torch.no_grad()
def generate_transformer(model: TransformerLM, task, B=64, T=8192, burn=256,
                         sigma=0.0, seed=0, record_belief=True, window=None):
    """Free-running sampling with a sliding full-recompute window: the emitted
    process is an order-(window-1) stationary Markov process by construction."""
    model.eval().to(DEVICE)
    W = window or (model.ctx - 1)
    model._gen = torch.Generator(device=DEVICE).manual_seed(seed)
    model._sigma = sigma
    gen = model._gen
    buf = torch.from_numpy(task.sample(B, W, seed=seed + 999).astype(np.int64)).to(DEVICE)
    syms = torch.empty(B, T, dtype=torch.int8, device=DEVICE)
    beliefs = (torch.empty(B, T, model.m, dtype=torch.float16, device=DEVICE)
               if record_belief else None)
    for t in range(burn + T):
        logits, _ = model(buf)
        p = torch.softmax(logits[:, -1], dim=-1)
        s = torch.multinomial(p, 1, generator=gen).squeeze(1)
        buf = torch.cat([buf[:, 1:], s[:, None]], dim=1)
        if t >= burn:
            syms[:, t - burn] = s.to(torch.int8)
            if record_belief:
                beliefs[:, t - burn] = p.to(torch.float16)
    model._sigma = 0.0
    return syms.cpu().numpy(), (beliefs.cpu().numpy() if record_belief else None)


@torch.no_grad()
def collect_states_transformer(model: TransformerLM, inputs, sigma=0.0, seed=0):
    model.eval()
    model._gen = torch.Generator(device=DEVICE).manual_seed(seed)
    model._sigma = sigma
    logits, states = model(inputs)
    model._sigma = 0.0
    return logits, states, torch.softmax(logits, dim=-1)


def train_base_transformer(task, seed=0, rel_tol=0.01, steps=5000, d=64,
                           n_layers=2, heads=4, ctx=256):
    torch.manual_seed(seed)
    model = TransformerLM(task.m, d=d, n_layers=n_layers, heads=heads, ctx=ctx).to(DEVICE)
    h = task.entropy_rate()
    train_lm(model, task, steps=steps, lr=3e-4, seed=seed, L=ctx - 1,
             target_bits=h * (1 + rel_tol))
    return model


def distill_transformer(teacher: TransformerLM, task, seed=0, rel_tol=0.005,
                        steps=8000, skip=16):
    torch.manual_seed(seed + 31)
    student = TransformerLM(teacher.m, d=teacher.width,
                            n_layers=len(teacher.blocks),
                            heads=teacher.blocks[0].heads, ctx=teacher.ctx).to(DEVICE)

    def kl_loss(logits, x):
        with torch.no_grad():
            t_logits, _ = teacher(x[:, :-1])
            t_logp = F.log_softmax(t_logits[:, skip:], dim=-1)
        s_logp = F.log_softmax(logits[:, skip:], dim=-1)
        return F.kl_div(s_logp.reshape(-1, teacher.m),
                        t_logp.reshape(-1, teacher.m),
                        log_target=True, reduction="batchmean")

    target = val_ce_bits(teacher, task) * (1 + rel_tol)
    train_lm(student, task, steps=steps, lr=3e-4, seed=seed + 13, L=teacher.ctx - 1,
             target_bits=target, loss_fn=kl_loss)
    return student
