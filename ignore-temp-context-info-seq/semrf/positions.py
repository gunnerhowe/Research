"""Pluggable positional / attention-bias mechanisms.

Every scheme implements a common interface so the *same* backbone can host any
of them:

    embed(x, positions)          -> x'      (input-additive schemes)
    rope(q, k, positions)        -> q', k'  (rotary schemes)
    attn_bias(tok_emb, T, device)-> bias    (additive-bias schemes) or None

A scheme sets the class flags `adds_to_embedding`, `uses_rope`, `produces_bias`
so the model knows which hooks to call. Only ONE mechanism is active per model.

Schemes
-------
nope        : no positional information at all (baseline; decoder-only self-
              attention is not fully permutation invariant, so this is a real
              baseline, cf. Haviv et al. 2022 / Kazemnejad et al. 2023).
sinusoidal  : fixed sinusoidal absolute embeddings (Vaswani et al. 2017).
learned     : learned absolute position embeddings (GPT-style).
rope        : rotary position embeddings (Su et al. 2021).
alibi       : linear distance bias with per-head slopes (Press et al. 2022).
t5          : bucketed learned relative-position bias (Raffel et al. 2020).
semrf       : Semantic Reference Frames (ours).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

POSITION_TYPES = ["nope", "sinusoidal", "learned", "rope", "alibi", "t5", "cable", "semrf"]


class PositionModule(nn.Module):
    """Base class. Subclasses override only the hooks they need."""

    adds_to_embedding: bool = False
    uses_rope: bool = False
    produces_bias: bool = False
    per_layer_bias: bool = False   # bias recomputed per layer from hidden state
    can_extrapolate: bool = True   # False => hard failure beyond trained length

    def embed(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return x

    def rope(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        return q, k

    def attn_bias(self, tok_emb: torch.Tensor, T: int, device) -> Optional[torch.Tensor]:
        return None

    def attn_bias_layer(self, x_normed: torch.Tensor, layer_idx: int) -> Optional[torch.Tensor]:
        return None


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
class NoPE(PositionModule):
    def __init__(self, cfg: ModelConfig):
        super().__init__()


class Sinusoidal(PositionModule):
    adds_to_embedding = True

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.d = cfg.d_model

    def _table(self, T: int, device, dtype):
        pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(0, self.d, 2, device=device, dtype=torch.float32)
        div = torch.exp(-math.log(10000.0) * i / self.d)
        pe = torch.zeros(T, self.d, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.to(dtype)

    def embed(self, x, positions):
        T = x.shape[1]
        return x + self._table(T, x.device, x.dtype).unsqueeze(0)


class LearnedAbsolute(PositionModule):
    adds_to_embedding = True
    can_extrapolate = False

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.max_len = cfg.max_seq_len
        self.emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        nn.init.normal_(self.emb.weight, std=0.02)

    def embed(self, x, positions):
        T = x.shape[1]
        if T > self.max_len:
            raise ValueError(
                f"learned-absolute positions only trained up to {self.max_len}, got {T}"
            )
        return x + self.emb(positions)[None, :, :]


# --------------------------------------------------------------------------- #
# RoPE
# --------------------------------------------------------------------------- #
class RoPE(PositionModule):
    uses_rope = True

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.head_dim = cfg.d_model // cfg.n_heads
        assert self.head_dim % 2 == 0, "RoPE needs even head_dim"
        inv_freq = 1.0 / (
            cfg.rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, T, device, dtype):
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))          # (T, hd/2)
        emb = torch.cat([freqs, freqs], dim=-1)                    # (T, hd)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def rope(self, q, k, positions):
        # q, k: (B, H, T, hd)
        T = q.shape[2]
        cos, sin = self._cos_sin(T, q.device, q.dtype)            # (T, hd)
        cos = cos[None, None]
        sin = sin[None, None]
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


# --------------------------------------------------------------------------- #
# ALiBi
# --------------------------------------------------------------------------- #
class ALiBi(PositionModule):
    produces_bias = True

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        slopes = torch.tensor(self._get_slopes(cfg.n_heads), dtype=torch.float32)
        self.register_buffer("slopes", slopes, persistent=False)   # (H,)

    @staticmethod
    def _get_slopes(n):
        def pow2(k):
            start = 2 ** (-(2 ** -(math.log2(k) - 3)))
            return [start * (start ** i) for i in range(k)]

        if math.log2(n).is_integer():
            return pow2(n)
        closest = 2 ** math.floor(math.log2(n))
        base = pow2(closest)
        extra = ALiBi._get_slopes(2 * closest)[0::2][: n - closest]
        return base + extra

    def attn_bias(self, tok_emb, T, device):
        pos = torch.arange(T, device=device)
        dist = (pos[None, :] - pos[:, None]).abs().float()         # (T, T)
        bias = -self.slopes.to(device)[:, None, None] * dist[None]  # (H, T, T)
        return bias.unsqueeze(0)                                    # (1, H, T, T)


# --------------------------------------------------------------------------- #
# T5 relative-position bias
# --------------------------------------------------------------------------- #
class T5RelativeBias(PositionModule):
    produces_bias = True

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.num_buckets = cfg.t5_num_buckets
        self.max_distance = cfg.t5_max_distance
        self.rel_bias = nn.Embedding(cfg.t5_num_buckets, cfg.n_heads)
        nn.init.normal_(self.rel_bias.weight, std=0.02)

    def _bucket(self, rel_pos):
        # causal / unidirectional bucketing (all keys are at or before query)
        num_buckets = self.num_buckets
        n = (-rel_pos).clamp(min=0)                                # distance into the past >= 0
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_large = max_exact + (
            torch.log(n.float().clamp(min=1) / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        val_large = torch.minimum(
            val_large, torch.full_like(val_large, num_buckets - 1)
        )
        return torch.where(is_small, n, val_large)

    def attn_bias(self, tok_emb, T, device):
        pos = torch.arange(T, device=device)
        rel = pos[None, :] - pos[:, None]                          # key - query
        buckets = self._bucket(rel)                                # (T, T)
        bias = self.rel_bias(buckets)                              # (T, T, H)
        bias = bias.permute(2, 0, 1).unsqueeze(0)                  # (1, H, T, T)
        return bias


# --------------------------------------------------------------------------- #
# CABLE (Veisi & Mansourian 2025, arXiv:2503.08067)
# --------------------------------------------------------------------------- #
class CableBias(PositionModule):
    """Context-aware biases for length extrapolation (CABLE).

    Faithful to the reference implementation (github.com/axiomlab/cable,
    pos_methods/cable.py, class CABLE — the variant used for their GPT
    results): each layer owns a Linear(d_model -> n_heads); per-token,
    per-head increments -softplus(.) are accumulated by a causal cumulative
    sum S, and the additive bias between query i and key j is S_i - S_j
    (<= 0 for j <= i). The bias is recomputed at every layer from that
    layer's pre-attention (LayerNormed) hidden state, unlike ALiBi/T5/SemRF
    which compute one bias from the token embeddings and share it.
    Parameters use PyTorch's default Linear init, as in the reference.
    """

    produces_bias = True
    per_layer_bias = True

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.layers = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers)]
        )
        for lin in self.layers:
            lin._skip_custom_init = True   # keep reference (PyTorch default) init

    def attn_bias_layer(self, x_normed, layer_idx):
        # x_normed: (B, T, d) — the LayerNormed input to this layer's attention
        inc = -F.softplus(self.layers[layer_idx](x_normed))        # (B, T, H)
        sums = torch.cumsum(inc.float(), dim=1).transpose(1, 2)    # (B, H, T)
        bias = sums.unsqueeze(3) - sums.unsqueeze(2)               # (B, H, T, T): S_i - S_j
        return bias.to(x_normed.dtype)


# --------------------------------------------------------------------------- #
# SemRF (ours)
# --------------------------------------------------------------------------- #
class SemRF(PositionModule):
    """Semantic Reference Frames.

    Token content is projected to an anchor space and softly assigned to K
    learned anchors.  The attention bias between positions i and j is:

        b_ij = g_sem * (alpha_i^T B alpha_j)             # anchor-frame affinity
             + g_res * <W_q r_i, W_k r_j> / sqrt(d_r)    # within-frame residual alignment
             + g_time * ( -slope(i) * |i - j| )          # frame-conditioned time decay

    where r = u - alpha @ A is the residual offset of the token from its
    (soft) anchor centroid, and slope(i) = alpha_i . softplus(S_head) is a
    per-head, per-token temporal decay rate — a semantically conditioned
    generalization of ALiBi.

    The bias is computed once from the token embeddings and shared across
    layers (like ALiBi / T5).  Absolute position never enters.
    """

    produces_bias = True

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        da = cfg.resolved_anchor_dim()
        K = cfg.semrf_num_anchors
        H = cfg.n_heads
        dr = cfg.semrf_res_dim
        self.K, self.H, self.da, self.dr = K, H, da, dr
        self.tau = cfg.semrf_tau
        self.hard = cfg.semrf_hard
        self.per_head = cfg.semrf_per_head_content
        self.use_sem = cfg.semrf_use_sem
        self.use_res = cfg.semrf_use_res
        self.use_time = cfg.semrf_use_time

        self.to_u = nn.Linear(d, da, bias=False)
        self.anchors = nn.Parameter(torch.randn(K, da) / math.sqrt(da))

        nb = H if self.per_head else 1
        if self.use_sem:
            self.B_anchor = nn.Parameter(torch.zeros(nb, K, K))
        if self.use_res:
            self.res_q = nn.Linear(da, nb * dr, bias=False)
            self.res_k = nn.Linear(da, nb * dr, bias=False)
        if self.use_time:
            # Per-head, per-frame raw slope (softplus => non-negative decay).
            # Initialize every frame to ALiBi's per-head slope so the module
            # *starts* at a strong, known-good positional operating point and
            # only differentiates frames as training proceeds.
            slopes = torch.tensor(ALiBi._get_slopes(H), dtype=torch.float32).clamp(min=1e-4)
            raw = torch.log(torch.expm1(slopes))          # inverse softplus
            self.time_slope = nn.Parameter(raw[:, None].repeat(1, K).contiguous())

        # Per-term log-gates. Start with the temporal term on (gate=1) and the
        # content terms near-off (gate~=0.05), so SemRF ~= per-head ALiBi at
        # init and learns to add semantic structure. Order: [sem, res, time].
        self.log_gate = nn.Parameter(torch.tensor([-3.0, -3.0, 0.0]))

    # -- soft/hard anchor assignment --------------------------------------- #
    def assign(self, u):
        sim = u @ self.anchors.t() / self.tau                      # (B, T, K)
        soft = sim.softmax(dim=-1)
        if self.hard:
            idx = sim.argmax(dim=-1)
            hard = F.one_hot(idx, self.K).to(soft.dtype)
            return hard + soft - soft.detach()                     # straight-through
        return soft

    def attn_bias(self, tok_emb, T, device):
        B = tok_emb.shape[0]
        H = self.H
        u = self.to_u(tok_emb)                                     # (B, T, da)
        alpha = self.assign(u)                                     # (B, T, K)
        centroid = alpha @ self.anchors                            # (B, T, da)
        r = u - centroid                                           # (B, T, da)
        gates = torch.exp(self.log_gate)

        out = tok_emb.new_zeros(B, H, T, T)

        if self.use_sem:
            if self.per_head:
                s = torch.einsum("bik,hkl,bjl->bhij", alpha, self.B_anchor, alpha)
            else:
                aB = alpha @ self.B_anchor[0]                      # (B, T, K)
                s = torch.einsum("btk,bsk->bts", aB, alpha).unsqueeze(1)
            out = out + gates[0] * s

        if self.use_res:
            if self.per_head:
                q = self.res_q(r).view(B, T, H, self.dr).transpose(1, 2)
                k = self.res_k(r).view(B, T, H, self.dr).transpose(1, 2)
                s = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.dr)
            else:
                q = self.res_q(r)                                  # (B, T, dr)
                k = self.res_k(r)
                s = torch.einsum("bid,bjd->bij", q, k).unsqueeze(1) / math.sqrt(self.dr)
            out = out + gates[1] * s

        if self.use_time:
            slope = F.softplus(self.time_slope)                    # (H, K)
            tok_slope = torch.einsum("btk,hk->bht", alpha, slope)  # (B, H, T)
            pos = torch.arange(T, device=device)
            dist = (pos[None, :] - pos[:, None]).abs().float()     # (T, T)
            s = -tok_slope.unsqueeze(-1) * dist[None, None]        # (B, H, T, T)
            out = out + gates[2] * s

        return out

    @torch.no_grad()
    def assignment_of(self, tok_emb):
        """Return hard anchor id per token (for interpretability analysis)."""
        u = self.to_u(tok_emb)
        return (u @ self.anchors.t()).argmax(dim=-1)


# --------------------------------------------------------------------------- #
_REGISTRY = {
    "nope": NoPE,
    "sinusoidal": Sinusoidal,
    "learned": LearnedAbsolute,
    "rope": RoPE,
    "alibi": ALiBi,
    "t5": T5RelativeBias,
    "cable": CableBias,
    "semrf": SemRF,
}


def build_position_module(cfg: ModelConfig) -> PositionModule:
    key = cfg.position.lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown position type {cfg.position!r}; choose from {POSITION_TYPES}")
    return _REGISTRY[key](cfg)
