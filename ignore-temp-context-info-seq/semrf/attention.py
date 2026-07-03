"""Multi-head causal self-attention that consumes an optional rotary transform
and/or an additive attention-bias, so a single implementation serves every
positional scheme.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    # set to True on a layer to store softmax attention weights in self.last_attn
    capture_attn = False
    last_attn = None

    def forward(self, x, pos_module, positions, attn_bias):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                       # (3, B, H, T, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if pos_module.uses_rope:
            q, k = pos_module.rope(q, k, positions)

        if self.capture_attn:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            if attn_bias is not None:
                scores = scores + attn_bias.to(scores.dtype)
            scores = scores.masked_fill(causal, float("-inf"))
            attn = scores.softmax(dim=-1)
            self.last_attn = attn.detach()
            out = attn @ v
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.proj(out)

        p = self.dropout if self.training else 0.0
        if attn_bias is not None:
            # Fold the causal mask into the additive float bias, then let SDPA
            # consume it. (Kernels fall back to the math path for float masks,
            # which is expected for the bias-based schemes.)
            causal = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            mask = attn_bias.to(q.dtype).masked_fill(causal, float("-inf"))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)
