"""Decoder-only transformer language model with a pluggable positional scheme."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .config import ModelConfig
from .positions import build_position_module


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, cfg: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, pos_module, positions, attn_bias):
        xn = self.ln1(x)
        if pos_module.per_layer_bias:
            attn_bias = pos_module.attn_bias_layer(xn, self.layer_idx)
        x = x + self.drop(self.attn(xn, pos_module, positions, attn_bias))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = build_position_module(cfg)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, layer_idx=i) for i in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 style scaled init on residual projections
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    def _init_weights(self, m):
        if getattr(m, "_skip_custom_init", False):
            return  # e.g. CABLE's bias projections keep PyTorch default init
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.shape
        device = idx.device
        positions = torch.arange(T, device=device)

        tok = self.tok_emb(idx)                      # (B, T, d)
        x = self.pos.embed(tok, positions)           # add absolute PE if applicable
        x = self.drop(x)

        attn_bias = None
        if self.pos.produces_bias and not self.pos.per_layer_bias:
            # SemRF uses raw token content; ALiBi/T5 ignore tok and use T only.
            # Per-layer schemes (CABLE) compute theirs inside each Block.
            attn_bias = self.pos.attn_bias(tok, T, device)

        for blk in self.blocks:
            x = blk(x, self.pos, positions, attn_bias)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = self._loss(logits, targets, loss_mask)
        return logits, loss

    @staticmethod
    def _loss(logits, targets, loss_mask):
        B, T, V = logits.shape
        ce = F.cross_entropy(
            logits.view(-1, V), targets.reshape(-1), reduction="none"
        ).view(B, T)
        if loss_mask is not None:
            m = loss_mask.to(ce.dtype)
            denom = m.sum().clamp(min=1.0)
            return (ce * m).sum() / denom
        return ce.mean()

    def num_params(self, trainable_only=True):
        return sum(
            p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only)
        )
