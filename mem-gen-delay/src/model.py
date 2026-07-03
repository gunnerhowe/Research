"""1-layer transformer without LayerNorm, following Nanda et al. (2023).

No LayerNorm is a deliberate design choice: the weight-norm causal delay law
(arXiv:2606.13753) shows LayerNorm decouples weight scale from function, which
would make our weight-norm-matched control vacuous.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(att.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        self.attn = Attention(d_model, n_heads)
        self.fc1 = nn.Linear(d_model, d_mlp, bias=False)
        self.fc2 = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.fc2(F.relu(self.fc1(x)))
        return x


class GrokTransformer(nn.Module):
    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4,
                 d_mlp: int = 512, n_layers: int = 1, n_ctx: int = 3):
        super().__init__()
        self.p = p
        self.embed = nn.Embedding(p + 1, d_model)
        self.pos = nn.Parameter(torch.empty(n_ctx, d_model))
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_mlp) for _ in range(n_layers))
        self.unembed = nn.Linear(d_model, p, bias=False)
        nn.init.normal_(self.embed.weight, std=1.0 / math.sqrt(d_model))
        nn.init.normal_(self.pos, std=1.0 / math.sqrt(d_model))

    def forward(self, tokens):
        x = self.embed(tokens) + self.pos
        for blk in self.blocks:
            x = blk(x)
        h = x[:, -1, :]  # final-position residual stream (the rep the readout consumes)
        return self.unembed(h), h


def total_weight_norm(model: nn.Module) -> torch.Tensor:
    """Total L2 norm over all base-model parameters (excludes any aux heads)."""
    return torch.sqrt(sum(p.pow(2).sum() for p in model.parameters()))


@torch.no_grad()
def rescale_to_norm(model: nn.Module, target: float):
    cur = total_weight_norm(model)
    r = target / cur.clamp_min(1e-12)
    for p in model.parameters():
        p.mul_(r)
