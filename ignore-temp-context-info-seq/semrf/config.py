"""Configuration dataclasses for models and training.

Everything an experiment needs to be reproduced is captured here and serialized
to JSON alongside results.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class ModelConfig:
    # --- backbone (kept identical across positional variants for fair comparison) ---
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.0
    max_seq_len: int = 512          # buffer size for absolute schemes
    tie_embeddings: bool = True

    # --- which positional / bias mechanism ---
    #   one of: nope | sinusoidal | learned | rope | alibi | t5 | semrf
    position: str = "semrf"

    # --- RoPE ---
    rope_base: float = 10000.0

    # --- T5 relative bias ---
    t5_num_buckets: int = 32
    t5_max_distance: int = 128

    # --- SemRF (ours) ---
    semrf_num_anchors: int = 32
    semrf_anchor_dim: Optional[int] = None   # defaults to d_model
    semrf_res_dim: int = 32
    semrf_tau: float = 1.0
    semrf_use_sem: bool = True               # anchor-affinity term  (a_i . B . a_j)
    semrf_use_res: bool = True               # residual-alignment term
    semrf_use_time: bool = True              # frame-conditioned temporal decay
    semrf_hard: bool = False                 # hard nearest-anchor (straight-through)
    semrf_per_head_content: bool = False     # per-head sem/res terms (else shared)

    def resolved_anchor_dim(self) -> int:
        return self.semrf_anchor_dim or self.d_model

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    steps: int = 5000
    batch_size: int = 32
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)
    eval_every: int = 500
    eval_batches: int = 20
    amp: bool = True                 # bf16 autocast on Ampere+
    amp_dtype: str = "bfloat16"      # bfloat16 | float16
    seed: int = 0
    device: str = "cuda"
    log_every: int = 100
    compile: bool = False            # torch.compile (off by default for stability)

    def to_dict(self):
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d
