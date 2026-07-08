"""E1/E0(b): step ablations and directional interventions (M2 only).

Ablation influence I_t (anchor ground truth, PLAN.md sec. 3):
  replace the thought fed at slot t and measure |Delta margin| downstream, where
  margin = log P(target answer) - log P(neg_target answer), teacher-forced.

Directional interventions (E1):
  c_t -> c_t + eps * ||c_t|| * d for unit directions d; spectral directions (top right
  singular vector v1 of J_t; top slow-mode subspace) vs magnitude-matched random unit
  directions.  Both signs are applied; effects averaged.
"""

from __future__ import annotations

import numpy as np
import torch

from .harness import Harness, LatentRun, N_LATENT
from .prosqa import Problem


def ablation_influence(h: Harness, run: LatentRun, problem: Problem,
                       base_margin: float, mean_thoughts: torch.Tensor) -> dict:
    """I_t for t=1..6 under mean / zero / (optionally) donor replacement."""
    out = {"mean": [], "zero": []}
    for t in range(1, N_LATENT + 1):
        r = h.rerun_from(run, problem, t, mean_thoughts[t - 1])
        out["mean"].append(abs(r["margin"] - base_margin))
        r = h.rerun_from(run, problem, t, torch.zeros_like(run.hs[0]))
        out["zero"].append(abs(r["margin"] - base_margin))
    return out


def directional_intervention(h: Harness, run: LatentRun, problem: Problem,
                             base_margin: float, t: int, direction: torch.Tensor,
                             eps_rel: float) -> dict:
    """Perturb c_t along +/- direction (unit vector), norm eps_rel*||c_t||."""
    c = h.fed_vector(run, t)
    scale = eps_rel * c.norm()
    d = direction / direction.norm()
    effects, flips, to_neg = [], [], []
    for sign in (+1.0, -1.0):
        r = h.rerun_from(run, problem, t, c + sign * scale * d)
        effects.append(abs(r["margin"] - base_margin))
        flips.append(r["correct"] != (base_margin > 0))
        to_neg.append(r["pred_neg"])
    return {
        "effect": float(np.mean(effects)),
        "effect_max": float(np.max(effects)),
        "flip_rate": float(np.mean(flips)),
        "to_neg_rate": float(np.mean(to_neg)),
    }


def subspace_ablation(h: Harness, run: LatentRun, problem: Problem,
                      base_margin: float, t: int, Q: torch.Tensor) -> dict:
    """Project the component of c_t in span(Q) out of c_t (Q: 768 x k, orthonormal)."""
    c = h.fed_vector(run, t)
    c_new = c - Q @ (Q.T @ c)
    r = h.rerun_from(run, problem, t, c_new)
    return {
        "effect": abs(r["margin"] - base_margin),
        "flip": bool(r["correct"] != (base_margin > 0)),
        "removed_norm_frac": float((Q.T @ c).norm() / c.norm()),
    }


def random_unit_directions(n: int, dim: int, seed: int, device: str) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = torch.randn(n, dim, generator=g)
    d = d / d.norm(dim=1, keepdim=True)
    return d.to(device)


def random_orthonormal(dim: int, k: int, seed: int, device: str) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    M = torch.randn(dim, k, generator=g)
    Q, _ = torch.linalg.qr(M)
    return Q.to(device)
