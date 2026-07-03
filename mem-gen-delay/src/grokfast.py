"""Grokfast-EMA gradient filter (Lee et al. 2024, arXiv:2405.20233)."""
import torch


def gradfilter_ema(model, grads: dict | None, alpha: float = 0.98, lamb: float = 2.0) -> dict:
    if grads is None:
        return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    for n, p in model.named_parameters():
        if p.grad is not None:
            grads[n].mul_(alpha).add_(p.grad.detach(), alpha=1.0 - alpha)
            p.grad.add_(grads[n], alpha=lamb)
    return grads
