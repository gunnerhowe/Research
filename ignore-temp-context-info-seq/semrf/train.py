"""Generic training loop shared by synthetic and char-LM experiments."""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import torch

from .config import TrainConfig
from .utils import Timer, get_amp_dtype


def make_optimizer(model, cfg: TrainConfig):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas)


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    prog = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    prog = min(1.0, max(0.0, prog))
    coeff = 0.5 * (1 + math.cos(math.pi * prog))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * coeff)


def run_training(
    model,
    cfg: TrainConfig,
    train_batch_fn: Callable[[int], tuple],
    evaluate_fn: Optional[Callable[[int], Dict]] = None,
    log_prefix: str = "",
) -> List[Dict]:
    """Train `model` for cfg.steps.

    train_batch_fn(step) -> (input_ids, targets, loss_mask_or_None)
    evaluate_fn(step)    -> dict of metrics (called every cfg.eval_every)
    Returns the list of logged history records.
    """
    device = cfg.device
    model.to(device).train()
    opt = make_optimizer(model, cfg)
    amp_dtype = get_amp_dtype(cfg.amp_dtype)
    use_scaler = cfg.amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    history: List[Dict] = []
    timer = Timer()

    for step in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)

        x, y, mask = train_batch_fn(step)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=cfg.amp):
            _, loss = model(x, targets=y, loss_mask=mask)

        opt.zero_grad(set_to_none=True)
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        if (step + 1) % cfg.log_every == 0 or step == 0:
            print(
                f"{log_prefix} step {step+1}/{cfg.steps} "
                f"loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} "
                f"({timer.elapsed():.0f}s)"
            )

        if evaluate_fn is not None and ((step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps):
            model.eval()
            metrics = evaluate_fn(step + 1)
            metrics["step"] = step + 1
            metrics["train_loss"] = float(loss.item())
            history.append(metrics)
            print(f"{log_prefix} [eval @ {step+1}] " + " ".join(f"{k}={v}" for k, v in metrics.items() if k != "step"))
            model.train()

    return history
