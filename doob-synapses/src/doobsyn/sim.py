"""Sequential continual-learning trainer and retention metrics.

Protocol (pre-registered, PLAN.md):
  - Task 0 is learned with plain SGD (establish the first memory; nothing to
    protect yet). Tasks 1..T-1 are learned with SGD + the consolidation operator,
    which injects the swept intrinsic noise sigma. So sigma acts only where there
    is a memory to protect, and task-0 accuracy is not confounded by sigma.
  - After each task we snapshot the anchor (current weights) and accumulate the
    diagonal Fisher (online EWC), and derive the per-weight barrier.
  - We record the full accuracy matrix A[t][j] = accuracy on task j after
    training through task t, and report retention / forgetting / plasticity.

Every method sees the IDENTICAL task optimiser and the IDENTICAL injected noise
at a given sigma; only the consolidation drift differs. This is what makes GATE F
a fair isolation of the barrier conditioning.
"""
from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn

from .data import get_tasks
from .diffusion import (Consolidator, ConsolConfig, Memory, BennaFusiState,
                        diagonal_fisher, update_memory)
from .models import build_model, n_params


def _batches(X, y, bs, gen):
    n = X.shape[0]
    idx = torch.randperm(n, generator=gen, device=X.device)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield X[j], y[j]


@torch.no_grad()
def accuracy(model, X, y, bs=2048):
    model.eval()
    correct = 0
    for i in range(0, X.shape[0], bs):
        logits = model(X[i:i + bs])
        correct += (logits.argmax(1) == y[i:i + bs]).sum().item()
    return correct / X.shape[0]


def run_sequence(testbed, tasks, *, method="doob", sigma=0.0, seed=0,
                 lr_task=0.1, lr_c=0.1, epochs=2, batch_size=128,
                 anchor_strength=1.0, kappa=1.0, barrier_scale=0.3, decay=1.0,
                 fisher_batches=8, hidden=None, n_layers=2, device="cpu",
                 replay_buffer=0, benna_fusi=False, bss2_noise=None, quantize=False):
    """Train the full task sequence with one method at one noise level; return the
    accuracy matrix and summary metrics. `tasks` is a list of task dicts (see
    data.py). replay_buffer>0 activates plain reservoir replay; benna_fusi=True
    activates the complex-synapse baseline (both are E3 baselines)."""
    dev = torch.device(device)
    model = build_model(testbed, hidden=hidden, n_layers=n_layers, seed=seed, device=dev)
    # move task tensors to device once
    T = len(tasks)
    tk = []
    for t in tasks:
        tk.append({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in t.items()})
    mem = Memory()
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator(device=dev); gen.manual_seed(1000 + seed)
    A = np.full((T, T), np.nan)
    # optional BSS-2 device-noise emulation (E2): one model per run so the
    # fixed-pattern mismatch is consistent across the task sequence.
    noise_model = None
    if bss2_noise is not None:
        from .bss2 import make_noise_model
        noise_model = make_noise_model(model, bss2_noise, dev, seed=7000 + seed)
    bf = None
    buf_X, buf_y = None, None
    clamp_hits = clamp_total = 0
    t0 = time.time()

    for t in range(T):
        task = tk[t]
        opt = torch.optim.SGD(model.parameters(), lr=lr_task)
        consol = None
        if t > 0 and method != "none":
            cfg = ConsolConfig(method=method, sigma=sigma, lr_c=lr_c,
                               anchor_strength=anchor_strength, kappa=kappa)
            consol = Consolidator(mem, cfg, dev, seed=10 * seed + t,
                                  noise_model=noise_model, quantize=quantize).bind(model)
        elif t > 0 and method == "none" and sigma > 0:
            cfg = ConsolConfig(method="none", sigma=sigma, lr_c=lr_c)
            consol = Consolidator(mem, cfg, dev, seed=10 * seed + t,
                                  noise_model=noise_model, quantize=quantize).bind(model)
        if benna_fusi and t == 0:
            bf = BennaFusiState(model, device=dev)

        for _ep in range(epochs):
            for xb, yb in _batches(task["Xtr"], task["ytr"], batch_size, gen):
                model.train()
                opt.zero_grad(set_to_none=True)
                if replay_buffer > 0 and buf_X is not None and buf_X.shape[0] > 0:
                    ridx = torch.randint(0, buf_X.shape[0], (min(batch_size, buf_X.shape[0]),),
                                         generator=gen, device=dev)
                    x_in = torch.cat([xb, buf_X[ridx]], 0)
                    y_in = torch.cat([yb, buf_y[ridx]], 0)
                else:
                    x_in, y_in = xb, yb
                loss = loss_fn(model(x_in), y_in)
                loss.backward()
                opt.step()
                if consol is not None:
                    consol.step()
                if bf is not None:
                    bf.relax()
        if consol is not None:
            clamp_hits += consol.clamp_hits; clamp_total += consol.clamp_total

        # evaluate on all tasks seen so far
        for j in range(t + 1):
            A[t, j] = accuracy(model, tk[j]["Xte"], tk[j]["yte"])

        # fold this task into the consolidation state (anchor + Fisher + barrier)
        fisher = diagonal_fisher(
            model, loss_fn,
            list(_batches(task["Xtr"], task["ytr"], batch_size, gen)),
            dev, n_batches=fisher_batches)
        update_memory(mem, model, fisher, decay=decay, barrier_scale=barrier_scale)

        # reservoir update for replay baseline
        if replay_buffer > 0:
            xb_all, yb_all = task["Xtr"], task["ytr"]
            take = min(replay_buffer // T, xb_all.shape[0])
            sel = torch.randperm(xb_all.shape[0], generator=gen, device=dev)[:take]
            if buf_X is None:
                buf_X, buf_y = xb_all[sel].clone(), yb_all[sel].clone()
            else:
                buf_X = torch.cat([buf_X, xb_all[sel]], 0)
                buf_y = torch.cat([buf_y, yb_all[sel]], 0)

    wall = time.time() - t0
    out = summarize(A, wall, n_params(model))
    out["clamp_frac"] = (clamp_hits / clamp_total) if clamp_total > 0 else 0.0
    return out


def summarize(A, wall, nparams):
    """Standard CL metrics from the accuracy matrix A[t][j]."""
    T = A.shape[0]
    final = A[T - 1]                                   # accuracy on each task at the end
    diag = np.array([A[j, j] for j in range(T)])       # accuracy right after learning task j
    avg_acc = float(np.nanmean(final))
    # retention = mean final accuracy on the PAST tasks (0..T-2): the forgettable ones
    retention = float(np.nanmean(final[:T - 1]))
    # forgetting = mean drop from best-ever to final on past tasks
    forget = []
    for j in range(T - 1):
        best = np.nanmax(A[j:, j])
        forget.append(best - final[j])
    forgetting = float(np.nanmean(forget)) if forget else float("nan")
    plasticity = float(np.nanmean(diag))               # how well each task is learned
    return {
        "A": A.tolist(),
        "final": final.tolist(),
        "avg_acc": avg_acc,
        "retention": retention,
        "forgetting": forgetting,
        "plasticity": plasticity,
        "wall_s": wall,
        "n_params": nparams,
    }


def sweep_noise(testbed, sigmas, *, method="doob", seeds=(0, 1, 2, 3, 4),
                tasks_kw=None, **run_kw):
    """Retention vs sigma across seeds for one method. Returns
    {sigma: {seed metrics...}} and stacked retention array (n_seed, n_sigma)."""
    tasks_kw = tasks_kw or {}
    out = {"method": method, "sigmas": list(map(float, sigmas)),
           "seeds": list(seeds), "runs": [], "retention": [], "avg_acc": [],
           "plasticity": [], "forgetting": []}
    for s in seeds:
        tasks = get_tasks(testbed, seed=s, **tasks_kw)
        ret_row, avg_row, pla_row, for_row = [], [], [], []
        for sig in sigmas:
            r = run_sequence(testbed, tasks, method=method, sigma=float(sig),
                             seed=s, **run_kw)
            out["runs"].append({"seed": s, "sigma": float(sig), **{k: r[k] for k in
                                ("retention", "avg_acc", "plasticity", "forgetting", "wall_s")}})
            ret_row.append(r["retention"]); avg_row.append(r["avg_acc"])
            pla_row.append(r["plasticity"]); for_row.append(r["forgetting"])
        out["retention"].append(ret_row); out["avg_acc"].append(avg_row)
        out["plasticity"].append(pla_row); out["forgetting"].append(for_row)
    return out
