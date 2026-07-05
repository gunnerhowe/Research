"""Training-overhead measurement for the crossing budget (house rule 9).

Back-to-back in the same process, on the SC2 model/batch shape:
  (1) task-only fine-tune steps,
  (2) task + per-layer one-sided crossing-budget steps,
plus an estimator-only microbenchmark (segment estimator forward+backward on
one layer's traces). CUDA-synchronized timings, warmup discarded.

Writes results/timing.json.
"""

import time

import torch

from common import CKPT, DEVICE, RESULTS, log, save_json

from exp3_budget_training import make_budget_reg

from eventrice import data as D
from eventrice.delta import GRUClassifier
from eventrice.estimator import crossing_rate_segment, make_levels

N_STEPS = 60
WARMUP = 10
B = 256


def timed_steps(model, x, y, reg, n=N_STEPS):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    ts = []
    for i in range(n + WARMUP):
        xb, yb = x[:B], y[:B]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        if reg is not None:
            logits, traces = model(xb, return_traces=True)
            loss = torch.nn.functional.cross_entropy(logits, yb) + 3.0 * reg(traces)
        else:
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        if i >= WARMUP:
            ts.append(time.perf_counter() - t0)
    t = torch.tensor(ts)
    return dict(mean_ms=float(t.mean() * 1e3), sd_ms=float(t.std() * 1e3))


def main():
    xtr, ytr = D.load_sc2("train", DEVICE)
    xcal = xtr[:1024]
    model = GRUClassifier(40, 128, 2, 35).to(DEVICE)
    model.load_state_dict(torch.load(CKPT / "sc2_base_s0.pt",
                                     weights_only=True))
    reg = make_budget_reg(model, xcal, rho=0.35)

    task_only = timed_steps(model, xtr, ytr, None)
    model.load_state_dict(torch.load(CKPT / "sc2_base_s0.pt",
                                     weights_only=True))
    with_budget = timed_steps(model, xtr, ytr, reg)

    # estimator microbench: one layer's traces, forward+backward
    tr = torch.randn(B, 128, 100, device=DEVICE, requires_grad=True)
    levels = make_levels(tr, 16)
    torch.cuda.synchronize()
    ts = []
    for i in range(N_STEPS + WARMUP):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c = crossing_rate_segment(tr, levels, eps=0.1)
        c.sum().backward()
        tr.grad = None
        torch.cuda.synchronize()
        if i >= WARMUP:
            ts.append(time.perf_counter() - t0)
    t = torch.tensor(ts)

    out = dict(
        batch=B, n_steps=N_STEPS,
        task_only=task_only, with_budget=with_budget,
        overhead_pct=100.0 * (with_budget["mean_ms"] / task_only["mean_ms"] - 1.0),
        estimator_fwd_bwd_ms=dict(mean_ms=float(t.mean() * 1e3),
                                  sd_ms=float(t.std() * 1e3)),
        device=torch.cuda.get_device_name(0),
    )
    save_json(out, RESULTS / "timing.json")
    log(f"task-only {task_only['mean_ms']:.1f}ms, with budget "
        f"{with_budget['mean_ms']:.1f}ms -> overhead {out['overhead_pct']:.1f}%")


if __name__ == "__main__":
    main()
