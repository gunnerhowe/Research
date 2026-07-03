"""Per-iteration timing benchmark backing the Cost paragraph of the paper.

Measures back-to-back training-iteration times for MSE-only, +Sobolev, and
+Kac-Rice on the exp2 configuration (PE-MLP width 256 x 3, 16,384 points, CPU),
same thread count as the experiment suite. Back-to-back measurement on the same
process means shared-machine load affects all configurations comparably; the
ratio is the robust quantity. Writes results/paper/timing.json.

Usage: python experiments/bench_timing.py [--threads 4] [--iters 30]
"""

import argparse
import json
import time

import torch

from common import ROOT
from kacrice.crossing import KacRiceLoss, field_and_grad
from kacrice.losses import SobolevLoss
from kacrice.models import PEMLP


def bench(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters * 1000  # ms/iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--n", type=int, default=16384)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    model = PEMLP(2, 1, 256, 3, n_freqs=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    kr, sob = KacRiceLoss(), SobolevLoss()
    pts = torch.rand(args.n, 2) * 2 - 1
    y = torch.rand(args.n)
    yg = torch.randn(args.n, 2)

    def step_mse():
        pred = model(pts).squeeze(-1)
        loss = (pred - y).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    def step_sobolev():
        pred, pgrad = field_and_grad(model, pts)
        loss = (pred - y).pow(2).mean() + 0.05 * sob(pgrad, yg)
        opt.zero_grad(); loss.backward(); opt.step()

    def step_kacrice():
        pred, pgrad = field_and_grad(model, pts)
        loss = (pred - y).pow(2).mean() + 0.05 * kr(pred, pgrad, y, yg)
        opt.zero_grad(); loss.backward(); opt.step()

    res = {
        "config": {"n_points": args.n, "threads": args.threads,
                   "iters": args.iters, "model": "PEMLP 256x3 nf8",
                   "device": "cpu"},
        "ms_per_iter": {
            "mse": bench(step_mse, args.iters),
            "sobolev": bench(step_sobolev, args.iters),
            "kacrice": bench(step_kacrice, args.iters),
        },
    }
    res["ratio_vs_mse"] = {
        k: v / res["ms_per_iter"]["mse"] for k, v in res["ms_per_iter"].items()
    }
    out = ROOT / "results" / "paper" / "timing.json"
    out.write_text(json.dumps(res, indent=1))
    for k, v in res["ms_per_iter"].items():
        print(f"{k}: {v:.0f} ms/iter ({res['ratio_vs_mse'][k]:.2f}x)")


if __name__ == "__main__":
    main()
