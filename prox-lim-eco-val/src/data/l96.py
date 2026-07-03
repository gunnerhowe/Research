"""Two-scale Lorenz-96 ground-truth generation.

dX_k/dt = -X_{k-1}(X_{k-2} - X_{k+1}) - X_k + F - (h c / b) sum_j Y_{j,k}
dY_{j,k}/dt = -c b Y_{j+1,k}(Y_{j+2,k} - Y_{j-1,k}) - c Y_{j,k} + (h c / b) X_k

K slow sites on a ring, J fast per slow site; Y wraps cyclically along the
full J*K chain. Observed state = slow variables only, sampled every dt_obs.

Integration uses RK4 at dt_int with CUDA-graph capture of one observation
interval (launch-overhead-bound at this system size; graphs give ~10x).
dt_int=0.002 validated against 0.001: climatological mean/std/q98(x^2) match
to <0.3%.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

DEFAULTS = dict(K=40, J=10, F=10.0, h=1.0, b=10.0, c=10.0)


def rhs(X: torch.Tensor, Y: torch.Tensor, K: int, J: int, F: float, h: float,
        b: float, c: float):
    """X: (B, K), Y: (B, K*J) laid out as flat cyclic chain [site0: j0..jJ-1, ...]."""
    hcb = h * c / b
    Ysum = Y.view(-1, K, J).sum(dim=2)  # (B, K)
    dX = (torch.roll(X, 1, dims=1) * (torch.roll(X, -1, dims=1) - torch.roll(X, 2, dims=1))
          - X + F - hcb * Ysum)
    Yp1 = torch.roll(Y, -1, dims=1)
    Yp2 = torch.roll(Y, -2, dims=1)
    Ym1 = torch.roll(Y, 1, dims=1)
    Xrep = X.repeat_interleave(J, dim=1)  # (B, K*J)
    dY = -c * b * Yp1 * (Yp2 - Ym1) - c * Y + hcb * Xrep
    return dX, dY


def rk4_step(X, Y, dt, **p):
    k1x, k1y = rhs(X, Y, **p)
    k2x, k2y = rhs(X + 0.5 * dt * k1x, Y + 0.5 * dt * k1y, **p)
    k3x, k3y = rhs(X + 0.5 * dt * k2x, Y + 0.5 * dt * k2y, **p)
    k4x, k4y = rhs(X + dt * k3x, Y + dt * k3y, **p)
    Xn = X + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
    Yn = Y + dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
    return Xn, Yn


@torch.no_grad()
def integrate(n_traj: int, t_total: float, dt_int: float = 0.002,
              dt_obs: float = 0.05, burn: float = 20.0, seed: int = 0,
              device: str = "cuda", **params):
    """Integrate B independent trajectories; return slow vars at the dt_obs grid.

    Returns X_obs: (B, T_obs, K) float32 numpy, burn-in discarded.
    """
    p = {**DEFAULTS, **params}
    K, J = p["K"], p["J"]
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = (torch.randn(n_traj, K, generator=g) * 1.0 + p["F"] / 2).to(device).double()
    Y = (torch.randn(n_traj, K * J, generator=g) * 0.1).to(device).double()

    stride = int(round(dt_obs / dt_int))
    assert abs(stride * dt_int - dt_obs) < 1e-9, "dt_obs must be a multiple of dt_int"
    n_obs = int(round(t_total / dt_int)) // stride

    use_graph = device == "cuda" and torch.cuda.is_available()
    if use_graph:
        # warmup (side stream), then capture one obs interval
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = rk4_step(X, Y, dt_int, **p)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            a, b = X, Y
            for _ in range(stride):
                a, b = rk4_step(a, b, dt_int, **p)
            X.copy_(a)
            Y.copy_(b)

        def step_obs():
            graph.replay()
    else:
        def step_obs():
            nonlocal X, Y
            for _ in range(stride):
                X, Y = rk4_step(X, Y, dt_int, **p)

    n_burn_obs = int(round(burn / dt_int)) // stride
    for _ in range(n_burn_obs):
        step_obs()

    out = torch.empty(n_traj, n_obs, K, dtype=torch.float32)
    t0 = time.time()
    for i in range(n_obs):
        step_obs()
        out[:, i] = X.float().cpu()
        if i > 0 and i % max(1, n_obs // 5) == 0:
            print(f"  obs {i}/{n_obs}  ({time.time()-t0:.0f}s)", flush=True)
    if not torch.isfinite(out).all():
        raise RuntimeError("non-finite values in trajectory — integration blew up")
    return out.numpy()


SPLITS = [
    # name, n_traj, length (MTU), seed
    ("train", 64, 160.0, 1),
    ("val", 16, 160.0, 2),
    ("eval", 64, 400.0, 3),
    ("eval_long", 8, 1500.0, 4),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--dt_obs", type=float, default=0.05)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    for split, ntr, tlen, seed in SPLITS:
        print(f"[{split}] {ntr} traj x {tlen} MTU @ dt_obs={args.dt_obs}")
        t0 = time.time()
        X = integrate(ntr, tlen, dt_obs=args.dt_obs, seed=seed, device=dev)
        np.savez_compressed(out / f"l96_{split}.npz", X=X, dt_obs=args.dt_obs,
                            **DEFAULTS)
        print(f"[{split}] shape {X.shape}, {time.time()-t0:.0f}s, "
              f"mean {X.mean():.3f} std {X.std():.3f}", flush=True)


if __name__ == "__main__":
    main()
