"""Kuramoto-Sivashinsky ground truth (secondary system).

u_t = -u u_x - u_xx - u_xxxx on [0, L) periodic, L=22 (strongly chaotic,
attractor dim ~8; Lyapunov time ~20 tu). ETDRK4 (Kassam & Trefethen 2005)
in spectral space, N=64 modes, float64. Observed every dt_obs.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

KS_DEFAULTS = dict(L=22.0, N=64)


def etdrk4_setup(N, L, h, device):
    k = 2 * np.pi / L * torch.fft.fftfreq(N, 1.0 / N).to(device).double()
    lin = k**2 - k**4                       # linear operator eigenvalues
    E = torch.exp(h * lin)
    E2 = torch.exp(h * lin / 2)
    M = 32
    r = torch.exp(2j * np.pi * (torch.arange(M, device=device).double() + 0.5) / M)
    LR = h * lin.cdouble()[:, None] + r[None, :]
    Q = h * ((torch.exp(LR / 2) - 1) / LR).mean(1).real
    f1 = h * ((-4 - LR + torch.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3).mean(1).real
    f2 = h * ((2 + LR + torch.exp(LR) * (-2 + LR)) / LR**3).mean(1).real
    f3 = h * ((-4 - 3 * LR - LR**2 + torch.exp(LR) * (4 - LR)) / LR**3).mean(1).real
    g = -0.5j * k
    return E, E2, Q, f1, f2, f3, g


def etdrk4_step(v, E, E2, Q, f1, f2, f3, g):
    def nl(vv):
        u = torch.fft.ifft(vv, dim=-1).real
        return g * torch.fft.fft(u * u, dim=-1)
    Nv = nl(v)
    a = E2 * v + Q * Nv
    Na = nl(a)
    b = E2 * v + Q * Na
    Nb = nl(b)
    c = E2 * a + Q * (2 * Nb - Nv)
    Nc = nl(c)
    return E * v + Nv * f1 + (Na + Nb) * f2 + Nc * f3


@torch.no_grad()
def integrate(n_traj, t_total, dt_int=0.25, dt_obs=1.0, burn=200.0, seed=0,
              device="cuda", L=22.0, N=64):
    E, E2, Q, f1, f2, f3, g = etdrk4_setup(N, L, dt_int, device)
    gen = torch.Generator().manual_seed(seed)
    x = torch.linspace(0, L, N + 1)[:-1]
    u0 = (0.1 * torch.randn(n_traj, N, generator=gen)).double().to(device)
    v = torch.fft.fft(u0, dim=-1)

    stride = int(round(dt_obs / dt_int))
    assert abs(stride * dt_int - dt_obs) < 1e-9
    for _ in range(int(round(burn / dt_int))):
        v = etdrk4_step(v, E, E2, Q, f1, f2, f3, g)
    n_obs = int(round(t_total / dt_int)) // stride
    out = torch.empty(n_traj, n_obs, N, dtype=torch.float32)
    t0 = time.time()
    for i in range(n_obs):
        for _ in range(stride):
            v = etdrk4_step(v, E, E2, Q, f1, f2, f3, g)
        out[:, i] = torch.fft.ifft(v, dim=-1).real.float().cpu()
        if i > 0 and i % max(1, n_obs // 5) == 0:
            print(f"  obs {i}/{n_obs} ({time.time()-t0:.0f}s)", flush=True)
    if not torch.isfinite(out).all():
        raise RuntimeError("KS integration produced non-finite values")
    return out.numpy()


SPLITS = [  # name, n_traj, length (tu), seed
    ("train", 64, 1600.0, 11),
    ("val", 16, 1600.0, 12),
    ("eval", 64, 4000.0, 13),
    ("eval_long", 8, 15000.0, 14),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/ks")
    ap.add_argument("--dt_obs", type=float, default=1.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for split, ntr, tlen, seed in SPLITS:
        print(f"[{split}] {ntr} traj x {tlen} tu @ dt_obs={args.dt_obs}")
        t0 = time.time()
        X = integrate(ntr, tlen, dt_obs=args.dt_obs, seed=seed, device=dev)
        np.savez_compressed(out / f"l96_{split}.npz", X=X, dt_obs=args.dt_obs,
                            **KS_DEFAULTS)
        print(f"[{split}] shape {X.shape}, {time.time()-t0:.0f}s, "
              f"mean {X.mean():.3f} std {X.std():.3f}", flush=True)


if __name__ == "__main__":
    main()
