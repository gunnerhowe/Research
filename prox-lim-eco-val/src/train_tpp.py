"""Fit TPPs on ground-truth event streams (and control variants).

Outputs (runs/):
  tpp_gt.pt    — TPP fit on real event timings
  tpp_pois.pt  — TPP fit on Poissonized events (same rate, no structure)
  tpp_shuf.pt  — TPP fit on interval-shuffled events (renewal check)
Also writes data/events.json with the shared event definition (kind, u, tau).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from events import hard_events, poissonize_events, shuffle_intervals, streams
from tpp import RecurrentTPP

ROOT = Path(__file__).resolve().parent.parent


def fit_tpp(ev_streams: torch.Tensor, val_streams: torch.Tensor, dt: float,
            steps: int, win: int, warmup: int, batch: int, device: str,
            seed: int, label: str) -> RecurrentTPP:
    g = torch.Generator().manual_seed(seed)
    model = RecurrentTPP(hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    N, T = ev_streams.shape
    win = min(win, T - 1)
    ev_dev = ev_streams.to(device)
    val_dev = val_streams.to(device)

    def val_ll():
        model.eval()
        with torch.no_grad():
            ll = model.log_likelihood(val_dev.float(), dt, warmup=warmup).item()
        model.train()
        return ll

    for it in range(steps):
        rows = torch.randint(0, N, (batch,), generator=g)
        starts = torch.randint(0, T - win, (batch,), generator=g)
        idx = starts[:, None] + torch.arange(win)[None, :]
        w = ev_dev[rows[:, None], idx].float()
        loss = -model.log_likelihood(w, dt, warmup=warmup)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 500 == 0 or it == steps - 1:
            print(f"  [{label}] it {it}: train NLL/step {loss.item():.5f}  "
                  f"val LL/step {val_ll():.5f}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="neg")
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--tau_frac", type=float, default=0.025,
                    help="soft-event temperature as fraction of std(x)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_traj", type=int, default=0,
                    help="fit TPPs on the first N train trajectories only; "
                         "event threshold u is then READ from the existing "
                         "events.json (fixed task definition), not recomputed")
    ap.add_argument("--data_dir", type=str, default="data")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ddir = (ROOT / args.data_dir) if not Path(args.data_dir).is_absolute() \
        else Path(args.data_dir)

    tr = np.load(ddir / "l96_train.npz")
    va = np.load(ddir / "l96_val.npz")
    X, Xv = torch.from_numpy(tr["X"]), torch.from_numpy(va["X"])
    dt = float(tr["dt_obs"])

    if args.n_traj > 0:
        with open(ddir / "events.json") as f:
            cfg = json.load(f)
        cfg["n_traj_tpp"] = args.n_traj
        X = X[:args.n_traj]
    else:
        from events import observable
        s = observable(X, args.kind)
        u = float(np.quantile(s.numpy(), args.quantile))
        tau = args.tau_frac * X.std().item()
        cfg = dict(kind=args.kind, quantile=args.quantile, u=u, tau=tau, dt=dt)
    with open(ddir / "events.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"event def: {cfg}")
    u = cfg["u"]

    ev = hard_events(X, u, args.kind)      # (B, T-1, K)
    ev_v = hard_events(Xv, u, args.kind)
    rate = ev.float().mean().item()        # per step per stream
    print(f"train events: {int(ev.sum())}  rate/step {rate:.5f} "
          f"({rate/dt:.4f}/site/MTU)")

    # analytic homogeneous-Poisson baseline on val (grid form)
    lam = rate / dt
    ev_v_f = ev_v.float()
    pois_ll = (ev_v_f * math.log(lam) - lam * dt).mean().item()
    print(f"homogeneous-Poisson val LL/step: {pois_ll:.5f}")

    g = torch.Generator().manual_seed(123)
    variants = {
        "gt": ev,
        "pois": poissonize_events(ev, g),
        "shuf": shuffle_intervals(ev, g),
    }
    st_val = streams(ev_v)
    results = {}
    for name, evv in variants.items():
        print(f"fitting tpp_{name} ...")
        model = fit_tpp(streams(evv), st_val, dt, args.steps, args.win,
                        args.warmup, args.batch, device, args.seed, name)
        with torch.no_grad():
            ll = model.log_likelihood(st_val.float().to(device), dt,
                                      warmup=args.warmup).item()
        results[name] = ll
        torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                    "val_ll": ll}, ddir / f"tpp_{name}.pt")
        print(f"tpp_{name}: REAL-event val LL/step = {ll:.5f}")

    print("\nsummary (LL/step on real val events; higher=better):")
    print(f"  homog-Poisson : {pois_ll:.5f}")
    for k, v in results.items():
        print(f"  tpp_{k:5s}   : {v:.5f}")


if __name__ == "__main__":
    main()
