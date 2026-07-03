"""Create coarse-timestep data variants by subsampling dt_obs=0.05 splits.

Writes data/dt<X>/l96_<split>.npz and prints event-structure diagnostics on
the coarse grid (rate, IET CV, Fano) so the regime choice is informed.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from events import hard_events, intervals_pooled  # noqa: E402

STRIDES = [int(s) for s in sys.argv[1:]] or [2, 4]


def fano(ev, dt, w_mtu):
    w = int(round(w_mtu / dt))
    if w < 1:
        return float("nan")
    B, T, K = ev.shape
    TT = T // w * w
    if TT < w:
        return float("nan")
    c = ev.float()[:, :TT].reshape(B, -1, w, K).sum(dim=2)
    return float(c.var(unbiased=True) / c.mean())


for stride in STRIDES:
    dt = 0.05 * stride
    out = ROOT / f"data/dt{str(dt).replace('.', '')}"
    out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "eval", "eval_long"]:
        d = np.load(ROOT / f"data/l96_{split}.npz")
        X = d["X"][:, ::stride]
        np.savez_compressed(out / f"l96_{split}.npz", X=X, dt_obs=dt,
                            **{k: d[k] for k in d.files if k not in ("X", "dt_obs")})
    # diagnostics on eval_long at this dt (threshold: q0.95 of -x, same marginal)
    X = torch.from_numpy(np.load(out / "l96_eval_long.npz")["X"])
    u = float(np.quantile((-X).numpy(), 0.95))
    ev = hard_events(X, u, "neg")
    iet = intervals_pooled(ev, dt)
    cv = float(iet.std() / iet.mean())
    rate = float(ev.float().mean() / dt)
    print(f"dt={dt}: u={u:.3f} rate={rate:.4f}/site/MTU  IET mean={iet.mean():.2f} "
          f"CV={cv:.3f}  Fano2={fano(ev, dt, 2.0):.3f} Fano20={fano(ev, dt, 20.0):.3f}")
