"""Generic INR fitting loop with pluggable auxiliary losses.

Total objective:  MSE(f(x_i), y_i)
                  + kacrice_w * KacRice(f, y at batch points)
                  + sobolev_w * Sobolev(grad f, grad y at batch points)
                  + ffl_w     * FFL(f on grid, y on grid)          [grid required]

Kac-Rice and Sobolev consume only pointwise values/gradients at the (possibly
scattered) training points. FFL additionally needs `ffl_grid`: the regular grid and a
ground-truth grid image — on scattered data that image must come from interpolation,
which is exactly the handicap the spec wants to expose.
"""

import time

import torch

from .crossing import field_and_grad


@torch.no_grad()
def eval_chunked(model, coords, chunk=65536):
    outs = []
    for i in range(0, coords.shape[0], chunk):
        outs.append(model(coords[i : i + chunk]))
    return torch.cat(outs).squeeze(-1)


def fit(
    model,
    points,
    gt_vals,
    gt_grads=None,
    *,
    kacrice=None,
    kacrice_w=0.0,
    sobolev=None,
    sobolev_w=0.0,
    ffl=None,
    ffl_w=0.0,
    ffl_grid=None,  # ((H,W,2) coords, (H,W) gt) for 2D; ((M,1) coords, (M,) gt) 1D
    ffl_crop=None,  # if set, apply FFL on a random crop of this side length
    iters=2000,
    lr=1e-4,
    cosine=True,
    batch=None,
    eval_fn=None,
    eval_every=250,
    seed=0,
    verbose=True,
):
    """Fit model to (points, gt_vals). Returns history list of dicts."""
    torch.manual_seed(seed)
    device = points.device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05)
        if cosine
        else None
    )
    n = points.shape[0]
    needs_grad = (kacrice is not None and kacrice_w > 0) or (
        sobolev is not None and sobolev_w > 0
    )
    if needs_grad and gt_grads is None:
        raise ValueError("gradient-based aux losses need gt_grads")

    history = []
    t0 = time.time()
    for it in range(1, iters + 1):
        if batch is not None and batch < n:
            idx = torch.randint(0, n, (batch,), device=device)
            pts, yv = points[idx], gt_vals[idx]
            yg = gt_grads[idx] if gt_grads is not None else None
        else:
            pts, yv, yg = points, gt_vals, gt_grads

        if needs_grad:
            pred, pgrad = field_and_grad(model, pts)
        else:
            pred, pgrad = model(pts).squeeze(-1), None

        loss_mse = (pred - yv).pow(2).mean()
        loss = loss_mse
        parts = {"mse": loss_mse.item()}

        if kacrice is not None and kacrice_w > 0:
            lk = kacrice(pred, pgrad, yv, yg)
            loss = loss + kacrice_w * lk
            parts["kacrice"] = lk.item()
        if sobolev is not None and sobolev_w > 0:
            ls = sobolev(pgrad, yg)
            loss = loss + sobolev_w * ls
            parts["sobolev"] = ls.item()
        if ffl is not None and ffl_w > 0:
            gc, gt_grid = ffl_grid
            if gt_grid.dim() == 2:
                h, w = gt_grid.shape
                if ffl_crop is not None and ffl_crop < min(h, w):
                    c = ffl_crop
                    i = torch.randint(0, h - c + 1, (1,)).item()
                    j = torch.randint(0, w - c + 1, (1,)).item()
                    gc_c = gc[i : i + c, j : j + c].reshape(-1, gc.shape[-1])
                    tgt = gt_grid[i : i + c, j : j + c]
                    pred_grid = model(gc_c).squeeze(-1)
                    lf = ffl(pred_grid.view(1, 1, c, c), tgt.reshape(1, 1, c, c))
                else:
                    pred_grid = model(gc.reshape(-1, gc.shape[-1])).squeeze(-1)
                    lf = ffl(pred_grid.view(1, 1, h, w), gt_grid.view(1, 1, h, w))
            else:
                pred_grid = model(gc).squeeze(-1)
                lf = ffl(pred_grid.view(1, 1, -1), gt_grid.view(1, 1, -1))
            loss = loss + ffl_w * lf
            parts["ffl"] = lf.item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()

        if it % eval_every == 0 or it == iters:
            rec = {"iter": it, "time": time.time() - t0, **parts}
            if eval_fn is not None:
                model.eval()
                rec.update(eval_fn(model))
                model.train()
            history.append(rec)
            if verbose:
                msg = " ".join(
                    f"{k}={v:.4g}" for k, v in rec.items() if k not in ("iter", "time")
                )
                print(f"  [{it:5d}] {msg}")
    return history
