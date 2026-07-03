"""Experiment 2 - THE DECISIVE ONE: image fitting from scattered non-uniform samples.

Supervision is ONLY the scattered set {(x_i, y_i)}. Every loss must build its
targets from that:
  * FFL needs a regular grid -> the GT grid image is scattered-interpolated
    (griddata), which smears exactly the high-frequency content it is supposed
    to supervise. This is the structural handicap of frequency-domain losses.
  * Sobolev and Kac-Rice are pointwise/mesh-free. Their GT gradients are
    estimated from the same interpolated grid (no information advantage), but
    Kac-Rice only consumes them through a *distributional* statistic (crossing
    density per level), which should be more robust to pointwise interpolation
    error than Sobolev's per-point gradient matching.

With --oracle, adds diagnostic runs using true image gradients at sample points
(upper bound separating method potential from gradient-estimation noise).

Usage: python experiments/exp2_nonuniform.py [--n-samples 16384] [--mode blobs]
"""

import argparse

import torch

from common import (RESULTS, plot_band_errors, plot_psnr_curves, plot_recons,
                    save_history, summary_table)
from kacrice.crossing import KacRiceLoss
from kacrice.data import (bilinear_sample, grid_coords, image_gradients,
                          load_image, sample_coords, scattered_to_grid)
from kacrice.losses import FocalFrequencyLoss, SobolevLoss
from kacrice.metrics import hf_psnr, psnr, radial_band_error
from kacrice.models import PEMLP
from kacrice.train import eval_chunked, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=16384)
    ap.add_argument("--mode", default="blobs", choices=["uniform", "blobs", "ramp"])
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--image", default="camera")
    ap.add_argument("--kacrice-w", type=float, default=0.05)
    ap.add_argument("--sobolev-w", type=float, default=0.05)
    ap.add_argument("--ffl-w", type=float, default=1.0)
    ap.add_argument("--oracle", action="store_true",
                    help="add diagnostic runs with true gradients")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device
    s = args.size

    img = load_image(args.image, s).to(dev)

    # --- supervision: scattered non-uniform samples only -------------------
    pts = sample_coords(args.n_samples, mode=args.mode, seed=args.seed, device=dev)
    y = bilinear_sample(img, pts)

    # --- derived targets, built ONLY from the scattered data ---------------
    interp_img = scattered_to_grid(pts, y, s).to(dev)          # for FFL
    est_grad_field = image_gradients(interp_img)               # for Sobolev/KacRice
    est_grads = bilinear_sample(est_grad_field, pts)

    # oracle gradients (diagnostic only - not legal supervision)
    true_grads = bilinear_sample(image_gradients(img), pts)

    grid = grid_coords(s, device=dev)

    def eval_fn(model):
        pred = eval_chunked(model, grid).reshape(s, s).clamp(0, 1)
        return {
            "psnr": psnr(pred, img),
            "hf_psnr": hf_psnr(pred, img),
            "bands": radial_band_error(pred, img),
        }

    def pemlp():
        torch.manual_seed(args.seed)
        return PEMLP(in_features=2, hidden_features=256, hidden_layers=3, n_freqs=8).to(dev)

    configs = {
        "mse": (dict(), None),
        "ffl_interp": (dict(ffl=FocalFrequencyLoss(), ffl_w=args.ffl_w,
                            ffl_grid=(grid, interp_img)), None),
        "sobolev_est": (dict(sobolev=SobolevLoss(), sobolev_w=args.sobolev_w),
                        est_grads),
        "kacrice_est": (dict(kacrice=KacRiceLoss(n_levels=16),
                             kacrice_w=args.kacrice_w), est_grads),
    }
    if args.oracle:
        configs["sobolev_oracle"] = (
            dict(sobolev=SobolevLoss(), sobolev_w=args.sobolev_w), true_grads)
        configs["kacrice_oracle"] = (
            dict(kacrice=KacRiceLoss(n_levels=16), kacrice_w=args.kacrice_w),
            true_grads)

    # reference: PSNR of the plain interpolated image (what griddata alone gives)
    print(f"[baseline] griddata interpolation: psnr={psnr(interp_img, img):.3f} "
          f"hf_psnr={hf_psnr(interp_img, img):.3f}")

    histories, recons = {}, {}
    for name, (kw, grads) in configs.items():
        print(f"== {name}")
        model = pemlp()
        histories[name] = fit(
            model, pts, y, grads, iters=args.iters, lr=1e-3, eval_fn=eval_fn,
            eval_every=max(args.iters // 10, 50), **kw,
        )
        recons[name] = eval_chunked(model, grid).reshape(s, s).clamp(0, 1)

    tag = f"{args.mode}_n{args.n_samples}"
    save_history(histories, RESULTS / f"exp2_{tag}_history.json")
    plot_psnr_curves(histories, RESULTS / f"exp2_{tag}_psnr.png",
                     title=f"Non-uniform ({args.mode}, {args.n_samples} pts)")
    plot_psnr_curves(histories, RESULTS / f"exp2_{tag}_hfpsnr.png", key="hf_psnr",
                     title=f"High-frequency PSNR, non-uniform ({args.mode})")
    plot_band_errors({k: h[-1]["bands"] for k, h in histories.items()},
                     RESULTS / f"exp2_{tag}_bands.png")
    recons["griddata_interp"] = interp_img.cpu()
    plot_recons(recons, img.cpu(), RESULTS / f"exp2_{tag}_recons.png",
                title=f"Exp2 non-uniform ({args.mode}) reconstructions")
    print()
    print(summary_table(histories))


if __name__ == "__main__":
    main()
