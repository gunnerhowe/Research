"""Experiment 1b: 2D image fitting on a regular grid (sanity check, not the win).

All losses get full-information supervision: the true grid image and its
central-difference gradients. Expected outcome per the spec: Kac-Rice ~ties
FFL/Sobolev on-grid.

Usage: python experiments/exp1_2d.py [--size 256] [--iters 2000]
"""

import argparse

import torch

from common import (RESULTS, plot_band_errors, plot_psnr_curves, plot_recons,
                    save_history, summary_table)
from kacrice.crossing import KacRiceLoss
from kacrice.data import bilinear_sample, grid_coords, image_gradients, load_image
from kacrice.losses import FocalFrequencyLoss, SobolevLoss
from kacrice.metrics import hf_psnr, psnr, radial_band_error
from kacrice.models import PEMLP, SIREN
from kacrice.train import eval_chunked, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--image", default="camera")
    ap.add_argument("--kacrice-w", type=float, default=0.05)
    ap.add_argument("--sobolev-w", type=float, default=0.05)
    ap.add_argument("--ffl-w", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()
    dev = args.device
    s = args.size

    img = load_image(args.image, s).to(dev)
    coords = grid_coords(s, device=dev)
    y = img.reshape(-1)
    grads = image_gradients(img).reshape(-1, 2)

    def eval_fn(model):
        pred = eval_chunked(model, coords).reshape(s, s).clamp(0, 1)
        return {
            "psnr": psnr(pred, img),
            "hf_psnr": hf_psnr(pred, img),
            "bands": radial_band_error(pred, img),
        }

    def pemlp():
        torch.manual_seed(0)
        return PEMLP(in_features=2, hidden_features=256, hidden_layers=3, n_freqs=8).to(dev)

    configs = {
        "pemlp_mse": dict(),
        "pemlp_ffl": dict(ffl=FocalFrequencyLoss(), ffl_w=args.ffl_w,
                          ffl_grid=(coords, img)),
        "pemlp_sobolev": dict(sobolev=SobolevLoss(), sobolev_w=args.sobolev_w),
        "pemlp_kacrice": dict(kacrice=KacRiceLoss(n_levels=16),
                              kacrice_w=args.kacrice_w),
    }

    histories, recons = {}, {}
    for name, kw in configs.items():
        print(f"== {name}")
        model = pemlp()
        histories[name] = fit(
            model, coords, y, grads, iters=args.iters, lr=1e-3,
            batch=args.batch, eval_fn=eval_fn,
            eval_every=max(args.iters // 10, 50), **kw,
        )
        recons[name] = eval_chunked(model, coords).reshape(s, s).clamp(0, 1)

    save_history(histories, RESULTS / "exp1_2d_history.json")
    plot_psnr_curves(histories, RESULTS / "exp1_2d_psnr.png",
                     title=f"2D {args.image} {s}x{s}, regular grid")
    plot_psnr_curves(histories, RESULTS / "exp1_2d_hfpsnr.png", key="hf_psnr",
                     title="High-frequency PSNR (regular grid)")
    plot_band_errors({k: h[-1]["bands"] for k, h in histories.items()},
                     RESULTS / "exp1_2d_bands.png")
    plot_recons(recons, img.cpu(), RESULTS / "exp1_2d_recons.png",
                title="Exp1b reconstructions (top) and highpass residuals (bottom)")
    print()
    print(summary_table(histories))


if __name__ == "__main__":
    main()
