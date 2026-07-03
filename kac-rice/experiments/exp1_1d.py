"""Experiment 1a: 1D multisine fitting on a regular grid (spectral-bias sanity check).

Backbone: PEMLP (ReLU + positional encoding), the classic spectrally-biased INR.
Compare: MSE only / +FFL / +Sobolev / +Kac-Rice, plus SIREN+MSE as the
architecture baseline. Expectation per the spec: Kac-Rice roughly TIES the
frequency/gradient losses here — this is the sanity check, not the win.

Usage: python experiments/exp1_1d.py [--iters 3000] [--device cuda]
"""

import argparse

import torch

from common import RESULTS, plot_psnr_curves, save_history, summary_table
from kacrice.crossing import KacRiceLoss
from kacrice.data import multisine
from kacrice.losses import FocalFrequencyLoss, SobolevLoss
from kacrice.metrics import psnr, spectrum_1d_error
from kacrice.models import PEMLP, SIREN
from kacrice.train import eval_chunked, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--kacrice-w", type=float, default=0.05)
    ap.add_argument("--sobolev-w", type=float, default=0.05)
    ap.add_argument("--ffl-w", type=float, default=1.0)
    ap.add_argument("--n-freqs", type=int, default=6,
                    help="PE octaves; 6 leaves the top signal freq (47) beyond "
                         "comfortable reach so spectral bias is actually visible")
    args = ap.parse_args()
    dev = args.device

    f, df = multisine(freqs=(2, 5, 11, 23, 47), seed=0)
    x = torch.linspace(-1, 1, args.n_train, device=dev).unsqueeze(-1)
    y = f(x)
    dy = df(x).unsqueeze(-1)

    x_eval = torch.linspace(-1, 1, 4096, device=dev).unsqueeze(-1)
    y_eval = f(x_eval)

    def eval_fn(model):
        pred = eval_chunked(model, x_eval)
        bands = spectrum_1d_error(pred, y_eval)
        return {
            "psnr": psnr(pred, y_eval, data_range=(y_eval.max() - y_eval.min()).item()),
            "hf_err": float(bands[len(bands) // 2 :].mean()),
        }

    def pemlp():
        torch.manual_seed(0)
        return PEMLP(in_features=1, hidden_features=256, hidden_layers=3,
                     n_freqs=args.n_freqs).to(dev)

    configs = {
        "pemlp_mse": dict(),
        "pemlp_ffl": dict(ffl=FocalFrequencyLoss(), ffl_w=args.ffl_w,
                          ffl_grid=(x, y)),
        "pemlp_sobolev": dict(sobolev=SobolevLoss(), sobolev_w=args.sobolev_w),
        "pemlp_kacrice": dict(kacrice=KacRiceLoss(n_levels=16), kacrice_w=args.kacrice_w),
    }

    histories = {}
    for name, kw in configs.items():
        print(f"== {name}")
        histories[name] = fit(
            pemlp(), x, y, dy, iters=args.iters, lr=1e-3, eval_fn=eval_fn,
            eval_every=max(args.iters // 20, 50), **kw,
        )

    print("== siren_mse (architecture baseline)")
    torch.manual_seed(0)
    siren = SIREN(in_features=1, hidden_features=256, hidden_layers=3).to(dev)
    histories["siren_mse"] = fit(
        siren, x, y, iters=args.iters, lr=1e-4, eval_fn=eval_fn,
        eval_every=max(args.iters // 20, 50),
    )

    save_history(histories, RESULTS / "exp1_1d_history.json")
    plot_psnr_curves(histories, RESULTS / "exp1_1d_psnr.png",
                     title="1D multisine, regular grid")
    plot_psnr_curves(histories, RESULTS / "exp1_1d_hferr.png", key="hf_err",
                     title="1D high-frequency band error (lower=better)")
    print()
    print(summary_table(histories, keys=("psnr", "hf_err")))


if __name__ == "__main__":
    main()
