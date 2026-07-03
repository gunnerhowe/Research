"""Full paper experiment suite. One process, sequential runs, incremental JSON.

Suites
------
exp1_1d   : 1D multisine, regular grid. Loss configs on PEMLP + SIREN/FINER arch
            baselines. 3 seeds (model init).
exp2_1d   : 1D multisine, scattered non-uniform samples (ramp density). FFL must
            interpolate to a grid; Sobolev/Kac-Rice estimate derivatives from the
            same interpolant.
exp1_2d   : 2D image fitting, regular grid (camera + synthetic). Sanity check:
            expected ~tie.
exp2_2d   : THE DECISIVE ONE. 2D image from scattered samples; modes blobs/ramp/
            uniform. Includes oracle-gradient diagnostics and SIREN composability
            (blobs mode only).
ablations : eps_scale, n_levels, beta, sample budget (camera, blobs).

Every run appends a record to results/paper/<suite>.json (flushed after each run)
and final reconstructions for seed 0 are saved to results/paper/recons/.

Usage: python experiments/run_paper.py [--suites ...] [--seeds 3] [--quick]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import ROOT
from kacrice.crossing import KacRiceLoss
from kacrice.data import (bilinear_sample, grid_coords, image_gradients,
                          load_image, multisine, sample_coords, scattered_to_grid,
                          synthetic_texture)
from kacrice.losses import FocalFrequencyLoss, SobolevLoss
from kacrice.metrics import hf_psnr, psnr, radial_band_error, spectrum_1d_error, ssim
from kacrice.models import FINER, PEMLP, SIREN
from kacrice.train import eval_chunked, fit

OUT = ROOT / "results" / "paper"
RECONS = OUT / "recons"
OUT.mkdir(parents=True, exist_ok=True)
RECONS.mkdir(exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"  # overridden by --device
TAG = ""  # per-process filename suffix so parallel workers never share a file


# ------------------------------------------------------------------ helpers

def save_record(suite, rec):
    path = OUT / f"{suite}{TAG}.json"
    data = json.loads(path.read_text()) if path.exists() else []
    data.append(rec)
    path.write_text(json.dumps(data))


def already_done(suite, **fields):
    """Resume support: skip runs recorded by ANY worker (any tag) for this suite."""
    for path in OUT.glob(f"{suite}*.json"):
        for rec in json.loads(path.read_text()):
            if all(rec.get(k) == v for k, v in fields.items()):
                return True
    return False


def serializable(hist):
    out = []
    for r in hist:
        out.append({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in r.items()})
    return out


def model_factory(kind, in_features, seed):
    torch.manual_seed(seed)
    if kind == "pemlp":
        nf = 6 if in_features == 1 else 8
        return PEMLP(in_features, 1, 256, 3, n_freqs=nf).to(DEV)
    if kind == "siren":
        return SIREN(in_features, 1, 256, 3).to(DEV)
    if kind == "finer":
        return FINER(in_features, 1, 256, 3).to(DEV)
    raise ValueError(kind)


def aux_kwargs(config, *, eps_scale=0.15, n_levels=16, kr_w=0.05, sob_w=0.05,
               ffl_w=1.0, ffl_grid=None):
    """Map a config name suffix to fit() kwargs."""
    if config.endswith("mse"):
        return {}
    if "ffl" in config:
        return dict(ffl=FocalFrequencyLoss(), ffl_w=ffl_w, ffl_grid=ffl_grid)
    if "sobolev" in config:
        return dict(sobolev=SobolevLoss(), sobolev_w=sob_w)
    if "kacrice" in config:
        return dict(kacrice=KacRiceLoss(n_levels=n_levels, eps_scale=eps_scale),
                    kacrice_w=kr_w)
    raise ValueError(config)


def needs_grads(config):
    return ("sobolev" in config) or ("kacrice" in config)


# ------------------------------------------------------------------ 1D suites

def signal_setup():
    f, df = multisine(freqs=(2, 5, 11, 23, 47), seed=0)
    x_eval = torch.linspace(-1, 1, 4096, device=DEV).unsqueeze(-1)
    y_eval = f(x_eval)
    return f, df, x_eval, y_eval


def eval_1d(x_eval, y_eval):
    rng = (y_eval.max() - y_eval.min()).item()

    def fn(model):
        pred = eval_chunked(model, x_eval)
        bands = spectrum_1d_error(pred, y_eval)
        return {"psnr": psnr(pred, y_eval, data_range=rng),
                "hf_err": float(bands[4:].mean()), "bands": bands}

    return fn


def run_exp1_1d(seeds, iters):
    f, df, x_eval, y_eval = signal_setup()
    x = torch.linspace(-1, 1, 1024, device=DEV).unsqueeze(-1)
    y, dy = f(x), df(x).unsqueeze(-1)
    ev = eval_1d(x_eval, y_eval)

    configs = [("pemlp", "mse"), ("pemlp", "ffl"), ("pemlp", "sobolev"),
               ("pemlp", "kacrice"), ("siren", "mse"), ("finer", "mse")]
    for seed in seeds:
        for kind, cfg in configs:
            name = f"{kind}_{cfg}"
            t = time.time()
            model = model_factory(kind, 1, seed)
            kw = aux_kwargs(cfg, ffl_grid=(x, y))
            hist = fit(model, x, y, dy if needs_grads(cfg) else None,
                       iters=iters, lr=1e-3 if kind == "pemlp" else 5e-4,
                       eval_fn=ev, eval_every=iters // 20, verbose=False, **kw)
            print(f"exp1_1d {name} seed{seed}: psnr={hist[-1]['psnr']:.2f} "
                  f"({time.time()-t:.0f}s)", flush=True)
            save_record("exp1_1d", dict(
                suite="exp1_1d", config=name, seed=seed,
                history=serializable(hist)))
            if seed == seeds[0]:
                pred = eval_chunked(model, x_eval).cpu().numpy()
                np.save(RECONS / f"exp1_1d_{name}.npy", pred)
    np.save(RECONS / "exp1_1d_gt.npy", y_eval.cpu().numpy())


def run_exp2_1d(seeds, iters, n_samples=384):
    f, df, x_eval, y_eval = signal_setup()
    ev = eval_1d(x_eval, y_eval)
    grid_n = 1024
    xg = torch.linspace(-1, 1, grid_n, device=DEV)

    configs = ["mse", "ffl_interp", "sobolev_est", "kacrice_est"]
    for seed in seeds:
        # non-uniform 1D samples: ramp density (dense right, sparse left)
        g = torch.Generator().manual_seed(seed)
        xs = []
        need = n_samples
        while need > 0:
            cand = torch.rand(4 * need, generator=g) * 2 - 1
            acc = torch.rand(4 * need, generator=g) < (0.15 + 0.85 * (cand + 1) / 2)
            xs.append(cand[acc][:need])
            need = n_samples - sum(t.shape[0] for t in xs)
        x = torch.cat(xs).sort().values.to(DEV).unsqueeze(-1)
        y = f(x)

        # grid interpolant from scattered data only (what FFL needs; also the
        # source of estimated derivatives for Sobolev/Kac-Rice)
        y_interp = torch.from_numpy(
            np.interp(xg.cpu().numpy(), x[:, 0].cpu().numpy(), y.cpu().numpy())
        ).float().to(DEV)
        d_interp = torch.gradient(y_interp, spacing=(2.0 / (grid_n - 1),))[0]
        dy_est = torch.from_numpy(
            np.interp(x[:, 0].cpu().numpy(), xg.cpu().numpy(),
                      d_interp.cpu().numpy())
        ).float().to(DEV).unsqueeze(-1)

        interp_eval = torch.from_numpy(
            np.interp(x_eval[:, 0].cpu().numpy(), x[:, 0].cpu().numpy(),
                      y.cpu().numpy())
        ).float().to(DEV)
        interp_psnr = psnr(interp_eval, y_eval,
                           data_range=(y_eval.max() - y_eval.min()).item())

        for cfg in configs:
            t = time.time()
            model = model_factory("pemlp", 1, seed)
            kw = aux_kwargs(cfg, ffl_grid=(xg.unsqueeze(-1), y_interp))
            hist = fit(model, x, y, dy_est if needs_grads(cfg) else None,
                       iters=iters, lr=1e-3, eval_fn=ev,
                       eval_every=iters // 20, verbose=False, **kw)
            print(f"exp2_1d {cfg} seed{seed}: psnr={hist[-1]['psnr']:.2f} "
                  f"({time.time()-t:.0f}s)", flush=True)
            save_record("exp2_1d", dict(
                suite="exp2_1d", config=cfg, seed=seed, n_samples=n_samples,
                interp_psnr=interp_psnr, history=serializable(hist)))
            if seed == seeds[0]:
                pred = eval_chunked(model, x_eval).cpu().numpy()
                np.save(RECONS / f"exp2_1d_{cfg}.npy", pred)
        if seed == seeds[0]:
            np.save(RECONS / "exp2_1d_samples.npy",
                    torch.cat([x, y.unsqueeze(-1)], -1).cpu().numpy())


# ------------------------------------------------------------------ 2D suites

def eval_2d(grid, img, s):
    def fn(model):
        pred = eval_chunked(model, grid).reshape(s, s).clamp(0, 1)
        return {"psnr": psnr(pred, img), "hf_psnr": hf_psnr(pred, img),
                "ssim": ssim(pred, img), "bands": radial_band_error(pred, img)}

    return fn


def get_image(name, s):
    if name == "synthetic":
        return synthetic_texture(s).to(DEV)
    return load_image(name, s).to(DEV)


def run_exp1_2d(seeds, iters, images=("camera",), s=128):
    grid = grid_coords(s, device=DEV)
    configs = [("pemlp", "mse"), ("pemlp", "ffl"), ("pemlp", "sobolev"),
               ("pemlp", "kacrice"), ("siren", "mse"), ("finer", "mse")]
    for image in images:
        img = get_image(image, s)
        y = img.reshape(-1)
        grads = image_gradients(img).reshape(-1, 2)
        ev = eval_2d(grid, img, s)
        for seed in seeds:
            for kind, cfg in configs:
                name = f"{kind}_{cfg}"
                if already_done("exp1_2d", image=image, config=name, seed=seed):
                    print(f"exp1_2d {image} {name} seed{seed}: skip (done)",
                          flush=True)
                    continue
                t = time.time()
                model = model_factory(kind, 2, seed)
                kw = aux_kwargs(cfg, ffl_grid=(grid.reshape(s, s, 2), img))
                hist = fit(model, grid, y, grads if needs_grads(cfg) else None,
                           iters=iters, lr=1e-3 if kind == "pemlp" else 5e-4,
                           eval_fn=ev, eval_every=iters // 10, verbose=False, **kw)
                print(f"exp1_2d {image} {name} seed{seed}: "
                      f"psnr={hist[-1]['psnr']:.2f} ({time.time()-t:.0f}s)",
                      flush=True)
                save_record("exp1_2d", dict(
                    suite="exp1_2d", image=image, config=name, seed=seed,
                    history=serializable(hist)))
                if seed == seeds[0] and image == "camera":
                    pred = eval_chunked(model, grid).reshape(s, s).clamp(0, 1)
                    np.save(RECONS / f"exp1_2d_{name}.npy", pred.cpu().numpy())
        if image == "camera":
            np.save(RECONS / "exp1_2d_gt.npy", img.cpu().numpy())


def exp2_setup(img, s, mode, seed, n_samples):
    pts = sample_coords(n_samples, mode=mode, seed=seed, device=DEV)
    y = bilinear_sample(img, pts)
    interp_img = scattered_to_grid(pts, y, s).to(DEV)
    est_grads = bilinear_sample(image_gradients(interp_img), pts)
    true_grads = bilinear_sample(image_gradients(img), pts)
    return pts, y, interp_img, est_grads, true_grads


def run_exp2_2d(seeds, iters, s=256, n_samples=8192, ffl_crop=128, parts=None):
    """parts: subset of {blobs, ramp, uniform, synthetic} for parallel workers."""
    grid = grid_coords(s, device=DEV)
    core = ["mse", "ffl_interp", "sobolev_est", "kacrice_est"]
    plans = []
    for mode in ("blobs", "ramp", "uniform"):
        cfgs = list(core)
        if mode == "blobs":
            cfgs += ["sobolev_oracle", "kacrice_oracle", "siren_mse",
                     "siren_kacrice_est", "siren_ffl_interp",
                     "siren_sobolev_est"]
        if parts is None or mode in parts:
            plans.append(("camera", mode, cfgs))
    if parts is None or "synthetic" in parts:
        plans.append(("synthetic", "blobs", list(core)))

    for image, mode, cfgs in plans:
        img = get_image(image, s)
        ev = eval_2d(grid, img, s)
        for seed in seeds:
            pts, y, interp_img, est_grads, true_grads = exp2_setup(
                img, s, mode, seed, n_samples)
            interp_metrics = dict(
                psnr=psnr(interp_img, img), hf_psnr=hf_psnr(interp_img, img),
                ssim=ssim(interp_img, img),
                bands=radial_band_error(interp_img, img).tolist())
            for cfg in cfgs:
                if already_done("exp2_2d", image=image, mode=mode, config=cfg,
                                seed=seed):
                    print(f"exp2_2d {image} {mode} {cfg} seed{seed}: skip (done)",
                          flush=True)
                    continue
                t = time.time()
                kind = "siren" if cfg.startswith("siren") else "pemlp"
                base_cfg = cfg.replace("siren_", "")
                model = model_factory(kind, 2, seed)
                kw = aux_kwargs(base_cfg,
                                ffl_grid=(grid.reshape(s, s, 2), interp_img))
                grads = None
                if needs_grads(base_cfg):
                    grads = true_grads if cfg.endswith("oracle") else est_grads
                hist = fit(model, pts, y, grads, iters=iters,
                           lr=1e-3 if kind == "pemlp" else 5e-4,
                           ffl_crop=ffl_crop,
                           eval_fn=ev, eval_every=iters // 10, verbose=False, **kw)
                print(f"exp2_2d {image} {mode} {cfg} seed{seed}: "
                      f"psnr={hist[-1]['psnr']:.2f} "
                      f"hf={hist[-1]['hf_psnr']:.2f} ({time.time()-t:.0f}s)",
                      flush=True)
                save_record("exp2_2d", dict(
                    suite="exp2_2d", image=image, mode=mode, config=cfg,
                    seed=seed, n_samples=n_samples, interp=interp_metrics,
                    history=serializable(hist)))
                if seed == seeds[0] and image == "camera" and mode == "blobs":
                    pred = eval_chunked(model, grid).reshape(s, s).clamp(0, 1)
                    np.save(RECONS / f"exp2_2d_{cfg}.npy", pred.cpu().numpy())
            if seed == seeds[0] and image == "camera" and mode == "blobs":
                np.save(RECONS / "exp2_2d_gt.npy", img.cpu().numpy())
                np.save(RECONS / "exp2_2d_interp.npy", interp_img.cpu().numpy())
                np.save(RECONS / "exp2_2d_pts.npy", pts.cpu().numpy())


def run_ablations(seeds, iters, s=256, n_samples=8192, ffl_crop=128):
    """All on camera/blobs with the kacrice_est pathway (plus mse/ffl refs for
    the sample-budget sweep). Run at reduced iters/seeds: these are sensitivity
    curves, not headline numbers; the default point is inherited from exp2_2d."""
    grid = grid_coords(s, device=DEV)
    img = get_image("camera", s)
    ev = eval_2d(grid, img, s)

    def one(tag, seed, cfg="kacrice_est", n=n_samples, **hp):
        pts, y, interp_img, est_grads, _ = exp2_setup(img, s, "blobs", seed, n)
        kind = "pemlp"
        model = model_factory(kind, 2, seed)
        kw = aux_kwargs(cfg, ffl_grid=(grid.reshape(s, s, 2), interp_img), **hp)
        grads = est_grads if needs_grads(cfg) else None
        t = time.time()
        hist = fit(model, pts, y, grads, iters=iters, lr=1e-3, eval_fn=ev,
                   ffl_crop=ffl_crop, eval_every=iters // 5, verbose=False, **kw)
        print(f"ablate {tag} seed{seed}: psnr={hist[-1]['psnr']:.2f} "
              f"hf={hist[-1]['hf_psnr']:.2f} ({time.time()-t:.0f}s)", flush=True)
        save_record("ablations", dict(
            suite="ablations", tag=tag, config=cfg, seed=seed, n_samples=n,
            hp=hp, history=serializable(hist)))

    for seed in seeds:
        one("eps_0.15", seed)                      # default reference point
        for es in (0.05, 0.1, 0.3, 0.6):           # eps_scale sweep
            one(f"eps_{es}", seed, eps_scale=es)
        for nl in (4, 8, 32):                      # n_levels (16 = default)
            one(f"levels_{nl}", seed, n_levels=nl)
        for w in (0.01, 0.2, 1.0):                 # beta (0.05 = default)
            one(f"beta_{w}", seed, kr_w=w)
        for n in (4096, 16384):                    # sample budget (8192 = default)
            for cfg in ("mse", "ffl_interp", "kacrice_est"):
                one(f"n{n}_{cfg}", seed, cfg=cfg, n=n)
        for cfg in ("mse", "ffl_interp"):          # refs at default N, same iters
            one(f"n{n_samples}_{cfg}", seed, cfg=cfg)
        one(f"n{n_samples}_kacrice_est", seed)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=[
        "exp1_1d", "exp2_1d", "exp1_2d", "exp2_2d", "ablations"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--iters-1d", type=int, default=3000)
    ap.add_argument("--iters-2d", type=int, default=2000)
    ap.add_argument("--ablation-iters", type=int, default=1000)
    ap.add_argument("--ablation-seeds", type=int, default=2)
    ap.add_argument("--device", default="cpu",
                    help="cpu (default; gpu is contended) or cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="", help="suffix for result files "
                    "(parallel workers must use distinct tags)")
    ap.add_argument("--exp2-modes", default=None,
                    help="comma list: blobs,ramp,uniform,synthetic")
    ap.add_argument("--seed-list", default=None,
                    help="comma list of specific seeds (overrides --seeds)")
    ap.add_argument("--quick", action="store_true", help="1 seed, short iters")
    args = ap.parse_args()

    global DEV, TAG
    DEV = args.device
    TAG = args.tag
    torch.set_num_threads(args.threads)

    if args.seed_list:
        seeds = [int(x) for x in args.seed_list.split(",")]
    else:
        seeds = list(range(1 if args.quick else args.seeds))
    it1 = 300 if args.quick else args.iters_1d
    it2 = 200 if args.quick else args.iters_2d
    ita = 100 if args.quick else args.ablation_iters

    print(f"device={DEV} threads={args.threads} seeds={seeds} "
          f"iters={it1}/{it2}/{ita}", flush=True)
    t0 = time.time()
    if "exp1_1d" in args.suites:
        run_exp1_1d(seeds, it1)
    if "exp2_1d" in args.suites:
        run_exp2_1d(seeds, it1)
    if "exp1_2d" in args.suites:
        run_exp1_2d(seeds, it2)
    if "exp2_2d" in args.suites:
        parts = args.exp2_modes.split(",") if args.exp2_modes else None
        run_exp2_2d(seeds, it2, parts=parts)
    if "ablations" in args.suites:
        run_ablations(seeds[: args.ablation_seeds], ita)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
