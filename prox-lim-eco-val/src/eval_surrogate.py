"""Evaluate a surrogate checkpoint against ground-truth references.

Produces runs/<name>/metrics.json with:
  - long-rollout event-timing metrics (IET KS/W1, CV, Fano curve error,
    return-period curve error, aggregate-event transfer stats, TPP-LL)
  - marginal/spectral metrics (state W1, PSD log-distance, ACF error)
  - short-horizon skill (CRPS, ensemble-mean RMSE, spread)
  - stability (diverged fraction, mean survival)
GT references are computed once and cached at runs/eval_refs.pt.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

from events import hard_events
from metrics import (RL_QUANTILES, acf_mean, censor, crps_ensemble, fano_curve,
                     event_ll_under, hazard_curve, pooled_iets, psd_mean,
                     return_periods, summarize_events)
from surrogate import DetSurrogate, FlowSurrogate, rollout
from tpp import RecurrentTPP

ROOT = Path(__file__).resolve().parent.parent
AGG_Q = 0.98  # aggregate (spatial-max) event threshold quantile


def load_event_cfg(ddir):
    with open(ddir / "events.json") as f:
        return json.load(f)


def build_refs(ddir, force=False):
    ref_path = ddir / "eval_refs.pt"
    if ref_path.exists() and not force:
        return torch.load(ref_path, weights_only=False)
    ecfg = load_event_cfg(ddir)
    u, kind = ecfg["u"], ecfg["kind"]
    d = np.load(ddir / "l96_eval_long.npz")
    X = torch.from_numpy(d["X"])
    dt = float(d["dt_obs"])
    s = {"neg": -X, "value": X, "energy": X * X}[kind]
    refs = {}
    ev = hard_events(X, u, kind)
    refs["iets"] = pooled_iets(ev, dt)
    refs["rate"] = float(ev.float().mean().item() / dt)
    refs["iet_cv"] = float(refs["iets"].std() / refs["iets"].mean())
    refs["fano"] = fano_curve(ev, dt)
    refs["hazard"] = hazard_curve(refs["iets"], dt)
    refs["rl_thresholds"] = [float(np.quantile(s.numpy(), q)) for q in RL_QUANTILES]
    refs["rl_rp"] = return_periods(X, refs["rl_thresholds"], kind, dt)
    # aggregate events: spatial max of x above its q0.98
    mx = X.max(dim=2, keepdim=True).values
    u_agg = float(np.quantile(mx.numpy(), AGG_Q))
    refs["u_agg"] = u_agg
    ev_a = (mx[:, 1:] > u_agg) & (mx[:, :-1] <= u_agg)
    iets_a = pooled_iets(ev_a, dt)
    refs["agg"] = dict(rate=float(ev_a.float().mean().item() / dt),
                       cv=float(iets_a.std() / iets_a.mean()),
                       fano=fano_curve(ev_a, dt))
    refs["marginal_sample"] = X.reshape(-1)[
        torch.randperm(X.numel(), generator=torch.Generator().manual_seed(0))[:200000]
    ].numpy()
    refs["psd"] = psd_mean(X)
    refs["acf"] = acf_mean(X)
    refs["dt"] = dt
    torch.save(refs, ref_path)
    return refs


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cond = ck["args"]["condition"]
    arch = ck["args"].get("arch", "cnn")
    cls = DetSurrogate if cond == "det" else FlowSurrogate
    model = cls(width=96, depth=4, arch=arch).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    return model, ck["args"]


@torch.no_grad()
def evaluate(ckpt_path: str, n_roll: int = 64, t_roll: int = 4000,
             n_sampler_steps: int = 6, ens: int = 16, seed: int = 999,
             data_dir: str = "data"):
    device = "cuda"
    ddir = (ROOT / data_dir) if not Path(data_dir).is_absolute() \
        else Path(data_dir)
    ecfg = load_event_cfg(ddir)
    u, tau, kind = ecfg["u"], ecfg["tau"], ecfg["kind"]
    refs = build_refs(ddir)
    dt = refs["dt"]
    model, targs = load_model(ckpt_path, device)

    dv = np.load(ddir / "l96_eval.npz")
    Xe = torch.from_numpy(dv["X"])
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    # ---- long rollouts -------------------------------------------------
    bs = torch.randint(0, Xe.shape[0], (n_roll,), generator=g)
    ts = torch.randint(0, Xe.shape[1] - 1, (n_roll,), generator=g)
    x0 = Xe[bs, ts].to(device)
    xn0 = model.normalize(x0)
    xn = rollout(model, xn0, t_roll, n_sampler_steps=n_sampler_steps,
                 use_checkpoint=False)
    x_phys = model.denormalize(
        torch.nan_to_num(xn, nan=0.0, posinf=0.0, neginf=0.0)).cpu()
    surv = censor(xn.cpu())

    out = dict(ckpt=str(ckpt_path), condition=targs["condition"],
               seed=targs["seed"], n_roll=n_roll, t_roll=t_roll)
    out["div_frac"] = float((surv < t_roll).float().mean())
    out["mean_survival_mtu"] = float(surv.float().mean() * dt)

    # event metrics (primary definition)
    esum, iets, ev = summarize_events(x_phys, u, kind, dt, surv)
    out["events"] = esum
    hz = hazard_curve(iets, dt) if len(iets) > 100 else np.array([])
    if len(iets) > 100:
        ks = sps.ks_2samp(iets, refs["iets"])
        out["iet_ks"] = float(ks.statistic)
        out["iet_w1"] = float(sps.wasserstein_distance(iets, refs["iets"]))
        gh = refs.get("hazard", np.array([]))
        if len(gh) and len(hz):
            ok_h = np.isfinite(hz) & np.isfinite(gh)
            out["hazard_rmse"] = float(np.sqrt(np.mean((hz[ok_h] - gh[ok_h]) ** 2)))
        gt_f, sf = refs["fano"], esum["fano"]
        common = [w for w in gt_f if np.isfinite(gt_f[w]) and np.isfinite(sf[w])]
        out["fano_logerr"] = float(np.mean(
            [abs(np.log(sf[w] / gt_f[w])) for w in common]))
        out["rate_ratio"] = esum.get("rate", np.nan) / refs["rate"]
    # return periods
    rp = return_periods(x_phys, refs["rl_thresholds"], kind, dt, surv)
    out["rl_rp"] = rp
    out["rl_logerr"] = float(np.nanmean(
        [abs(np.log(a / b)) for a, b in zip(rp, refs["rl_rp"])
         if np.isfinite(a) and a > 0]) if any(np.isfinite(rp)) else np.nan)
    # aggregate transfer events
    mx = x_phys.max(dim=2, keepdim=True).values
    ev_a = (mx[:, 1:] > refs["u_agg"]) & (mx[:, :-1] <= refs["u_agg"])
    iets_a = pooled_iets(ev_a, dt, surv)
    out["agg"] = dict(
        rate=float(ev_a.float().mean().item() / dt),
        cv=float(iets_a.std() / iets_a.mean()) if len(iets_a) > 10 else np.nan,
        fano=fano_curve(ev_a, dt, surv))
    # TPP-LL of rollout events under GT-TPP (self-history warmup 200 steps)
    gt_tpp = RecurrentTPP(hidden=64).to(device)
    gt_tpp.load_state_dict(torch.load(ddir / "tpp_gt.pt",
                                      map_location=device,
                                      weights_only=True)["state_dict"])
    out["tpp_ll"] = event_ll_under(gt_tpp, ev, dt, warmup=200, device=device)

    # marginal / spectral
    samp = x_phys.reshape(-1)
    samp = samp[torch.randperm(samp.numel(), generator=g)[:200000]].numpy()
    out["marg_w1"] = float(sps.wasserstein_distance(samp, refs["marginal_sample"]))
    psd = psd_mean(x_phys)
    if len(psd) and len(refs["psd"]):
        eps = 1e-12
        out["psd_logdist"] = float(np.sqrt(np.mean(
            (np.log(psd + eps) - np.log(refs["psd"] + eps)) ** 2)))
    acf = acf_mean(x_phys)
    out["acf_rmse"] = float(np.sqrt(np.mean((acf - refs["acf"]) ** 2)))

    # ---- short-horizon skill -------------------------------------------
    n_ic, leads = 256, [1, 5, 10, 20]
    bs = torch.randint(0, Xe.shape[0], (n_ic,), generator=g)
    ts = torch.randint(0, Xe.shape[1] - max(leads) - 1, (n_ic,), generator=g)
    x0 = Xe[bs, ts].to(device)
    m_ens = 1 if isinstance(model, DetSurrogate) else ens
    xn_cur = model.normalize(x0)[None].expand(m_ens, -1, -1).reshape(-1, x0.shape[1])
    crps, rmse, spread = {}, {}, {}
    step = 0
    for lead in leads:
        while step < lead:
            z = torch.randn(xn_cur.shape, device=device)
            xn_cur = model.sample_step(xn_cur, z, n_sampler_steps)
            step += 1
        pred = model.denormalize(xn_cur).reshape(m_ens, n_ic, -1).cpu()
        tgt = Xe[bs, ts + lead]
        crps[lead] = crps_ensemble(pred, tgt)
        rmse[lead] = float((pred.mean(0) - tgt).pow(2).mean().sqrt())
        spread[lead] = float(pred.std(0).mean()) if m_ens > 1 else 0.0
    out["crps"] = crps
    out["rmse"] = rmse
    out["spread"] = spread

    # raw curves for figures
    curves = dict(
        iets=iets[:200000] if len(iets) else np.array([]),
        fano_w=np.array(list(esum["fano"].keys()), dtype=float),
        fano=np.array(list(esum["fano"].values()), dtype=float),
        rl_thresholds=np.array(refs["rl_thresholds"]),
        rl_rp=np.array(rp), psd=psd, acf=acf, hazard=hz,
        agg_fano=np.array(list(out["agg"]["fano"].values()), dtype=float),
        marg_sample=samp[:100000])
    np.savez_compressed(Path(ckpt_path).parent / "curves.npz", **curves)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_roll", type=int, default=64)
    ap.add_argument("--t_roll", type=int, default=4000)
    ap.add_argument("--tag", default="metrics")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--n_sampler_steps", type=int, default=6)
    args = ap.parse_args()
    out = evaluate(args.ckpt, n_roll=args.n_roll, t_roll=args.t_roll,
                   data_dir=args.data_dir,
                   n_sampler_steps=args.n_sampler_steps)
    path = Path(args.ckpt).parent / f"{args.tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("condition", "seed", "div_frac", "iet_ks",
                               "iet_w1", "fano_logerr", "rate_ratio",
                               "rl_logerr", "tpp_ll", "marg_w1",
                               "psd_logdist", "acf_rmse")}, indent=2,
                     default=float))
    print(f"full metrics -> {path}")


if __name__ == "__main__":
    main()
