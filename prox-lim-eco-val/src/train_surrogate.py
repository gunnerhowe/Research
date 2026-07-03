"""Unified surrogate trainer.

Phase 1 (--phase base): train the base generative surrogate (or the
deterministic baseline with --condition det) on one-step transitions.

Phase 2 (--phase ft): fine-tune a phase-1 checkpoint under a condition:
  base    : FM loss only (compute-matched control)
  tpp     : FM + TPP likelihood-ratio aux (GT-TPP vs self-TPP)   [intervention]
  pois    : same as tpp but GT-TPP replaced by Poissonized-TPP   [specificity]
  shuf    : same as tpp but GT-TPP replaced by shuffled-TPP      [optional]
  tpp_mle : plain -log p_gt_tpp aux (no self-TPP)                [ablation]
  marg    : FM + debiased Sinkhorn on Jiang-style summary stats  [dissociation]
  push    : FM + pushforward stability term (Brandstetter)       [stability]
  det     : MSE only (deterministic baseline)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from events import hard_events, soft_events, streams
from losses import STATS_FNS, sinkhorn_stats_loss, tpp_ratio_loss
from surrogate import (DetSurrogate, EMA, FlowSurrogate, GraphedRollout,
                       rollout_windowed)
from tpp import RecurrentTPP

ROOT = Path(__file__).resolve().parent.parent
TPP_CONDS = {"tpp", "pois", "shuf", "tpp_mle", "margtpp"}


def load_split(name, ddir):
    d = np.load(ddir / f"l96_{name}.npz")
    return torch.from_numpy(d["X"]), float(d["dt_obs"])


def load_event_cfg(ddir):
    with open(ddir / "events.json") as f:
        return json.load(f)


def build_model(condition, device, arch="cnn"):
    cls = DetSurrogate if condition == "det" else FlowSurrogate
    return cls(width=96, depth=4, arch=arch).to(device)


def fit_self_tpp(self_tpp, self_opt, w_hard, dt, warmup, n_steps):
    for _ in range(n_steps):
        loss = -self_tpp.log_likelihood(w_hard, dt, warmup=warmup)
        self_opt.zero_grad(set_to_none=True)
        loss.backward()
        self_opt.step()
    return loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["base", "ft"], required=True)
    ap.add_argument("--condition", default="base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=str, default="",
                    help="phase-1 checkpoint for --phase ft")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--steps", type=int, default=-1)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_sampler_steps", type=int, default=6)
    # aux settings
    ap.add_argument("--aux_weight", type=float, default=10.0)
    ap.add_argument("--marg_weight", type=float, default=1.0)
    ap.add_argument("--t_roll", type=int, default=100)
    ap.add_argument("--t_hist", type=int, default=100)
    ap.add_argument("--b_roll", type=int, default=16)
    ap.add_argument("--detach_every", type=int, default=25)
    ap.add_argument("--aux_every", type=int, default=2)
    ap.add_argument("--no_graphs", action="store_true")
    ap.add_argument("--self_tpp_steps", type=int, default=5)
    ap.add_argument("--self_tpp_warm", type=int, default=200)
    ap.add_argument("--sink_pts", type=int, default=1024)
    ap.add_argument("--sink_eps", type=float, default=0.1)
    ap.add_argument("--n_traj", type=int, default=0,
                    help="restrict training to the first N trajectories (0=all)")
    ap.add_argument("--arch", choices=["cnn", "mlp"], default="cnn")
    ap.add_argument("--stats", choices=["l96", "ks"], default="l96",
                    help="summary-stat family for the marg control")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--ref_dir", type=str, default="",
                    help="dir for event defs, TPP ckpts, marg-stat source and "
                         "event histories (clean references); defaults to "
                         "data_dir. Use when training on noisy observations.")
    args = ap.parse_args()
    ddir = (ROOT / args.data_dir) if not Path(args.data_dir).is_absolute() \
        else Path(args.data_dir)
    rdir = ((ROOT / args.ref_dir) if not Path(args.ref_dir).is_absolute()
            else Path(args.ref_dir)) if args.ref_dir else ddir
    cond = args.condition
    steps = args.steps if args.steps > 0 else (20000 if args.phase == "base" else 1500)
    out = Path(args.out) if args.out else ROOT / f"runs/{args.phase}_{cond}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    torch.manual_seed(args.seed * 1000 + 7)
    g_cpu = torch.Generator().manual_seed(args.seed * 1000 + 11)

    X, dt = load_split("train", ddir)
    Xv, _ = load_split("val", ddir)
    X_ref, _ = load_split("train", rdir)   # clean reference trajectories
    if args.n_traj > 0:
        X = X[:args.n_traj]
        X_ref = X_ref[:args.n_traj]
    ecfg = load_event_cfg(rdir)
    u, tau, kind = ecfg["u"], ecfg["tau"], ecfg["kind"]

    # normalization from train split
    x_mean, x_std = X.mean().item(), X.std().item()
    Xn = (X - x_mean) / x_std
    sigma_d = (Xn[:, 1:] - Xn[:, :-1]).std().item()
    Xn = Xn.to(device)
    Xnv = ((Xv - x_mean) / x_std).to(device)
    B, T, K = Xn.shape

    model = build_model(cond, device, args.arch)
    model.set_norm(x_mean, x_std, sigma_d)
    if args.phase == "ft":
        ck = torch.load(args.init, map_location=device, weights_only=True)
        model.load_state_dict(ck["ema"])
    ema = EMA(model, decay=0.999 if args.phase == "base" else 0.995)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    is_tpp_cond = cond in TPP_CONDS and args.phase == "ft"
    is_marg = cond in ("marg", "margtpp") and args.phase == "ft"
    is_push = cond == "push" and args.phase == "ft"
    needs_rollout = is_tpp_cond or is_marg

    # --- aux machinery ---------------------------------------------------
    if is_tpp_cond:
        ref_name = {"tpp": "gt", "pois": "pois", "shuf": "shuf",
                    "tpp_mle": "gt", "margtpp": "gt"}[cond]
        gt_tpp = RecurrentTPP(hidden=64).to(device)
        gt_tpp.load_state_dict(
            torch.load(rdir / f"tpp_{ref_name}.pt",
                       map_location=device, weights_only=True)["state_dict"])
        # keep train() mode: cuDNN RNN backward (w.r.t. INPUTS, for the soft
        # events) is only supported in training mode; params frozen below.
        for p in gt_tpp.parameters():
            p.requires_grad_(False)
        use_ratio = cond != "tpp_mle"
        if use_ratio:
            self_tpp = RecurrentTPP(hidden=64).to(device)
            self_tpp.load_state_dict(
                torch.load(rdir / "tpp_pois.pt", map_location=device,
                           weights_only=True)["state_dict"])
            self_opt = torch.optim.Adam(self_tpp.parameters(), lr=3e-3)
        # catalog (clean-reference) hard events for history prefixes
        ev_gt = hard_events(X_ref, u, kind).to(device)   # (B, T-1, K) bool

    if is_marg:
        # reference (clean climatology) summary-stat pool, standardized
        stats_fn = STATS_FNS[args.stats]
        with torch.no_grad():
            pts = stats_fn(X_ref[:, :2000].to(device), dt)
            stat_mean = pts.mean(0)
            stat_std = pts.std(0) + 1e-8
            pool = ((pts - stat_mean) / stat_std)
            pool = pool[torch.randperm(pool.shape[0], generator=g_cpu)[:200000].to(device)]

    recorder = None
    if needs_rollout and not args.no_graphs:
        recorder = GraphedRollout(model, args.b_roll, K, args.t_roll,
                                  args.n_sampler_steps, device)

    def sample_rollout_batch():
        """Rollout from random train ICs; returns physical x_roll (Br, t_roll+1, K)
        and the GT event history (Br, t_hist, K)."""
        bs = torch.randint(0, B, (args.b_roll,), generator=g_cpu)
        t0 = torch.randint(args.t_hist + 1, T - 1, (args.b_roll,), generator=g_cpu)
        xn0 = Xn[bs, t0]
        xr = rollout_windowed(model, xn0, args.t_roll, win=args.detach_every,
                              n_sampler_steps=args.n_sampler_steps,
                              recorder=recorder)
        x_phys = model.denormalize(xr)
        if is_tpp_cond:
            hist = torch.stack([ev_gt[b, t - args.t_hist:t]
                                for b, t in zip(bs.tolist(), t0.tolist())])
            return x_phys, hist
        return x_phys, None

    def tpp_aux(x_phys, hist):
        x_safe = torch.nan_to_num(x_phys, nan=0.0, posinf=1e4, neginf=-1e4)
        # censor diverged rollouts (keep gradient only for sane ones)
        with torch.no_grad():
            ok = (x_safe.abs().max(dim=1).values.max(dim=1).values
                  < x_std * 8 + abs(x_mean))          # (Br,)
        w_soft = soft_events(x_safe, u, tau, kind)     # (Br, t_roll, K)
        w_full = torch.cat([hist.float(), w_soft], dim=1)
        w_full = w_full[ok]
        if w_full.shape[0] == 0:
            return None, 0.0, (0.0, 0.0, 0.0)
        st = streams(w_full)                           # (Br*K, t_hist+t_roll)
        loss, llg, lls = tpp_ratio_loss(
            gt_tpp, self_tpp if use_ratio else gt_tpp, st, dt,
            warmup=args.t_hist, ratio=use_ratio)
        with torch.no_grad():
            evr = hard_events(x_safe, u, kind).float().mean().item() / dt
        return loss, (~ok).float().mean().item(), (llg, lls, evr)

    def update_self_tpp(x_phys, hist):
        with torch.no_grad():
            x_safe = torch.nan_to_num(x_phys.detach(), nan=0.0, posinf=1e4,
                                      neginf=-1e4)
            wh = hard_events(x_safe, u, kind).float()
            w_full = torch.cat([hist.float(), wh], dim=1)
            st = streams(w_full)
        return fit_self_tpp(self_tpp, self_opt, st, dt, args.t_hist,
                            args.self_tpp_steps)

    # warm-start self-TPP on init-model rollouts
    if is_tpp_cond and use_ratio:
        print("warm-starting self-TPP ...", flush=True)
        for _ in range(args.self_tpp_warm // args.self_tpp_steps):
            with torch.no_grad():
                x_phys, hist = sample_rollout_batch()
            update_self_tpp(x_phys, hist)

    # --- training loop ----------------------------------------------------
    log = []
    t_start = time.time()
    for it in range(steps):
        bs = torch.randint(0, B, (args.batch,), generator=g_cpu)
        ts = torch.randint(0, T - 2, (args.batch,), generator=g_cpu)
        xn_t, xn_tp1 = Xn[bs, ts], Xn[bs, ts + 1]

        if cond == "det":
            base_loss = model.mse_loss(xn_t, xn_tp1)
        else:
            base_loss = model.fm_loss(xn_t, xn_tp1)

        aux_loss = torch.tensor(0.0, device=device)
        div_frac = 0.0
        llg = lls = evr = 0.0
        pending_self = None
        if is_push:
            with torch.no_grad():
                z = torch.randn_like(xn_t)
                xn_tilde = model.sample_step(xn_t, z, args.n_sampler_steps)
            xn_tp2 = Xn[bs, ts + 2]
            aux_loss = model.fm_loss(xn_tilde, xn_tp2)
        elif needs_rollout and it % args.aux_every == 0:
            x_phys, hist = sample_rollout_batch()
            if is_tpp_cond:
                al, div_frac, (llg, lls, evr) = tpp_aux(x_phys, hist)
                if al is not None:
                    aux_loss = aux_loss + args.aux_weight * al
                if use_ratio:
                    # defer: stepping self_tpp now would mutate params that
                    # are part of the pending autograd graph
                    pending_self = (x_phys, hist)
            if is_marg:
                x_safe = torch.nan_to_num(x_phys, nan=0.0, posinf=1e4,
                                          neginf=-1e4)
                aux_loss = aux_loss + args.marg_weight * sinkhorn_stats_loss(
                    x_safe, pool[torch.randint(0, pool.shape[0],
                                               (args.sink_pts,), device=device)],
                    stat_mean, stat_std, dt, n_pts=args.sink_pts,
                    eps=args.sink_eps, stats_fn=stats_fn)

        loss = base_loss + aux_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        if pending_self is not None:
            update_self_tpp(*pending_self)

        if it % 200 == 0 or it == steps - 1:
            with torch.no_grad():
                vb = torch.randint(0, Xnv.shape[0], (256,), generator=g_cpu)
                vt = torch.randint(0, Xnv.shape[1] - 1, (256,), generator=g_cpu)
                if cond == "det":
                    vl = model.mse_loss(Xnv[vb, vt], Xnv[vb, vt + 1]).item()
                else:
                    vl = model.fm_loss(Xnv[vb, vt], Xnv[vb, vt + 1]).item()
            rec = dict(it=it, base=base_loss.item(),
                       aux=float(aux_loss), llg=round(llg, 5),
                       lls=round(lls, 5), evr=round(evr, 4), val=vl,
                       gnorm=float(gn), div=div_frac,
                       sec=round(time.time() - t_start, 1))
            log.append(rec)
            print(json.dumps(rec), flush=True)

    torch.save({"model": model.state_dict(), "ema": ema.shadow,
                "args": vars(args), "log": log},
               out / "ckpt.pt")
    print(f"saved {out / 'ckpt.pt'}  ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
