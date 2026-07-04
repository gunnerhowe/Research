"""Phase 2 experiment suite: crossing-density budget as a PINN stabilizer (CGLE).

Suites (spec numbering, kill conditions enforced):
  exp3  vanilla-failure reproduction (STOP if vanilla does not fail)
  exp4  budget rescue, three budget sources (ref sim / physics prior / scalar)
  exp5  null control: uniform gradient damping + curvature damping, weight-swept
  exp6  baselines: Fourier-feature PINN (faithful ASPEN-style layer),
        time-curriculum (seq2seq), causal weighting
  exp7  composition: Fourier-feature PINN + budget
  abl   budget weight sweep

Every run appends to results/phase2/<suite><tag>.json. Resume-safe.
Usage: python experiments/run_phase2.py --suites exp3 exp4 ... [--iters N]
       [--device cuda|cpu] [--tag _a] [--seed-list 0,1]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import ROOT
from kacrice.cgle import (budgets_from_physics, budgets_from_reference,
                          budgets_scalar, cgle_chaotic_reference,
                          cgle_reference, spatial_crossing_profile)
from kacrice.crossing import CrossingBudgetLoss
from kacrice.metrics import psnr  # noqa: F401  (parity with phase 1 imports)
from kacrice.pinn import (CGLEMlp, CGLEMlpPeriodic, eval_on_reference,
                          make_points, make_points_chaotic, pinn_losses)

OUT = ROOT / "results" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)
REF_CACHE = OUT / "reference.npz"

DEV = "cpu"
TAG = ""

LEVELS = np.linspace(-1.05, 1.05, 16)
EPS = 0.08  # ~0.15 * std of the front field (std ~ 0.55 within [-1.1, 1.1])
W_ICBC = 100.0


# ------------------------------------------------------------------ helpers

def save_record(suite, rec):
    path = OUT / f"{suite}{TAG}.json"
    data = json.loads(path.read_text()) if path.exists() else []
    data.append(rec)
    path.write_text(json.dumps(data))


def already_done(suite, **fields):
    for path in OUT.glob(f"{suite}*.json"):
        for rec in json.loads(path.read_text()):
            if all(rec.get(k) == v for k, v in fields.items()):
                return True
    return False


def get_reference():
    if REF_CACHE.exists():
        z = np.load(REF_CACHE)
        return z["x"], z["t"], z["A"]
    print("computing CGLE reference (nx=1024, dt=1e-4, T=10)...", flush=True)
    x, t, A = cgle_reference(nx=1024, dt=1e-4, t1=10.0, n_save=201)
    np.savez_compressed(REF_CACHE, x=x, t=t, A=A)
    return x, t, A


def band_error_final(A_pred, A_ref, n_bands=6):
    """Relative spatial-spectrum error of Re(A), averaged over time slices."""
    fp = np.abs(np.fft.rfft(A_pred.real, axis=1))
    fr = np.abs(np.fft.rfft(A_ref.real, axis=1))
    n = fp.shape[1]
    edges = np.linspace(0, n, n_bands + 1).astype(int)
    return [float(np.abs(fp[:, s] - fr[:, s]).mean() /
                  max(np.abs(fr[:, s]).mean(), 1e-12))
            for s in (slice(edges[i], max(edges[i + 1], edges[i] + 1))
                      for i in range(n_bands))]


# ------------------------------------------------------------------ training

def train_pinn(seed, iters, aux=None, aux_w=0.0, fourier_m=0, schedule=None,
               causal=False, n_res=20000, eval_every=None, x_ref=None,
               t_ref=None, A_ref=None, lr=1e-3, normalize=True,
               model_factory=None, points=None, coeffs=None, t1=10.0):
    """One PINN run. aux: None | ('budget', {re,im}-CrossingBudgetLoss) |
    'grad_damp' | 'curv_damp'. schedule: None | 'curriculum'.
    model_factory/points/coeffs/t1 override the default (front) testbed.
    Returns history list + final prediction array."""
    torch.manual_seed(seed)
    model = (model_factory() if model_factory
             else CGLEMlp(fourier_m=fourier_m, normalize=normalize)).to(DEV)
    pts = points if points is not None else make_points(
        n_res=n_res, seed=seed, device=DEV)
    b_c = coeffs or (None, None)
    res_pts = pts[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    eval_every = eval_every or max(iters // 20, 250)

    if causal:
        n_bins = 32
        bin_idx = (res_pts[:, 1] / t1 * n_bins).long().clamp(max=n_bins - 1)
        bin_count = torch.bincount(bin_idx, minlength=n_bins).clamp_min(1)
        eps_causal = 1.0

    history = []
    t0 = time.time()
    for it in range(1, iters + 1):
        if it == int(iters * 0.5):
            for g in opt.param_groups:
                g["lr"] = lr * 0.1

        if schedule == "curriculum":
            frac = min(1.0, 0.25 + 0.75 * (it / iters))  # grow time window
            mask = res_pts[:, 1] <= t1 * frac
            cur = (res_pts[mask],) + tuple(pts[1:])
        else:
            cur = pts

        if b_c[0] is not None:
            r2, loss_icbc, (u, v, u_x, v_x, u_xx, v_xx) = pinn_losses(
                model, cur, b=b_c[0], c=b_c[1])
        else:
            r2, loss_icbc, (u, v, u_x, v_x, u_xx, v_xx) = pinn_losses(model, cur)

        if causal:
            with torch.no_grad():
                bin_mean = torch.zeros(n_bins, device=DEV).index_add_(
                    0, bin_idx, r2.detach()) / bin_count
                w_bin = torch.exp(-eps_causal * torch.cumsum(
                    torch.cat([torch.zeros(1, device=DEV), bin_mean[:-1]]), 0))
            loss_res = (w_bin[bin_idx] * r2).mean()
        else:
            loss_res = r2.mean()

        loss = loss_res + W_ICBC * loss_icbc
        parts = {"res": loss_res.item(), "icbc": loss_icbc.item()}

        if aux is not None and aux_w > 0:
            if isinstance(aux, tuple) and aux[0] == "budget":
                bl = aux[1]
                la = bl["re"](u, u_x.abs()) + bl["im"](v, v_x.abs())
            elif aux == "grad_damp":
                la = (u_x**2 + v_x**2).mean()
            elif aux == "curv_damp":
                la = (u_xx**2 + v_xx**2).mean()
            else:
                raise ValueError(f"unknown aux {aux!r}")
            loss = loss + aux_w * la
            parts["aux"] = la.item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % eval_every == 0 or it == iters:
            rel, _ = eval_on_reference(model, x_ref, t_ref, A_ref, DEV)
            rec = {"iter": it, "time": time.time() - t0, "rel_l2": rel, **parts}
            history.append(rec)
            if not np.isfinite(rel) or rel > 50:
                break  # hard divergence; stop burning compute
    rel, A_pred = eval_on_reference(model, x_ref, t_ref, A_ref, DEV)
    return history, A_pred, model


# ------------------------------------------------------------------ suites

def budget_pair(source, x, A_ref):
    if source == "ref":
        b = budgets_from_reference(A_ref, x, LEVELS, EPS)
    elif source == "phys":
        b = budgets_from_physics(LEVELS)
    elif source == "scalar":
        ref = budgets_from_reference(A_ref, x, LEVELS, EPS)
        cap = float(max(ref["re"].max(), ref["im"].max()))
        b = budgets_scalar(LEVELS, cap)
    else:
        raise ValueError(source)
    return {
        "re": CrossingBudgetLoss(LEVELS, b["re"], EPS).to(DEV),
        "im": CrossingBudgetLoss(LEVELS, b["im"], EPS).to(DEV),
    }


def run_one(suite, config, seed, iters, ref, **train_kw):
    if already_done(suite, config=config, seed=seed, iters=iters):
        print(f"{suite} {config} seed{seed}: skip (done)", flush=True)
        return
    x_ref, t_ref, A_ref = ref
    t0 = time.time()
    hist, A_pred, _ = train_pinn(seed, iters, x_ref=x_ref, t_ref=t_ref,
                                 A_ref=A_ref, **train_kw)
    rel = hist[-1]["rel_l2"]
    diverged = bool(not np.isfinite(rel) or rel > 1.0)
    bands = band_error_final(A_pred, A_ref)
    cross_re = spatial_crossing_profile(A_pred.real, x_ref, LEVELS, EPS)
    rec = dict(suite=suite, config=config, seed=seed, iters=iters,
               rel_l2=float(rel), diverged=diverged, bands=bands,
               crossings_re_mean=cross_re.mean(0).tolist(),
               history=hist, wall_s=time.time() - t0)
    save_record(suite, rec)
    print(f"{suite} {config} seed{seed}: rel_l2={rel:.4f} "
          f"diverged={diverged} ({time.time()-t0:.0f}s)", flush=True)
    if seed == 0:
        np.save(OUT / f"pred_{suite}_{config}.npy", A_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+",
                    default=["exp3", "exp4", "exp5", "exp6", "exp7", "abl"])
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-list", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    global DEV, TAG
    DEV = args.device
    TAG = args.tag
    torch.set_num_threads(args.threads)
    seeds = ([int(s) for s in args.seed_list.split(",")] if args.seed_list
             else list(range(args.seeds)))

    ref = get_reference()
    x_ref, _, A_ref = ref
    it = args.iters
    print(f"device={DEV} iters={it} seeds={seeds}", flush=True)

    if "diag" in args.suites:
        # raw-coordinate vanilla: does ASPEN's oscillatory failure mode appear
        # when the tanh MLP sees unnormalized (x,t)?
        for s in seeds:
            run_one("diag", "vanilla_raw", s, it, ref, normalize=False)

    if "diag100k" in args.suites:
        # full ASPEN protocol depth: does the oscillatory mode develop late?
        for s in seeds:
            run_one("diag100k", "vanilla_norm", s, it, ref)
            run_one("diag100k", "vanilla_raw", s, it, ref, normalize=False)

    if "invitro" in args.suites:
        # Positive control: does the budget clamp excess oscillation where it
        # DOES exist? A SIREN (omega=30) fit to sparse samples of a smooth
        # signal over-oscillates between samples -- an over-crossing patient.
        from kacrice.models import SIREN
        from kacrice.crossing import field_and_grad

        xg = torch.linspace(-1, 1, 2048, device=DEV).unsqueeze(-1)
        yg = torch.tanh(3 * xg[:, 0]) + 0.3 * torch.sin(2 * np.pi * xg[:, 0])
        lv = np.linspace(-1.2, 1.2, 16)
        bud = budgets_from_physics(lv, m_crossings=6.0, domain_len=2.0)["re"]
        bloss = CrossingBudgetLoss(lv, bud, eps=0.1).to(DEV)

        for s in seeds:
            g = torch.Generator().manual_seed(s)
            idx = torch.randperm(2048, generator=g)[:40].to(DEV)
            xs, ys = xg[idx], yg[idx]
            for cfg, w in (("mse", 0.0), ("mse_budget", 1.0)):
                if already_done("invitro", config=cfg, seed=s, iters=it):
                    print(f"invitro {cfg} seed{s}: skip", flush=True)
                    continue
                torch.manual_seed(s)
                m = SIREN(1, 1, 128, 3).to(DEV)
                op = torch.optim.Adam(m.parameters(), lr=1e-4)
                for k in range(2000):
                    pred, grad = field_and_grad(m, xs)
                    loss = (pred - ys).pow(2).mean()
                    if w > 0:
                        pm, gm = field_and_grad(m, xg)  # budget on dense probe
                        loss = loss + w * bloss(pm, gm.abs().squeeze(-1))
                    op.zero_grad(); loss.backward(); op.step()
                with torch.no_grad():
                    pd = m(xg).squeeze(-1)
                test_mse = float((pd - yg).pow(2).mean())
                pdn, gdn = field_and_grad(m, xg)
                from kacrice.crossing import crossing_density
                prof = crossing_density(
                    pdn.detach(), gdn.detach().abs().squeeze(-1),
                    torch.as_tensor(lv, dtype=torch.float32, device=DEV),
                    0.1).cpu().numpy()
                save_record("invitro", dict(
                    suite="invitro", config=cfg, seed=s, iters=it,
                    test_mse=test_mse, crossings=prof.tolist(),
                    budget=bud.tolist(),
                    pred=pd.cpu().numpy().tolist() if s == 0 else None))
                print(f"invitro {cfg} seed{s}: test_mse={test_mse:.5f} "
                      f"max_cross={prof.max():.2f} (budget {bud.max():.2f})",
                      flush=True)

    if "chaos" in args.suites:
        # Benjamin-Feir-unstable periodic testbed (b=2, c=-1.2): the regime
        # where excess-oscillation pathology and LEGITIMATE broadband content
        # coexist -- the original spec's intended battleground.
        CH = OUT / "reference_chaotic.npz"
        if CH.exists():
            zc = np.load(CH)
            xc, tc, Ac = zc["x"], zc["t"], zc["A"]
        else:
            print("computing chaotic CGLE reference...", flush=True)
            xc, tc, Ac = cgle_chaotic_reference()
            np.savez_compressed(CH, x=xc, t=tc, A=Ac)
        ref_c = (xc, tc, Ac)
        coeffs = (2.0, -1.2)
        factory = lambda: CGLEMlpPeriodic(L=64.0, t1=5.0)  # noqa: E731
        bud = budgets_from_reference(Ac, xc, LEVELS, EPS)
        pair = {"re": CrossingBudgetLoss(LEVELS, bud["re"], EPS).to(DEV),
                "im": CrossingBudgetLoss(LEVELS, bud["im"], EPS).to(DEV)}
        for s in seeds:
            ptsc = make_points_chaotic(xc, Ac[0], 64.0, 5.0, seed=s, device=DEV)
            common = dict(model_factory=factory, points=ptsc, coeffs=coeffs,
                          t1=5.0)
            run_one("chaos", "vanilla", s, it, ref_c, **common)
            run_one("chaos", "budget_ref", s, it, ref_c,
                    aux=("budget", pair), aux_w=1.0, **common)
            run_one("chaos", "grad_damp_w0.3", s, it, ref_c,
                    aux="grad_damp", aux_w=0.3, **common)

    if "exp3" in args.suites:
        for s in seeds:
            run_one("exp3", "vanilla", s, it, ref)

    if "exp4" in args.suites:
        for s in seeds:
            for src in ("ref", "phys", "scalar"):
                run_one("exp4", f"budget_{src}", s, it, ref,
                        aux=("budget", budget_pair(src, x_ref, A_ref)),
                        aux_w=1.0)

    if "exp5" in args.suites:
        for s in seeds:
            for w in (0.03, 0.3, 3.0):
                run_one("exp5", f"grad_damp_w{w}", s, it, ref,
                        aux="grad_damp", aux_w=w)
            for w in (0.001, 0.01, 0.1):  # curvature scale is much larger
                run_one("exp5", f"curv_damp_w{w}", s, it, ref,
                        aux="curv_damp", aux_w=w)

    if "exp6" in args.suites:
        for s in seeds:
            run_one("exp6", "fourier_pinn", s, it, ref, fourier_m=128)
            run_one("exp6", "curriculum", s, it, ref, schedule="curriculum")
            run_one("exp6", "causal", s, it, ref, causal=True)

    if "exp7" in args.suites:
        for s in seeds:
            run_one("exp7", "fourier_budget_ref", s, it, ref, fourier_m=128,
                    aux=("budget", budget_pair("ref", x_ref, A_ref)), aux_w=1.0)

    if "abl" in args.suites:
        for s in seeds[:2]:
            for w in (0.1, 10.0):
                run_one("abl", f"budget_ref_w{w}", s, it, ref,
                        aux=("budget", budget_pair("ref", x_ref, A_ref)),
                        aux_w=w)

    print("PHASE2 DONE", flush=True)


if __name__ == "__main__":
    main()
