"""Phase 3 experiment suite: differentiable Minkowski profiles.

Suites:
  exp9   in-vitro topology repair, seeded, with null controls:
         anchor-only / +smoothness null / +chi-cap alone (Goodhart exhibit) /
         +vector cap (ours). Grid-verified topology as ground truth.
  exp10  neural SDF from noisy point clouds: topology repair vs a
         persistent-homology baseline, plus the wall-clock cost axis.

Results append to results/phase3/<suite><tag>.json. Resume-safe.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from common import ROOT
from kacrice.crossing import CrossingBudgetLoss, crossing_density
from kacrice.minkowski import EulerProfileLoss, field_derivatives
from kacrice.models import SIREN

OUT = ROOT / "results" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

DEV = "cpu"
TAG = ""


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


# ------------------------------------------------------------------ exp9

GAUSS = lambda x, c, s: torch.exp(  # noqa: E731
    -((x - torch.tensor(c, dtype=x.dtype)) ** 2).sum(-1) / (2 * s**2))


def real_signal(x):
    return GAUSS(x, [-0.3, 0.0], 0.3)


def polluted_signal(x):
    return real_signal(x) + 0.7 * GAUSS(x, [0.55, 0.3], 0.15)


def grid_field(net, n=256, device="cpu"):
    xs = torch.linspace(-1, 1, n, device=device)
    gy, gx = torch.meshgrid(xs, xs, indexing="ij")
    with torch.no_grad():
        f = net(torch.stack([gx, gy], -1).reshape(-1, 2)).reshape(n, n)
    return f.cpu(), gx.cpu(), gy.cpu()


def grid_topo(f, u):
    m = (f > u).numpy()
    comps = ndimage.label(m)[1]
    holes = ndimage.label(~m)[1] - 1
    return int(comps), int(holes)


def run_exp9(seeds, iters=2000):
    levels = np.linspace(0.10, 0.95, 16)
    lv_t = torch.as_tensor(levels, dtype=torch.float32)

    # class perimeter budget from the clean analytic signal
    probe = torch.rand(200_000, 2) * 2 - 1
    pv = real_signal(probe)
    pg = probe - torch.tensor([-0.3, 0.0])
    pgn = (pv / 0.3**2) * pg.norm(dim=1)
    m1_budget = (1.5 * crossing_density(pv, pgn, lv_t, 0.05)).numpy()

    configs = ["anchor_only", "smooth_null", "chi_only", "vector_cap"]
    for seed in seeds:
        torch.manual_seed(seed)
        # pollute
        net0_state = None
        net = SIREN(2, 1, 128, 3, omega=15.0).to(DEV)
        dense = (torch.rand(8192, 2, generator=torch.Generator().manual_seed(
            seed)) * 2 - 1).to(DEV)
        y = polluted_signal(dense)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(1500):
            loss = (net(dense).squeeze(-1) - y).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net0_state = {k: v.clone() for k, v in net.state_dict().items()}

        anchor = dense[real_signal(dense) > 0.05]
        ya = real_signal(anchor)
        pts = (torch.rand(12000, 2, generator=torch.Generator().manual_seed(
            seed + 100)) * 2 - 1).to(DEV)

        for cfg in configs:
            if already_done("exp9", config=cfg, seed=seed):
                print(f"exp9 {cfg} seed{seed}: skip", flush=True)
                continue
            t0 = time.time()
            net.load_state_dict(net0_state)
            chi_cap = EulerProfileLoss(levels, [1.0] * 16, eps=0.05,
                                       volume=4.0, mode="cap").to(DEV)
            m1_cap = CrossingBudgetLoss(levels, m1_budget, eps=0.05).to(DEV)
            opt = torch.optim.Adam(net.parameters(), lr=5e-4)
            for _ in range(iters):
                la = (net(anchor).squeeze(-1) - ya).pow(2).mean()
                if cfg == "anchor_only":
                    loss = la
                elif cfg == "smooth_null":
                    v, g, h = field_derivatives(net, pts)
                    loss = la + 1e-3 * (h**2).sum(dim=(1, 2)).mean()
                elif cfg == "chi_only":
                    v, g, h = field_derivatives(net, pts)
                    loss = la + chi_cap(v, g, h)
                elif cfg == "vector_cap":
                    v, g, h = field_derivatives(net, pts)
                    loss = la + chi_cap(v, g, h) + m1_cap(v, g.norm(dim=1))
                opt.zero_grad(); loss.backward(); opt.step()

            f, gx, gy = grid_field(net)
            topo = {str(u): grid_topo(f, u) for u in (0.2, 0.3, 0.5)}
            spur_max = float(f[((gx - 0.55).abs() + (gy - 0.3).abs()
                                < 0.3)].max())
            clean = real_signal(torch.stack([gx, gy], -1).reshape(-1, 2)
                                ).reshape(256, 256)
            mse = float(((f - clean) ** 2).mean())
            ok = all(v == [1, 0] or tuple(v) == (1, 0)
                     for v in topo.values())
            rec = dict(suite="exp9", config=cfg, seed=seed, iters=iters,
                       topo=topo, spurious_max=spur_max, clean_mse=mse,
                       repaired=bool(ok), wall_s=time.time() - t0)
            save_record("exp9", rec)
            print(f"exp9 {cfg} seed{seed}: topo={topo} spur={spur_max:.3f} "
                  f"mse={mse:.5f} repaired={ok} ({time.time()-t0:.0f}s)",
                  flush=True)


# ------------------------------------------------------------------ exp10

def chamfer(a, b):
    """Symmetric Chamfer-L2 between two point sets (torch, chunked)."""
    d1 = torch.cdist(a, b).min(dim=1)[0].pow(2).mean()
    d2 = torch.cdist(b, a).min(dim=1)[0].pow(2).mean()
    return float(d1 + d2)


def run_exp10(seeds, iters=3000, grid_eval=96, omega=15.0, w_eik=0.1, n_dom=8000):
    from kacrice.sdf import (SHAPES, CubicalPHLoss, sample_pointcloud,
                             voxel_betti, occupancy_grid)
    from skimage.measure import marching_cubes

    ALLOWED = {"sphere": {0: 1, 1: 0, 2: 0}, "torus": {0: 1, 1: 1, 2: 0},
               "genus2": {0: 1, 1: 2, 2: 0}}
    BETTI_GT = {"sphere": (1, 0, 0), "torus": (1, 1, 0), "genus2": (1, 2, 0)}

    lv = np.linspace(-0.12, 0.12, 9)   # occupancy levels bracketing surface
    lv_t = torch.as_tensor(lv, dtype=torch.float32)

    for shape, (fn, chi_gt) in SHAPES.items():
        # class M1 (area) budget from the analytic shape
        probe = torch.rand(200_000, 3) * 2 - 1
        pv = -fn(probe)
        m1_clean = crossing_density(pv, torch.ones_like(pv), lv_t, 0.03)
        m1_budget = (1.5 * m1_clean).numpy()
        # clean surface reference points for Chamfer
        ref = sample_pointcloud(shape, n=4000, noise=0.0, outlier_frac=0.0,
                                seed=999)

        for seed in seeds:
            cloud = sample_pointcloud(shape, n=2000, noise=0.02,
                                      outlier_frac=0.05, seed=seed).to(DEV)
            dom = (torch.rand(n_dom, 3,
                              generator=torch.Generator().manual_seed(seed))
                   * 2 - 1).to(DEV)
            for cfg in ("eikonal_only", "smooth_null", "vector_match",
                        "ph_loss"):
                # PH costs ~340x more per iteration (smoke measurement); it
                # runs at half the iterations, recorded per run, and the
                # analysis reports both matched-iteration and matched-wallclock
                cfg_iters = iters // 2 if cfg == "ph_loss" else iters
                if already_done("exp10", shape=shape, config=cfg, seed=seed,
                                iters=cfg_iters, omega=omega, n_dom=n_dom):
                    print(f"exp10 {shape} {cfg} seed{seed}: skip", flush=True)
                    continue
                t0 = time.time()
                torch.manual_seed(seed)
                net = SIREN(3, 1, 128, 3, omega=omega).to(DEV)
                chi_loss = EulerProfileLoss(lv, [float(chi_gt)] * len(lv),
                                            eps=0.03, volume=8.0,
                                            mode="match").to(DEV)
                m1_cap = CrossingBudgetLoss(lv, m1_budget, eps=0.03).to(DEV)
                ph = (CubicalPHLoss(ALLOWED[shape], grid_n=48, device=DEV)
                      .to(DEV) if cfg == "ph_loss" else None)
                opt = torch.optim.Adam(net.parameters(), lr=5e-4)
                topo_time = 0.0
                for k in range(cfg_iters):
                    la = net(cloud).squeeze(-1).abs().mean()
                    v, g, h = field_derivatives(
                        net, dom, order=2 if cfg in ("smooth_null",
                                                     "vector_match") else 1)
                    le = (g.norm(dim=1) - 1).pow(2).mean()
                    loss = la + w_eik * le
                    tt = time.time()
                    if cfg == "smooth_null":
                        loss = loss + 1e-4 * (h**2).sum(dim=(1, 2)).mean()
                    elif cfg == "vector_match":
                        occ, gocc, hocc = -v, -g, -h
                        loss = (loss + 0.5 * chi_loss(occ, gocc, hocc)
                                + 0.5 * m1_cap(occ, gocc.norm(dim=1)))
                    elif cfg == "ph_loss":
                        loss = loss + 10.0 * ph(net)
                    topo_time += time.time() - tt
                    opt.zero_grad(); loss.backward(); opt.step()

                occ = occupancy_grid(net, n=grid_eval, device=DEV)
                betti = voxel_betti(occ, 0.0)
                topo_ok = tuple(betti) == BETTI_GT[shape]
                try:
                    verts, *_ = marching_cubes(occ, 0.0)
                    verts = torch.tensor(
                        verts / (grid_eval - 1) * 2 - 1, dtype=torch.float32)
                    idx = torch.randperm(verts.shape[0])[:4000]
                    cham = chamfer(verts[idx], ref)
                except Exception:
                    cham = float("nan")
                rec = dict(suite="exp10", shape=shape, config=cfg, seed=seed, omega=omega, w_eik=w_eik, n_dom=n_dom,
                           iters=cfg_iters, betti=list(betti),
                           betti_gt=list(BETTI_GT[shape]),
                           topo_correct=bool(topo_ok), chamfer=cham,
                           topo_ms_per_iter=1000 * topo_time / cfg_iters,
                           wall_s=time.time() - t0)
                save_record("exp10", rec)
                print(f"exp10 {shape} {cfg} seed{seed}: betti={betti} "
                      f"ok={topo_ok} cham={cham:.5f} "
                      f"topo={1000*topo_time/cfg_iters:.1f}ms/it "
                      f"({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=["exp9"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-list", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--iters10", type=int, default=3000)
    ap.add_argument("--omega", type=float, default=15.0)
    ap.add_argument("--w-eik", type=float, default=0.1)
    ap.add_argument("--n-dom", type=int, default=8000)
    args = ap.parse_args()

    global DEV, TAG
    DEV = args.device
    TAG = args.tag
    torch.set_num_threads(args.threads)
    seeds = ([int(s) for s in args.seed_list.split(",")] if args.seed_list
             else list(range(args.seeds)))
    print(f"device={DEV} seeds={seeds}", flush=True)
    if "exp9" in args.suites:
        run_exp9(seeds)
    if "exp10" in args.suites:
        run_exp10(seeds, iters=args.iters10, omega=args.omega, w_eik=args.w_eik, n_dom=args.n_dom)
    print("PHASE3 SUITE DONE", flush=True)


if __name__ == "__main__":
    main()





