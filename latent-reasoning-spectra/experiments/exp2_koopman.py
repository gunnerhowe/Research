"""E2 — Local-Jacobian vs routed dynamics; orbit-level Koopman/EDMD (PLAN.md).

Part A (routedness): for a subset of problems, compare the full influence Jacobian
G_{t->t+k} (all paths, incl. KV written at intermediate latent slots) against the
chained local product J_{t+k-1}...J_t.  R_{t,k} = ||G - prod J||_F / ||G||_F.

Part B (orbit-level operators): pooled EDMD over all M2 trajectories (c_1..c_6),
PCA-r coordinates, delay 1 and 2; Koopman spectrum; per-step orbit-level predictors
(koopman_residual, unit_participation) evaluated against branch labels and ablation
influence with the same statistics as E0.

Writes runs/exp2_routed.npz + results/exp2_koopman.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrspec.harness import Harness, N_LATENT  # noqa: E402
from lrspec.koopman import PooledEDMD, routedness  # noqa: E402
from lrspec.paths import DATA, RESULTS, ROOT  # noqa: E402
from lrspec.prosqa import load_problems  # noqa: E402
from lrspec import stats  # noqa: E402

import os as _os
RUNS = Path(_os.environ.get("LRSPEC_RUNS", ROOT / "runs"))
N_ROUTED = 100
TK_PAIRS = [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3)]
SEED = 0


def part_a():
    problems = load_problems(DATA / "prosqa_test.json")[:N_ROUTED]
    h = Harness("M2")
    R = {f"{t}_{k}": [] for t, k in TK_PAIRS}
    normG = {f"{t}_{k}": [] for t, k in TK_PAIRS}
    normChain = {f"{t}_{k}": [] for t, k in TK_PAIRS}
    t0 = time.time()
    for i, p in enumerate(problems):
        run = h.run_latent(p)
        J = {t: h.jacobian(run, t).cpu().numpy() for t in range(1, 6)}
        for t, k in TK_PAIRS:
            G = h.influence_jacobian(run, t, k).cpu().numpy()
            r = routedness(G, [J[t + j] for j in range(k)])
            R[f"{t}_{k}"].append(r["R"])
            normG[f"{t}_{k}"].append(r["norm_G"])
            normChain[f"{t}_{k}"].append(r["norm_chain"])
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"[E2a] {i+1}/{N_ROUTED} eta "
                  f"{(N_ROUTED-i-1)*el/(i+1)/60:.0f}min", flush=True)
    np.savez_compressed(RUNS / "exp2_routed.npz",
                        **{f"R_{k}": np.array(v) for k, v in R.items()},
                        **{f"normG_{k}": np.array(v) for k, v in normG.items()},
                        **{f"normChain_{k}": np.array(v) for k, v in normChain.items()})
    return {
        "n": len(problems),
        "R_median": {k: float(np.median(v)) for k, v in R.items()},
        "R_mean": {k: float(np.mean(v)) for k, v in R.items()},
        "norm_ratio_chain_over_G_median": {
            k: float(np.median(np.array(normChain[k]) / np.array(normG[k])))
            for k in normG},
    }


def part_b():
    cap = np.load(RUNS / "exp0_capture_M2.npz")
    abl = np.load(RUNS / "exp0_ablate_M2.npz")
    n = int(cap["n_done"][0])
    hs = cap["hs"][:n]
    labels = cap["labels"][:n]
    trajs = [hs[i, :N_LATENT] for i in range(n)]

    out = {}
    for r in (64, 128):
        ed = PooledEDMD(r=r, ridge=1e-6, delay=1).fit(trajs)
        lam = ed.spectrum()
        band = (np.abs(lam) >= 0.9) & (np.abs(lam) <= 1.1)
        res = {
            "explained_var": ed.explained_,
            "fit_residual": ed.resid_,
            "spectral_radius": float(np.abs(lam).max()),
            "n_unit_band": int(band.sum()),
            "unit_mass": float(np.abs(lam[band]).sum()),
            "top_eigs_abs": np.sort(np.abs(lam))[::-1][:12].tolist(),
        }
        # per-step orbit predictors
        kres, upart = [], []
        for tr in trajs:
            ps = ed.per_step(tr)
            kres.append(np.array(ps["koopman_residual"]))    # steps 1..5
            upart.append(np.array(ps["unit_participation"]))
        # vs branch labels (steps 1..5)
        ls, s_res, s_par = [], [], []
        for i in range(n):
            m = labels[i][:5] >= 0
            ls.append(labels[i][:5][m].astype(int))
            s_res.append(kres[i][m])
            s_par.append(upart[i][m])
        res["branch_auroc_koopman_residual"] = stats.pooled_auroc_ci(ls, s_res, seed=SEED)
        res["branch_auroc_unit_participation"] = stats.pooled_auroc_ci(ls, s_par, seed=SEED)
        # vs ablation influence (anchor): steps 1..5
        I = abl["I_mean"][:n]
        xs, ys = [], []
        for i in range(n):
            m = np.isfinite(I[i][:5])
            xs.append(upart[i][m]); ys.append(I[i][:5][m])
        res["anchor_spearman_unit_participation"] = stats.spearman_ci(xs, ys, seed=SEED)
        xs2 = [kres[i][np.isfinite(I[i][:5])] for i in range(n)]
        res["anchor_spearman_koopman_residual"] = stats.spearman_ci(xs2, ys, seed=SEED)
        out[f"edmd_r{r}"] = res

    # delay-2 EDMD: global spectrum only (per-step needs delay=1)
    ed2 = PooledEDMD(r=128, ridge=1e-6, delay=2).fit(trajs)
    lam2 = ed2.spectrum()
    out["edmd_r128_delay2"] = {
        "fit_residual": ed2.resid_,
        "spectral_radius": float(np.abs(lam2).max()),
        "top_eigs_abs": np.sort(np.abs(lam2))[::-1][:12].tolist(),
    }
    return out


def main():
    out = {"part_a_routedness": part_a(), "part_b_edmd": part_b()}

    # K3 assessment inputs: does ANY spectral predictor carry the signal?
    exp0 = json.load(open(RESULTS / "exp0_gate.json"))
    local_auroc = exp0["branch"]["M2"]["sigma1"]["auroc_ci"]
    koop_auroc = out["part_b_edmd"]["edmd_r128"][
        "branch_auroc_koopman_residual"]["auroc_ci"]
    local_ok = local_auroc[0] > 0.5
    koop_ok = koop_auroc[0] > 0.5
    out["K3"] = {
        "fires": not (local_ok or koop_ok),
        "local_sigma1_auroc_ci": local_auroc,
        "koopman_residual_auroc_ci": koop_auroc,
    }
    with open(RESULTS / "exp2_koopman.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps({"routedness_median": out["part_a_routedness"]["R_median"],
                      "K3": out["K3"]}, indent=2, default=float))


if __name__ == "__main__":
    main()
