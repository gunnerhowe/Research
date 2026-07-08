"""E1 — Causal validation of the spectrally-identified directions (PLAN.md).

For M2, on a fixed subset of problems and steps t=1..5:
  A) Directional perturbations c_t -> c_t + s * eps * ||c_t|| * d, s in {+1,-1}:
       d = v1(J_t)      top right singular vector (transient-expansion direction)
       d = q1(J_t)      leading direction of the dominant eigen-subspace basis
       d = random       n_rand magnitude-matched random unit vectors (seeded)
     Metrics: |Delta margin| (mean over signs), flip rate (margin sign change),
     steer-to-neg rate (margin becomes negative).
  B) Slow-mode subspace ablation: project out the top-4 eigen-subspace span(Q) of J_t
     from c_t, vs random 4-dim subspaces (matched by removed-norm via projection of the
     SAME c_t onto a random 4-dim subspace).
Success criterion (K2): spectral effects >= 2x random at matched eps with
paired Wilcoxon p < 0.01.

Writes runs/exp1_causal.npz + results/exp1_causal.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrspec.causal import random_orthonormal, random_unit_directions  # noqa: E402
from lrspec.harness import Harness, N_LATENT  # noqa: E402
from lrspec.paths import DATA, RESULTS, ROOT  # noqa: E402
from lrspec.prosqa import load_problems  # noqa: E402
from lrspec import stats  # noqa: E402

import os as _os
RUNS = Path(_os.environ.get("LRSPEC_RUNS", ROOT / "runs"))
N_PROBLEMS = 120
STEPS = [1, 2, 3, 4, 5]
EPS = [0.1, 0.3]
N_RAND = 5
SEED = 0


def main():
    cap = np.load(RUNS / "exp0_capture_M2.npz")
    v1 = cap["v1"]      # (N, 6, 768)
    eigQ = cap["eigQ"]  # (N, 6, 768, 4)
    prob_idx = cap["prob_idx"]
    problems = load_problems(DATA / "prosqa_test.json")
    by_idx = {p.idx: p for p in problems}

    h = Harness("M2")
    dev = h.device

    # effects[arm][(eps)] -> list over (problem, step) of |dmargin|
    arms = ["v1", "q1"] + [f"rand{r}" for r in range(N_RAND)]
    eff = {a: {e: [] for e in EPS} for a in arms}
    flip = {a: {e: [] for e in EPS} for a in arms}
    toneg = {a: {e: [] for e in EPS} for a in arms}
    sub_eff = {"spectral": [], "random": []}
    sub_flip = {"spectral": [], "random": []}
    meta = []

    t0 = time.time()
    n_done = 0
    for i in range(min(N_PROBLEMS, len(prob_idx))):
        p = by_idx[int(prob_idx[i])]
        run = h.run_latent(p)
        base = h.readout(run, p)
        bm = base["margin"]
        rand_dirs = random_unit_directions(N_RAND, 768, SEED + 1000 * i, dev)
        for t in STEPS:
            c = h.fed_vector(run, t)
            cn = c.norm()
            dirs = {
                "v1": torch.tensor(v1[i, t - 1], device=dev),
                "q1": torch.tensor(eigQ[i, t - 1, :, 0], device=dev),
            }
            for r in range(N_RAND):
                dirs[f"rand{r}"] = rand_dirs[r]
            for a, d in dirs.items():
                d = d / d.norm()
                for e in EPS:
                    dm = []
                    fl = []
                    tn = []
                    for s in (+1.0, -1.0):
                        r_ = h.rerun_from(run, p, t, c + s * e * cn * d, greedy=False)
                        dm.append(abs(r_["margin"] - bm))
                        fl.append((r_["margin"] > 0) != (bm > 0))
                        tn.append(r_["margin"] < 0)
                    eff[a][e].append(float(np.mean(dm)))
                    flip[a][e].append(float(np.mean(fl)))
                    toneg[a][e].append(float(np.mean(tn)))

            # B) subspace ablation
            Q = torch.tensor(eigQ[i, t - 1], device=dev)  # 768 x 4
            c_new = c - Q @ (Q.T @ c)
            r_ = h.rerun_from(run, p, t, c_new, greedy=False)
            sub_eff["spectral"].append(abs(r_["margin"] - bm))
            sub_flip["spectral"].append(float((r_["margin"] > 0) != (bm > 0)))
            re_, rf_ = [], []
            for rseed in range(N_RAND):
                Qr = random_orthonormal(768, 4, SEED + 7919 * i + 31 * t + rseed, dev)
                c_new = c - Qr @ (Qr.T @ c)
                r_ = h.rerun_from(run, p, t, c_new, greedy=False)
                re_.append(abs(r_["margin"] - bm))
                rf_.append(float((r_["margin"] > 0) != (bm > 0)))
            sub_eff["random"].append(float(np.mean(re_)))
            sub_flip["random"].append(float(np.mean(rf_)))
            meta.append({"i": i, "t": t, "branch": int(cap["labels"][i, t - 1]),
                         "base_margin": float(bm)})
        n_done += 1
        if n_done % 10 == 0:
            el = time.time() - t0
            print(f"[E1] {n_done}/{N_PROBLEMS} eta "
                  f"{(N_PROBLEMS-n_done)*el/n_done/60:.0f}min", flush=True)

    # ---- summarize ----
    out = {"n_problems": n_done, "steps": STEPS, "eps": EPS, "n_rand": N_RAND}
    rand_arms = [f"rand{r}" for r in range(N_RAND)]
    for e in EPS:
        v1e = np.array(eff["v1"][e])
        q1e = np.array(eff["q1"][e])
        rnd = np.mean([eff[a][e] for a in rand_arms], axis=0)
        out[f"eps_{e}"] = {
            "v1_mean": float(v1e.mean()), "q1_mean": float(q1e.mean()),
            "rand_mean": float(rnd.mean()),
            "ratio_v1_over_rand": float(v1e.mean() / rnd.mean()),
            "ratio_q1_over_rand": float(q1e.mean() / rnd.mean()),
            "wilcoxon_v1_vs_rand": stats.paired_wilcoxon(v1e, rnd),
            "wilcoxon_q1_vs_rand": stats.paired_wilcoxon(q1e, rnd),
            "flip_v1": float(np.mean(flip["v1"][e])),
            "flip_rand": float(np.mean([flip[a][e] for a in rand_arms])),
            "toneg_v1": float(np.mean(toneg["v1"][e])),
            "toneg_rand": float(np.mean([toneg[a][e] for a in rand_arms])),
        }
        # branch-step conditioning: is the v1 effect largest at branch steps?
        br = np.array([m["branch"] for m in meta])
        out[f"eps_{e}"]["v1_mean_at_branch"] = float(v1e[br == 1].mean())
        out[f"eps_{e}"]["v1_mean_at_nonbranch"] = float(v1e[br == 0].mean())

    se = np.array(sub_eff["spectral"]); sr = np.array(sub_eff["random"])
    out["subspace"] = {
        "spectral_mean": float(se.mean()), "random_mean": float(sr.mean()),
        "ratio": float(se.mean() / sr.mean()),
        "wilcoxon": stats.paired_wilcoxon(se, sr),
        "flip_spectral": float(np.mean(sub_flip["spectral"])),
        "flip_random": float(np.mean(sub_flip["random"])),
    }

    # ---- K2 verdict (PLAN.md sec. 5): ratio >= 2 and p < 0.01 at matched eps ----
    k2_ok = any(
        out[f"eps_{e}"]["ratio_v1_over_rand"] >= 2.0
        and out[f"eps_{e}"]["wilcoxon_v1_vs_rand"]["p"] < 0.01
        for e in EPS
    )
    out["K2"] = {"fires": (not k2_ok), "criterion": "v1/rand >= 2x and p < 0.01 at some eps"}

    np.savez_compressed(RUNS / "exp1_causal.npz",
                        meta=json.dumps(meta),
                        **{f"eff_{a}_{e}": np.array(eff[a][e]) for a in arms for e in EPS},
                        sub_spectral=se, sub_random=sr)
    with open(RESULTS / "exp1_causal.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps({k: out[k] for k in ["K2", "subspace"]}, indent=2, default=float))
    for e in EPS:
        print(e, {k: round(v, 3) for k, v in out[f"eps_{e}"].items()
                  if isinstance(v, float)})


if __name__ == "__main__":
    main()
