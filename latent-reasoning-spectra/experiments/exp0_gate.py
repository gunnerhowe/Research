"""E0 — Week-1 go/no-go: spectral prediction + null controls (PLAN.md).

Phases (subcommands):
  capture         per-problem per-step Jacobians + invariants for one model
  capture-pruned  same for the paired pruned-real linear-chain twins (M2)
  ablate          step-ablation influence I_t (anchor ground truth; M2)
  analyze         AUROC/AP + bootstrap CIs, paired null tests, K1 verdict
                  -> results/exp0_gate.json

Array outputs go to runs/ (gitignored); JSON summaries to results/ (committed).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrspec.harness import Harness, N_LATENT  # noqa: E402
from lrspec.paths import DATA, RESULTS, ROOT  # noqa: E402
from lrspec.prosqa import load_problems, prune_to_linear  # noqa: E402
from lrspec.spectra import invariants, top_eig_subspace  # noqa: E402
from lrspec import stats  # noqa: E402

import os as _os
RUNS = Path(_os.environ.get("LRSPEC_RUNS", ROOT / "runs"))
RUNS.mkdir(exist_ok=True)

SCALAR_KEYS = ["rho", "sigma1", "sigma2", "n_expanding", "henrici", "henrici_norm",
               "kappa", "unit_mass", "n_unit_band", "trace", "fro"]


def _labels_for(p) -> np.ndarray:
    lab = np.full(N_LATENT, -1, dtype=np.int8)
    for t in range(1, N_LATENT + 1):
        b = p.branch_label(t)
        lab[t - 1] = -1 if b is None else b
    return lab


def _degrees_for(p) -> np.ndarray:
    deg = np.full(N_LATENT, -1, dtype=np.int16)
    for t in range(1, N_LATENT + 1):
        d = p.branch_degree(t)
        deg[t - 1] = -1 if d is None else d
    return deg


def capture(model_key: str, problems, out_path: Path, store_dirs: bool):
    h = Harness(model_key)
    N = len(problems)
    inv_arr = np.full((N, N_LATENT, len(SCALAR_KEYS)), np.nan, dtype=np.float64)
    eig_abs = np.full((N, N_LATENT, 16), np.nan, dtype=np.float32)
    hs_arr = np.full((N, N_LATENT + 1, 768), np.nan, dtype=np.float32)
    v1_arr = np.full((N, N_LATENT, 768), np.nan, dtype=np.float32) if store_dirs else None
    u1_arr = np.full((N, N_LATENT, 768), np.nan, dtype=np.float32) if store_dirs else None
    eigQ_arr = np.full((N, N_LATENT, 768, 4), np.nan, dtype=np.float32) if store_dirs else None
    labels = np.zeros((N, N_LATENT), dtype=np.int8)
    degrees = np.zeros((N, N_LATENT), dtype=np.int16)
    hops = np.zeros(N, dtype=np.int16)
    margins = np.zeros(N)
    correct = np.zeros(N, dtype=bool)
    prob_idx = np.zeros(N, dtype=np.int32)

    start_i = 0
    if out_path.exists():
        # resume from the last periodic checkpoint (crash/OOM resilience)
        old = np.load(out_path)
        if old["inv"].shape[0] == N and int(old["n_done"][0]) < N:
            start_i = int(old["n_done"][0])
            inv_arr[:start_i] = old["inv"][:start_i]
            eig_abs[:start_i] = old["eig_abs"][:start_i]
            hs_arr[:start_i] = old["hs"][:start_i]
            labels[:start_i] = old["labels"][:start_i]
            degrees[:start_i] = old["degrees"][:start_i]
            hops[:start_i] = old["hops"][:start_i]
            margins[:start_i] = old["margins"][:start_i]
            correct[:start_i] = old["correct"][:start_i]
            prob_idx[:start_i] = old["prob_idx"][:start_i]
            if store_dirs and "v1" in old:
                v1_arr[:start_i] = old["v1"][:start_i]
                u1_arr[:start_i] = old["u1"][:start_i]
                eigQ_arr[:start_i] = old["eigQ"][:start_i]
            print(f"[{model_key}] resuming from {start_i}/{N}", flush=True)
        elif int(old["n_done"][0]) >= N:
            print(f"[{model_key}] already complete ({N}), skipping", flush=True)
            return

    t0 = time.time()
    for i, p in enumerate(problems):
        if i < start_i:
            continue
        run = h.run_latent(p)
        ro = h.readout(run, p)
        margins[i] = ro["margin"]
        correct[i] = ro["greedy_text"].split("#")[-1].replace(",", "").strip() == p.answer
        hs_arr[i] = run.hs.cpu().numpy()
        labels[i] = _labels_for(p)
        degrees[i] = _degrees_for(p)
        hops[i] = p.n_hops
        prob_idx[i] = p.idx
        for t in range(1, N_LATENT + 1):
            J = h.jacobian(run, t).cpu().numpy()
            eig_pair: list = [] if store_dirs else None
            inv = invariants(J, eig_out=eig_pair)
            inv_arr[i, t - 1] = [inv[k] for k in SCALAR_KEYS]
            eig_abs[i, t - 1] = inv["eig_abs_sorted"]
            if store_dirs:
                v1_arr[i, t - 1] = inv["v1"]
                u1_arr[i, t - 1] = inv["u1"]
                _, Q = top_eig_subspace(J, k=4, eig=eig_pair[0])
                eigQ_arr[i, t - 1] = Q[:, :4].astype(np.float32)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"[{model_key}] {i+1}/{N}  {el/ (i+1):.1f}s/problem  "
                  f"eta {(N - i - 1) * el / (i+1) / 60:.0f}min", flush=True)
            # periodic checkpoint
            _save(out_path, inv_arr, eig_abs, hs_arr, labels, degrees, hops,
                  margins, correct, prob_idx, v1_arr, u1_arr, eigQ_arr, n_done=i + 1)
    _save(out_path, inv_arr, eig_abs, hs_arr, labels, degrees, hops,
          margins, correct, prob_idx, v1_arr, u1_arr, eigQ_arr, n_done=N)
    print(f"[{model_key}] done in {(time.time()-t0)/60:.1f} min -> {out_path}")


def _save(path, inv_arr, eig_abs, hs_arr, labels, degrees, hops, margins, correct,
          prob_idx, v1_arr, u1_arr, eigQ_arr, n_done):
    d = dict(inv=inv_arr, eig_abs=eig_abs, hs=hs_arr, labels=labels, degrees=degrees,
             hops=hops, margins=margins, correct=correct, prob_idx=prob_idx,
             n_done=np.array([n_done]), scalar_keys=np.array(SCALAR_KEYS))
    if v1_arr is not None:
        d["v1"] = v1_arr
        d["u1"] = u1_arr
        d["eigQ"] = eigQ_arr
    np.savez_compressed(path, **d)


def ablate(problems, out_path: Path, capture_path: Path):
    """Step-ablation influence I_t for M2 (mean-thought + zero replacement)."""
    cap = np.load(capture_path)
    # dataset-mean fed thought per slot t: mean over problems of hs[:, t-1]
    mean_thoughts = torch.tensor(np.nanmean(cap["hs"][:, :N_LATENT], axis=0),
                                 device="cuda")
    h = Harness("M2")
    N = len(problems)
    I_mean = np.full((N, N_LATENT), np.nan)
    I_zero = np.full((N, N_LATENT), np.nan)
    flips_mean = np.zeros((N, N_LATENT), dtype=bool)
    t0 = time.time()
    for i, p in enumerate(problems):
        run = h.run_latent(p)
        base = h.readout(run, p)
        for t in range(1, N_LATENT + 1):
            r = h.rerun_from(run, p, t, mean_thoughts[t - 1])
            I_mean[i, t - 1] = abs(r["margin"] - base["margin"])
            flips_mean[i, t - 1] = (r["margin"] > 0) != (base["margin"] > 0)
            r = h.rerun_from(run, p, t, torch.zeros(768, device="cuda"))
            I_zero[i, t - 1] = abs(r["margin"] - base["margin"])
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"[ablate] {i+1}/{N}  eta {(N-i-1)*el/(i+1)/60:.0f}min", flush=True)
            np.savez_compressed(out_path, I_mean=I_mean, I_zero=I_zero,
                                flips_mean=flips_mean, n_done=np.array([i + 1]))
    np.savez_compressed(out_path, I_mean=I_mean, I_zero=I_zero,
                        flips_mean=flips_mean, n_done=np.array([N]))
    print(f"[ablate] done in {(time.time()-t0)/60:.1f} min")


# ---------------- analysis ----------------

def _inv_col(cap, key):
    return cap["inv"][:, :, list(cap["scalar_keys"]).index(key)]


def _predictor_matrix(cap) -> dict[str, np.ndarray]:
    """All predictor scores, shape (N, 6)."""
    keys = [str(k) for k in cap["scalar_keys"]]
    out = {k: cap["inv"][:, :, keys.index(k)] for k in keys}
    hs = cap["hs"]
    out["baseline_step"] = np.tile(np.arange(1, N_LATENT + 1), (hs.shape[0], 1)).astype(float)
    out["baseline_cnorm"] = np.linalg.norm(hs[:, :N_LATENT], axis=2)
    out["baseline_dc"] = np.linalg.norm(np.diff(hs, axis=1), axis=2)[:, :N_LATENT]
    return out


def _branch_auroc(cap, score: np.ndarray, seed=0) -> dict:
    """Pooled AUROC of score vs branch labels (padding excluded), cluster bootstrap."""
    labels = cap["labels"]
    ls, ss = [], []
    for i in range(labels.shape[0]):
        m = labels[i] >= 0
        if m.sum() == 0 or labels[i][m].min() == labels[i][m].max():
            # keep single-class problems; pooled AUROC handles them, bootstrap skips empties
            pass
        ls.append(labels[i][m].astype(int))
        ss.append(score[i][m])
    return stats.pooled_auroc_ci(ls, ss, seed=seed)


def _branch_auroc_stratified(cap, score: np.ndarray, seed=0) -> dict:
    """Step-index-confound control: z-score the predictor WITHIN each step column
    (across problems), then pool.  Also per-step AUROCs.  Secondary analysis."""
    labels = cap["labels"]
    z = np.full_like(score, np.nan, dtype=float)
    per_step = {}
    for t in range(labels.shape[1]):
        m = labels[:, t] >= 0
        if m.sum() < 10:
            continue
        col = score[m, t]
        mu, sd = np.nanmean(col), np.nanstd(col)
        z[m, t] = (col - mu) / (sd if sd > 0 else 1.0)
        lab_t = labels[m, t].astype(int)
        if lab_t.min() != lab_t.max():
            from sklearn.metrics import roc_auc_score
            per_step[f"t{t+1}"] = {
                "auroc": float(roc_auc_score(lab_t, col)),
                "n": int(m.sum()), "n_pos": int(lab_t.sum()),
            }
    ls, ss = [], []
    for i in range(labels.shape[0]):
        m = (labels[i] >= 0) & np.isfinite(z[i])
        ls.append(labels[i][m].astype(int))
        ss.append(z[i][m])
    out = stats.pooled_auroc_ci(ls, ss, seed=seed)
    out["per_step"] = per_step
    return out


def _anchor_stats(cap, abl, score: np.ndarray, seed=0) -> dict:
    """Spearman + top-tercile AUROC of score vs ablation influence I_mean."""
    I = abl["I_mean"]
    xs, ys, ls, ss = [], [], [], []
    for i in range(I.shape[0]):
        xi, yi = score[i], I[i]
        m = np.isfinite(xi) & np.isfinite(yi)
        if m.sum() < 3:
            continue
        xs.append(xi[m]); ys.append(yi[m])
        thr = np.quantile(yi[m], 2 / 3)
        ls.append((yi[m] >= thr).astype(int))
        ss.append(xi[m])
    sp = stats.spearman_ci(xs, ys, seed=seed)
    au = stats.pooled_auroc_ci(ls, ss, seed=seed)
    return {"spearman": sp, "auroc_top_tercile": au}


def analyze(seed: int = 0):
    capM2 = np.load(RUNS / "exp0_capture_M2.npz")
    capM3 = np.load(RUNS / "exp0_capture_M3.npz")
    capM4 = np.load(RUNS / "exp0_capture_M4.npz")
    abl = np.load(RUNS / "exp0_ablate_M2.npz")
    out = {"n_problems": int(capM2["n_done"][0])}

    # ---- branch prediction (E0a) ----
    branch = {}
    for name, cap in [("M2", capM2), ("M3", capM3), ("M4", capM4)]:
        preds = _predictor_matrix(cap)
        branch[name] = {k: _branch_auroc(cap, v, seed=seed) for k, v in preds.items()}
    out["branch"] = branch

    # ---- step-index-confound control (secondary): within-step z-scored AUROC ----
    predsM2 = _predictor_matrix(capM2)
    out["branch_stratified"] = {
        k: _branch_auroc_stratified(capM2, predsM2[k], seed=seed)
        for k in ["sigma1", "henrici_norm", "unit_mass", "baseline_cnorm", "baseline_dc"]
    }

    # ---- anchor prediction (E0b, M2) ----
    preds2 = _predictor_matrix(capM2)
    out["anchor"] = {k: _anchor_stats(capM2, abl, v, seed=seed)
                     for k, v in preds2.items()}

    # ---- paired branch-step separation, M2 vs nulls (K1 core) ----
    # per problem: mean sigma1 at branch steps minus at non-branch steps
    def sep(cap, key="sigma1"):
        s = _inv_col(cap, key)
        lab = cap["labels"]
        vals = []
        for i in range(lab.shape[0]):
            m = lab[i] >= 0
            b, nb = s[i][m & (lab[i] == 1)], s[i][m & (lab[i] == 0)]
            if len(b) and len(nb):
                vals.append(b.mean() - nb.mean())
            else:
                vals.append(np.nan)
        return np.array(vals)

    seps = {name: sep(cap) for name, cap in
            [("M2", capM2), ("M3", capM3), ("M4", capM4)]}
    mask = np.isfinite(seps["M2"]) & np.isfinite(seps["M3"]) & np.isfinite(seps["M4"])
    out["separation_sigma1"] = {
        "M2_mean": float(np.nanmean(seps["M2"])),
        "M3_mean": float(np.nanmean(seps["M3"])),
        "M4_mean": float(np.nanmean(seps["M4"])),
        "wilcoxon_M2_vs_M3": stats.paired_wilcoxon(seps["M2"][mask], seps["M3"][mask]),
        "wilcoxon_M2_vs_M4": stats.paired_wilcoxon(seps["M2"][mask], seps["M4"][mask]),
    }

    # ---- pruned-twin paired control ----
    pruned_path = RUNS / "exp0_capture_M2_pruned.npz"
    if pruned_path.exists():
        capP = np.load(pruned_path)
        orig_by_idx = {int(v): i for i, v in enumerate(capM2["prob_idx"])}
        s_o = _inv_col(capM2, "sigma1")
        s_p = _inv_col(capP, "sigma1")
        labP = capM2["labels"]  # ORIGINAL branch labels, aligned to original steps
        d_branch = []  # sigma1 drop at formerly-branch steps
        d_nonbranch = []
        acc_pruned = float(np.mean(capP["correct"][: int(capP["n_done"][0])]))
        for j, pi in enumerate(capP["prob_idx"][: int(capP["n_done"][0])]):
            i = orig_by_idx.get(int(pi))
            if i is None:
                continue
            m = labP[i] >= 0
            for t in np.where(m)[0]:
                diff = s_o[i, t] - s_p[j, t]
                (d_branch if labP[i, t] == 1 else d_nonbranch).append(diff)
        from scipy.stats import mannwhitneyu, wilcoxon
        res = {"n_pruned": int(capP["n_done"][0]), "accuracy_pruned": acc_pruned,
               "mean_drop_at_branch_steps": float(np.mean(d_branch)),
               "mean_drop_at_nonbranch_steps": float(np.mean(d_nonbranch)),
               "wilcoxon_branch_drop_p": float(wilcoxon(d_branch).pvalue),
               "mwu_branch_vs_nonbranch_p": float(
                   mannwhitneyu(d_branch, d_nonbranch, alternative="greater").pvalue)}
        out["pruned_control"] = res

    # ---- K1 verdict (operationalization in PLAN.md sec. 5) ----
    m2 = branch["M2"]["sigma1"]
    m3 = branch["M3"]["sigma1"]
    m4 = branch["M4"]["sigma1"]
    ci_overlap_m3 = not (m2["auroc_ci"][0] > m3["auroc_ci"][1])
    ci_overlap_m4 = not (m2["auroc_ci"][0] > m4["auroc_ci"][1])
    sep_ratio_m3 = out["separation_sigma1"]["M3_mean"] / out["separation_sigma1"]["M2_mean"] \
        if out["separation_sigma1"]["M2_mean"] else np.nan
    sep_ratio_m4 = out["separation_sigma1"]["M4_mean"] / out["separation_sigma1"]["M2_mean"] \
        if out["separation_sigma1"]["M2_mean"] else np.nan
    k1_fires = (ci_overlap_m3 and sep_ratio_m3 >= 0.5) or \
               (ci_overlap_m4 and sep_ratio_m4 >= 0.5)
    out["K1"] = {
        "fires": bool(k1_fires),
        "ci_overlap_M3": ci_overlap_m3, "ci_overlap_M4": ci_overlap_m4,
        "sep_ratio_M3": float(sep_ratio_m3), "sep_ratio_M4": float(sep_ratio_m4),
    }

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "exp0_gate.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps({"K1": out["K1"],
                      "branch_sigma1": {k: branch[k]["sigma1"]["auroc"] for k in branch},
                      "anchor_unit_mass": out["anchor"]["unit_mass"]["spearman"]},
                     indent=2, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["capture", "capture-pruned", "ablate", "analyze"])
    ap.add_argument("--model", default="M2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-pruned", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    problems = load_problems(DATA / "prosqa_test.json")
    if args.limit:
        problems = problems[: args.limit]

    if args.cmd == "capture":
        capture(args.model, problems,
                RUNS / f"exp0_capture_{args.model}.npz",
                store_dirs=args.model == "M2")
    elif args.cmd == "capture-pruned":
        twins = []
        for p in problems:
            q = prune_to_linear(p)
            if q is not None:
                twins.append(q)
            if len(twins) >= args.n_pruned:
                break
        print(f"pruned twins: {len(twins)}")
        capture("M2", twins, RUNS / "exp0_capture_M2_pruned.npz", store_dirs=False)
    elif args.cmd == "ablate":
        ablate(problems, RUNS / "exp0_ablate_M2.npz", RUNS / "exp0_capture_M2.npz")
    elif args.cmd == "analyze":
        analyze(seed=args.seed)


if __name__ == "__main__":
    main()
