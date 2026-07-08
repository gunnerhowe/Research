"""E3 — Scale / robustness (PLAN.md).

Subcommands:
  capture-valid    M2 capture on prosqa_valid (held-out replication of E0a)
  capture-natlin   M2 capture on the 5 natural linear-chain train instances
  capture-epochs   M2 capture on 100 test problems at curriculum epochs 10/20/30/40
                   (downloads those checkpoints on first use)
  analyze          bootstrap-seed stability (5 seeds), valid-set AUROC, epoch sweep
                   -> results/exp3_robustness.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrspec.paths import DATA, MODELS, RESULTS, ROOT  # noqa: E402
from lrspec.prosqa import load_problems  # noqa: E402
from lrspec import stats  # noqa: E402

RUNS = ROOT / "runs"
EPOCHS = [10, 20, 30, 40]
N_EPOCH_PROBLEMS = 100


def _capture(problems, out_path, checkpoint_path=None, store_dirs=False):
    from exp0_gate import capture  # reuse

    import exp0_gate
    # capture() constructs Harness internally; patch for custom checkpoints
    from lrspec.harness import Harness

    if checkpoint_path is None:
        capture("M2", problems, out_path, store_dirs=store_dirs)
        return
    # custom checkpoint: temporary Harness with checkpoint override
    h = Harness("M2", checkpoint_path=checkpoint_path)
    _capture_with_harness(h, problems, out_path)


def _capture_with_harness(h, problems, out_path):
    import time

    from exp0_gate import SCALAR_KEYS, _labels_for, _degrees_for, _save
    from lrspec.harness import N_LATENT
    from lrspec.spectra import invariants

    N = len(problems)
    inv_arr = np.full((N, N_LATENT, len(SCALAR_KEYS)), np.nan)
    eig_abs = np.full((N, N_LATENT, 16), np.nan, dtype=np.float32)
    hs_arr = np.full((N, N_LATENT + 1, 768), np.nan, dtype=np.float32)
    labels = np.zeros((N, N_LATENT), dtype=np.int8)
    degrees = np.zeros((N, N_LATENT), dtype=np.int16)
    hops = np.zeros(N, dtype=np.int16)
    margins = np.zeros(N)
    correct = np.zeros(N, dtype=bool)
    prob_idx = np.zeros(N, dtype=np.int32)
    t0 = time.time()
    for i, p in enumerate(problems):
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
            inv = invariants(J)
            inv_arr[i, t - 1] = [inv[k] for k in SCALAR_KEYS]
            eig_abs[i, t - 1] = inv["eig_abs_sorted"]
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{N} eta {(N-i-1)*el/(i+1)/60:.0f}min", flush=True)
    _save(out_path, inv_arr, eig_abs, hs_arr, labels, degrees, hops, margins,
          correct, prob_idx, None, None, None, n_done=N)


def capture_epochs():
    from huggingface_hub import hf_hub_download
    import shutil

    from lrspec.harness import Harness

    problems = load_problems(DATA / "prosqa_test.json")[:N_EPOCH_PROBLEMS]
    for ep in EPOCHS:
        dst = MODELS / f"coconut_ep{ep}.pt"
        if not dst.exists():
            p = hf_hub_download("bmarti44/coconut-curriculum-checkpoints",
                                f"coconut/checkpoint_{ep}")
            shutil.copy(p, dst)
        print(f"[epoch {ep}]", flush=True)
        h = Harness("M2", checkpoint_path=dst)
        _capture_with_harness(h, problems, RUNS / f"exp3_capture_M2_ep{ep}.npz")
        del h
        import torch
        torch.cuda.empty_cache()


def analyze():
    from exp0_gate import _branch_auroc, _inv_col, _predictor_matrix

    out = {}

    # ---- bootstrap-seed stability of the headline number ----
    capM2 = np.load(RUNS / "exp0_capture_M2.npz")
    preds = _predictor_matrix(capM2)
    seeds = {}
    for s in range(5):
        seeds[s] = _branch_auroc(capM2, preds["sigma1"], seed=s)
    out["seed_stability_sigma1"] = {
        "auroc": seeds[0]["auroc"],
        "ci_by_seed": {s: seeds[s]["auroc_ci"] for s in seeds},
    }

    # ---- held-out valid split ----
    vp = RUNS / "exp3_capture_M2_valid.npz"
    if vp.exists():
        capV = np.load(vp)
        predsV = _predictor_matrix(capV)
        out["valid_split"] = {
            "n": int(capV["n_done"][0]),
            "accuracy": float(np.mean(capV["correct"][: int(capV["n_done"][0])])),
            "branch_auroc_sigma1": _branch_auroc(capV, predsV["sigma1"]),
            "branch_auroc_henrici_norm": _branch_auroc(capV, predsV["henrici_norm"]),
        }

    # ---- natural linear chains (secondary check) ----
    np_path = RUNS / "exp3_capture_M2_natlin.npz"
    if np_path.exists():
        capN = np.load(np_path)
        nn = int(capN["n_done"][0])
        s1 = _inv_col(capN, "sigma1")
        lab = capN["labels"]
        vals = [s1[i][lab[i] >= 0] for i in range(nn)]
        # compare against E0 branch/non-branch means
        s1_m2 = _inv_col(capM2, "sigma1")
        labM2 = capM2["labels"]
        m = labM2 >= 0
        out["natural_linear_chains"] = {
            "n": nn,
            "accuracy": float(np.mean(capN["correct"][:nn])),
            "sigma1_mean": float(np.concatenate(vals).mean()),
            "M2_sigma1_mean_branch": float(s1_m2[m & (labM2 == 1)].mean()),
            "M2_sigma1_mean_nonbranch": float(s1_m2[m & (labM2 == 0)].mean()),
        }

    # ---- curriculum epoch sweep ----
    sweep = {}
    for ep in EPOCHS:
        pth = RUNS / f"exp3_capture_M2_ep{ep}.npz"
        if not pth.exists():
            continue
        capE = np.load(pth)
        predsE = _predictor_matrix(capE)
        sweep[ep] = {
            "accuracy": float(np.mean(capE["correct"][: int(capE["n_done"][0])])),
            "branch_auroc_sigma1": _branch_auroc(capE, predsE["sigma1"]),
        }
    if sweep:
        # matched-subset AUROC for the final model (first N_EPOCH_PROBLEMS problems)
        n_sub = N_EPOCH_PROBLEMS
        sub_labels = capM2["labels"][:n_sub]
        sub_score = preds["sigma1"][:n_sub]
        ls = [sub_labels[i][sub_labels[i] >= 0].astype(int) for i in range(n_sub)]
        ss = [sub_score[i][sub_labels[i] >= 0] for i in range(n_sub)]
        sweep["best"] = {
            "accuracy": float(np.mean(capM2["correct"][:n_sub])),
            "branch_auroc_sigma1": stats.pooled_auroc_ci(ls, ss),
        }
        out["epoch_sweep"] = sweep

    # M1 accuracy context from sanity gate
    sj = RESULTS / "sanity_accuracy.json"
    if sj.exists():
        out["sanity_accuracy"] = json.load(open(sj))

    with open(RESULTS / "exp3_robustness.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float)[:2000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["capture-valid", "capture-natlin",
                                    "capture-epochs", "analyze"])
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    if args.cmd == "capture-valid":
        problems = load_problems(DATA / "prosqa_valid.json")
        _capture(problems, RUNS / "exp3_capture_M2_valid.npz")
    elif args.cmd == "capture-natlin":
        problems = load_problems(DATA / "prosqa_train.json")
        lin = [p for p in problems if p.is_linear_chain]
        print(f"natural linear chains: {len(lin)}")
        _capture(lin, RUNS / "exp3_capture_M2_natlin.npz")
    elif args.cmd == "capture-epochs":
        capture_epochs()
    elif args.cmd == "analyze":
        analyze()


if __name__ == "__main__":
    main()
