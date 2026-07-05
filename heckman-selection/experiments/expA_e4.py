"""A-E4: benchmark matrices as missing-not-at-random panels (vignette).

Public model x task benchmark matrices are selective-reporting panels:
papers report their method on a CHOSEN subset of standard datasets. This
script runs ONE tight analysis on the frozen Papers-with-Code evaluation
dump (last public snapshot, mirrored on HuggingFace after the 2025 PWC
sunset): estimate the reporting-selection correlation rho with the classic
Heckman MLE (heckesel.selection, the A-E0-gated implementation).

Panel construction (documented):
- take a task's K most-reported datasets (accuracy-like metrics only,
  higher = better; modal metric name per dataset);
- models reporting on >= 2 of them form the panel rows; cells are
  standardized within dataset (z-scores among reported values);
- observable ability proxy for model m on dataset d: leave-one-out mean of
  m's z-scores on its OTHER reported datasets;
- outcome equation:  z_score ~ 1 + ability + dataset dummies
- selection equation: reported ~ 1 + ability + dataset dummies
- rho = correlation between the equations' errors: do models report on
  exactly the datasets where they do UNOBSERVABLY well?

Caveats (stated in the paper; this is a discussion vignette, not a
headline claim): NO exclusion restriction exists -- identification rides on
bivariate normality; scores are not iid across papers; the ability proxy
is itself noisy. rho > 0 is evidence consistent with selective reporting
on unobservables, not a causal estimate.

Run: python experiments/expA_e4.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_json, ROOT

from heckesel.selection import heckman_mle, heckman_two_step

GOOD_METRIC_WORDS = ("accuracy", "top 1", "top-1", "f1", "map", "ap50",
                     "psnr", "ssim", "bleu", "rouge", "miou", "dice",
                     "auc", "ap")
K_DATASETS = 8
MIN_REPORTED = 2
TASKS = ["Image Classification", "Semantic Segmentation",
         "Object Detection", "Machine Translation"]


def load_rows():
    with open(ROOT / "data" / "pwc_results.csv", newline="",
              encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            yield r


def numeric(v: str):
    v = v.strip().rstrip("%*")
    try:
        return float(v)
    except ValueError:
        return None


def build_panel(task: str):
    """-> (models, datasets, score matrix with NaN, reported mask)"""
    by_ds = defaultdict(list)
    for r in load_rows():
        if r["task_path"] != task:
            continue
        if not any(w in r["metric_name"].lower()
                   for w in GOOD_METRIC_WORDS):
            continue
        v = numeric(r["metric_value"])
        if v is None:
            continue
        by_ds[r["dataset"]].append((r["model_name"], r["metric_name"], v))

    top = sorted(by_ds, key=lambda d: -len(by_ds[d]))[:K_DATASETS]
    cells = {}
    for d in top:
        metric = Counter(m for _, m, _ in by_ds[d]).most_common(1)[0][0]
        for model, mname, v in by_ds[d]:
            if mname == metric:
                cells[(model, d)] = v  # last occurrence wins (dedup)

    models = sorted({m for m, d in cells})
    models = [m for m in models
              if sum((m, d) in cells for d in top) >= MIN_REPORTED]
    S = np.zeros((len(models), len(top)))
    Y = np.full((len(models), len(top)), np.nan)
    for i, m in enumerate(models):
        for j, d in enumerate(top):
            if (m, d) in cells:
                S[i, j] = 1.0
                Y[i, j] = cells[(m, d)]
    # z-score within dataset among reported
    for j in range(len(top)):
        col = Y[:, j]
        rep = ~np.isnan(col)
        if rep.sum() > 3:
            sd = np.nanstd(col) + 1e-9
            Y[:, j] = (col - np.nanmean(col)) / sd
    return models, top, Y, S


def heckman_panel(models, datasets, Y, S):
    n_m, n_d = Y.shape
    # ability proxy: leave-one-out mean z-score
    rows = []
    for i in range(n_m):
        rep = np.where(S[i] > 0.5)[0]
        for j in range(n_d):
            others = [Y[i, k] for k in rep if k != j
                      and not np.isnan(Y[i, k])]
            if len(others) == 0:
                continue
            ability = float(np.mean(others))
            rows.append((i, j, ability, S[i, j],
                         Y[i, j] if S[i, j] > 0.5 else np.nan))
    ability = np.array([r[2] for r in rows])
    s = np.array([r[3] for r in rows])
    y = np.array([r[4] for r in rows])
    dsj = np.array([r[1] for r in rows])
    D = np.zeros((len(rows), n_d - 1))
    for k in range(1, n_d):
        D[:, k - 1] = (dsj == k).astype(float)
    X = np.column_stack([np.ones(len(rows)), ability, D])

    ts = heckman_two_step(y, X, s, X)
    ml = heckman_mle(y, X, s, X, start=ts)
    se_rho = float("nan")
    cov = ml.extra.get("cov_theta")
    if cov is not None and np.all(np.isfinite(cov)):
        # delta method: rho = tanh(theta_last)
        var_ath = cov[-1, -1]
        se_rho = float((1 - ml.rho**2) * np.sqrt(max(var_ath, 0.0)))
    return {"n_cells": len(rows), "n_models": n_m,
            "n_datasets": n_d,
            "reported_frac": float(s.mean()),
            "rho_two_step": ts.rho, "rho_mle": ml.rho,
            "rho_mle_se": se_rho,
            "ability_coef_outcome": float(ml.beta[1]),
            "ability_coef_selection": float(ml.gamma[1]),
            "beta_lambda": ts.beta_lambda,
            "loglik": ml.loglik, "converged": ml.converged}


def heckman_pairwise(models, datasets, Y, S, anchor_j: int = 0,
                     max_targets: int = 5):
    """Cleaner two-dataset formulation: sample = models reporting the
    ANCHOR dataset; selection = also reports target; outcome = target
    z-score; regressor = anchor z-score. One Heckman per pair."""
    out = []
    rep_anchor = S[:, anchor_j] > 0.5
    za = Y[rep_anchor, anchor_j]
    order = np.argsort(-S.sum(axis=0))
    targets = [j for j in order if j != anchor_j][:max_targets]
    for j in targets:
        s = S[rep_anchor, j]
        y = np.where(s > 0.5, Y[rep_anchor, j], np.nan)
        if s.sum() < 12 or s.mean() > 0.95:
            continue
        X = np.column_stack([np.ones(len(za)), za])
        try:
            ts = heckman_two_step(y, X, s, X)
            ml = heckman_mle(y, X, s, X, start=ts)
            cov = ml.extra.get("cov_theta")
            se_rho = float((1 - ml.rho**2)
                           * np.sqrt(max(cov[-1, -1], 0.0))) \
                if cov is not None and np.all(np.isfinite(cov)) else None
            out.append({"anchor": datasets[anchor_j],
                        "target": datasets[j],
                        "n": int(len(za)), "n_reported": int(s.sum()),
                        "rho_mle": ml.rho, "rho_two_step": ts.rho,
                        "rho_mle_se": se_rho,
                        "boundary": bool(abs(ml.rho) > 0.99),
                        "converged": ml.converged})
        except Exception as e:
            out.append({"anchor": datasets[anchor_j],
                        "target": datasets[j], "error": repr(e)})
    return out


def main():
    out = {}
    for task in TASKS:
        try:
            models, datasets, Y, S = build_panel(task)
            if len(models) < 40:
                out[task] = {"skipped": f"panel too small ({len(models)})"}
                continue
            res = heckman_panel(models, datasets, Y, S)
            res["datasets"] = datasets
            res["boundary"] = bool(abs(res["rho_mle"]) > 0.99)
            res["pairwise"] = heckman_pairwise(models, datasets, Y, S)
            out[task] = res
            print(f"{task}: n_models={res['n_models']} "
                  f"cells={res['n_cells']} reported={res['reported_frac']:.2f} "
                  f"rho_mle={res['rho_mle']:+.3f} (se {res['rho_mle_se']:.3f})"
                  f"{' [BOUNDARY]' if res['boundary'] else ''}")
            for p in res["pairwise"]:
                if "rho_mle" in p:
                    print(f"   pair {p['anchor'][:18]:18s} -> "
                          f"{p['target'][:22]:22s} n={p['n']:3d} "
                          f"rep={p['n_reported']:3d} "
                          f"rho={p['rho_mle']:+.3f}"
                          f"{' [BOUNDARY]' if p['boundary'] else ''}")
        except Exception as e:  # vignette: report failures, don't die
            out[task] = {"error": repr(e)}
            print(task, "ERROR", repr(e))
    save_json("expA_e4.json", out)


if __name__ == "__main__":
    main()
