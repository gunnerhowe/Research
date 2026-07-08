"""Statistics: pooled AUROC/AP with cluster (per-problem) bootstrap CIs, paired tests."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

N_BOOT = 10_000


def pooled_auroc_ci(labels_by_problem: list[np.ndarray],
                    scores_by_problem: list[np.ndarray],
                    n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """AUROC over pooled steps; 95% percentile bootstrap resampling PROBLEMS."""
    y = np.concatenate(labels_by_problem)
    s = np.concatenate(scores_by_problem)
    point_auroc = float(roc_auc_score(y, s))
    point_ap = float(average_precision_score(y, s))
    rng = np.random.default_rng(seed)
    n = len(labels_by_problem)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = np.concatenate([labels_by_problem[i] for i in idx])
        sb = np.concatenate([scores_by_problem[i] for i in idx])
        if yb.min() == yb.max():
            continue
        boots.append(roc_auc_score(yb, sb))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "auroc": point_auroc,
        "auroc_ci": [float(lo), float(hi)],
        "ap": point_ap,
        "n_steps": int(len(y)),
        "n_pos": int(y.sum()),
        "n_problems": n,
    }


def spearman_ci(x_by_problem: list[np.ndarray], y_by_problem: list[np.ndarray],
                n_boot: int = N_BOOT, seed: int = 0) -> dict:
    from scipy.stats import spearmanr

    x = np.concatenate(x_by_problem)
    y = np.concatenate(y_by_problem)
    r, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x_by_problem)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = np.concatenate([x_by_problem[i] for i in idx])
        yb = np.concatenate([y_by_problem[i] for i in idx])
        boots.append(spearmanr(xb, yb).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"spearman": float(r), "p": float(p), "ci": [float(lo), float(hi)]}


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy.stats import wilcoxon

    a, b = np.asarray(a, float), np.asarray(b, float)
    stat, p = wilcoxon(a, b)
    return {
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "median_diff": float(np.median(a - b)),
        "p": float(p),
        "n": int(len(a)),
    }
