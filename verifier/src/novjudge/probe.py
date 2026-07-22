"""E3 mechanism: is the judge's novelty-SIGNAL response carried by a linear,
DISSOCIABLE direction in its residual stream? (The response to Breaking-the-Mirror.)

- capture: one forward per item, grab the scoring-position residual at each layer.
- probe: supervised logistic probe G-high vs G-low, IN-DISTRIBUTION (stem-grouped
  CV) and OUT-OF-DOMAIN (train ml -> test econ and vice versa).
- direction: difference-of-means G-direction per layer (reused by the E4 steer).
- dissociation: cosine of the G-direction against other candidate directions
  (e.g. a self-preference or uncertainty direction) must be low for the claim
  that this is a novelty-signal direction, not a rebadged known bias.

No fitting here presupposes E0's outcome; this runs only if E0 shows a G effect.
"""

from __future__ import annotations

import numpy as np


def capture_dataset(judge, rows: list[dict], layers: list[int]):
    """Return {layer: X[n_items, hidden]} plus aligned label arrays.

    rows are the scoring_rows (stem, cell, s, g, text, prior_work). Uses the
    frozen pointwise rubric so the residual is captured at the exact scoring
    position the judge would score from.
    """
    from novjudge.judge_local import capture_residuals_multi, expected_score
    from novjudge.rubric import pointwise_messages

    feats = {L: [] for L in layers}
    g, s, stem, dom, y = [], [], [], [], []
    for i, r in enumerate(rows):
        msgs = pointwise_messages(r["text"], r["prior_work"])
        with capture_residuals_multi(judge, layers) as store:
            yi = expected_score(judge, msgs)
        for L in layers:
            feats[L].append(store[L].numpy())
        g.append(r["g"]); s.append(r["s"]); stem.append(r["stem"])
        dom.append(r["domain"]); y.append(yi)
        if (i + 1) % 100 == 0:
            print(f"  capture {i + 1}/{len(rows)}")
    X = {L: np.asarray(feats[L]) for L in layers}
    return X, {"g": np.array(g), "s": np.array(s), "stem": np.array(stem),
               "domain": np.array(dom), "y": np.array(y)}


def _auroc(model, Xte, yte):
    from sklearn.metrics import roc_auc_score
    p = model.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, p))


def probe_in_distribution(X, y, stem, n_splits: int = 5, seed: int = 0) -> dict:
    """Stem-grouped CV AUROC for G-high vs G-low (no stem spans train and test)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    gkf = GroupKFold(n_splits=n_splits)
    aucs = []
    for tr, te in gkf.split(X, y, groups=stem):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0))
        m.fit(X[tr], y[tr])
        aucs.append(_auroc(m, X[te], y[te]))
    return {"auroc_mean": float(np.mean(aucs)), "auroc_std": float(np.std(aucs)),
            "folds": aucs}


def probe_ood_domain(X, y, domain, train_dom: str, test_dom: str) -> dict:
    """Train on one domain, test on the other (OOD generalization)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    tr = domain == train_dom
    te = domain == test_dom
    if tr.sum() < 10 or te.sum() < 10:
        return {"auroc": float("nan"), "n_train": int(tr.sum()), "n_test": int(te.sum())}
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    m.fit(X[tr], y[tr])
    return {"auroc": _auroc(m, X[te], y[te]),
            "n_train": int(tr.sum()), "n_test": int(te.sum())}


def diff_of_means_direction(X, y) -> np.ndarray:
    """Normalized (mean_{G=1} - mean_{G=0}) residual — the signal direction the
    E4 steer projects out."""
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
