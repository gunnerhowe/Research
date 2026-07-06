"""Downstream evaluation: does manufacturing the censored complement fix the
classifier where the corpus is blind?

A flexible classifier is trained on the curated corpus (optionally augmented or
reweighted) and evaluated on a frozen full-population test set and on its
CENSORED SLICE (the high-phi tail the selector progressively removes). The
paper's quantity of interest is the per-seed gain over the D_obs-only floor,
and above all the METHOD-MINUS-DECOY gap on the censored slice.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier


def train_eval(X_tr, y_tr, X_te, y_te, slice_mask, sample_weight=None,
               seed=0, hidden=(64, 64), alpha=1e-3, max_iter=800):
    clf = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha,
                        max_iter=max_iter, random_state=seed)
    if sample_weight is not None:
        # sklearn MLP has no sample_weight; emulate by weighted resampling
        rng = np.random.default_rng(seed)
        w = np.clip(sample_weight, 1e-6, None)
        w = w / w.sum()
        idx = rng.choice(len(X_tr), size=len(X_tr), replace=True, p=w)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc = (pred == y_te).astype(float)
    sm = slice_mask(X_te)
    return {
        "acc_full": float(acc.mean()),
        "acc_slice": float(acc[sm].mean()) if sm.any() else float("nan"),
        "acc_nonslice": float(acc[~sm].mean()) if (~sm).any() else float("nan"),
        "n_slice": int(sm.sum()),
    }


def smote_low_density(X_obs, y_obs, s_hat_obs, n_synth, k=5, seed=0,
                      low_frac=0.4):
    """B1: SMOTE-style oversampling of the lowest-selection (tail) observed
    points, class-conditionally."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for c in np.unique(y_obs):
        Xc = X_obs[y_obs == c]
        sc = s_hat_obs[y_obs == c]
        if len(Xc) < k + 1:
            continue
        thr = np.quantile(sc, low_frac)
        low = Xc[sc <= thr]
        if len(low) < 2:
            low = Xc
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(low))).fit(low)
        n_c = n_synth // len(np.unique(y_obs))
        base = low[rng.integers(0, len(low), n_c)]
        _, nbr = nn.kneighbors(base)
        pick = low[nbr[np.arange(n_c), rng.integers(1, nn.n_neighbors, n_c)]]
        lam = rng.random((n_c, 1))
        Xs.append(base + lam * (pick - base))
        ys.append(np.full(n_c, c))
    return np.concatenate(Xs), np.concatenate(ys).astype(int)


def match_temperature(diffusion, selector, gate_kde, guided_X, n_per_class,
                      seed, cand=(1.0, 1.3, 1.6, 2.0, 2.5)):
    """B2-diversity-matched: pick the unconditional-sampling temperature whose
    generated spread (mean nearest-neighbour distance) best matches the guided
    sampler's, so B2 is not beaten merely on diversity."""
    target = _mean_nn(guided_X)
    best, best_gap, best_T = None, np.inf, 1.0
    for T in cand:
        Xs = [diffusion.sample(n_per_class, c, temperature=T, seed=seed + c)
              for c in range(diffusion.n_classes)]
        X = np.concatenate(Xs)
        gap = abs(_mean_nn(X) - target)
        if gap < best_gap:
            best, best_gap, best_T = X, gap, T
    y = np.concatenate([np.full(n_per_class, c)
                        for c in range(diffusion.n_classes)]).astype(int)
    return best, y, best_T, target


def _mean_nn(X, k=1):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    d, _ = nn.kneighbors(X)
    return float(d[:, 1:].mean())
