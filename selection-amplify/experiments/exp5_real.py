"""E5 -- REAL-CORPUS PILOT (stretch; limits-characterization only, K5-gated).

One curated-vs-raw tabular pair: California housing, 2 standardized features
(MedInc, Latitude), target = above-median value. A REAL covariate drives the
MNAR curation: s_beta = sigmoid(-beta * MedInc_std) censors high-income areas
(as if only cheaper regions were logged). The raw, unfiltered pool is the
reference cover -- an affordance that on a genuine corpus may not exist, so this
rung is reported honestly as "as good as the uncurated pool's coverage", NEVER a
clean synthetic-strength claim, and is explicitly NOT a single point of failure.

K5 COVER-BLIND CHECK: if the reference pool were co-censored on the target
region, the density-ratio classifier confidence would collapse to the class
prior there (s_hat ~ const, epistemic uncertainty saturated). We verify it does
NOT, and report the collar coverage; if it did, we would report only synthetic.

Emits results/exp5_real.json.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_california_housing

import common as C
from selamp.bridge import generate_labeled
from selamp.diffusion import Diffusion
from selamp.downstream import train_eval
from selamp.entropy import mutual_info_OX
from selamp.selection import SelectionEstimator
from selamp.validate import GateKDE, IndependentValidator

BETAS = [0.0, 2.0, 6.0]
SEEDS = [0, 1, 2]
SLICE_Q = 0.70                       # censored slice = top 30% by MedInc


def _load():
    d = fetch_california_housing()
    X = d.data[:, [0, 6]]            # MedInc, Latitude
    y = (d.target > np.median(d.target)).astype(int)
    mu, sd = X.mean(0), X.std(0)
    return (X - mu) / sd, y, mu, sd


class RealCorpora:
    def __init__(self, beta, seed):
        X, y, self.mu, self.sd = _load()
        rng = np.random.default_rng(1000 + seed)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]
        n = len(X)
        # split: pool for corpus, disjoint raw reference, disjoint test
        a, b = int(0.5 * n), int(0.75 * n)
        Xp, yp = X[:a], y[:a]
        self.X_ref = X[a:b]                        # raw, unlabeled reference
        self.X_test, self.y_test = X[b:], y[b:]
        phi = Xp[:, 0]                             # standardized MedInc
        s = 1.0 / (1.0 + np.exp(beta * phi))       # sigmoid(-beta*phi)
        keep = rng.random(len(Xp)) < s
        self.X_obs, self.y_obs = Xp[keep], yp[keep]
        self.beta, self.seed = beta, seed
        self.obs_frac = float(s.mean())
        self._slice_thr = np.quantile(X[:, 0], SLICE_Q)

    def s_true(self, X):
        return 1.0 / (1.0 + np.exp(self.beta * X[:, 0]))

    def slice_mask(self, X):
        return X[:, 0] > self._slice_thr


def one(beta, seed):
    c = RealCorpora(beta, seed)
    sel = SelectionEstimator(**C.SELECTOR_KW).fit(c.X_obs, c.X_ref, c.obs_frac,
                                                  seed=seed)
    dm = Diffusion(**C.DIFFUSION_KW).fit(c.X_obs, c.y_obs, seed=seed,
                                         **C.DIFFUSION_FIT_KW)
    gate = GateKDE(c.X_ref, bandwidth=0.30, seed=seed)
    val = IndependentValidator(c.X_ref)
    from selamp.bridge import RewardConfig
    from scipy.spatial import cKDTree
    d, _ = cKDTree(c.X_obs).query(c.X_obs, k=2)
    cfg = RewardConfig(gamma=C.OP["gamma"],
                       tau_log=gate.quantile_threshold(c.X_ref, C.OP["tau_q"]),
                       veto_log=gate.quantile_threshold(c.X_ref, C.OP["veto_q"]),
                       u_max=sel.uncertainty_quantile(c.X_obs, C.OP["u_q"]),
                       d_max=C.OP["dmax_factor"] * float(np.median(d[:, 1])))
    npc = C.N_SYNTH_PER_CLASS
    Xm, ym, _ = generate_labeled(dm, sel, gate, npc, cfg, seed=seed)
    Xd, yd, _ = generate_labeled(dm, sel, gate, npc, cfg, decoy="rotate", seed=seed)

    def aug(Xs, ys):
        return np.concatenate([c.X_obs, Xs]), np.concatenate([c.y_obs, ys])
    s_hat_obs = sel.s_hat(c.X_obs)
    sets = {
        "B0_obs": (c.X_obs, c.y_obs, None),
        "B3_reweight": (c.X_obs, c.y_obs, 1.0 / np.clip(s_hat_obs, 0.05, 1.0)),
        "method": (*aug(Xm, ym), None),
        "decoy_rotate": (*aug(Xd, yd), None),
    }
    rows = []
    for name, (Xtr, ytr, w) in sets.items():
        ev = train_eval(Xtr, ytr, c.X_test, c.y_test, c.slice_mask,
                        sample_weight=w, seed=seed)
        rows.append(dict(beta=beta, seed=seed, method=name, **ev))

    # K5 cover-blind: in the censored slice, does s_hat carry signal (D_ref
    # covers it) rather than collapsing to a constant / saturated uncertainty?
    sm = c.slice_mask(c.X_test)
    s_hat_slice = sel.s_hat(c.X_test[sm])
    unc_slice = sel.uncertainty(c.X_test[sm])
    unc_core = sel.uncertainty(c.X_test[~sm])
    Xbig, ybig, _, _ = _load()
    iox = mutual_info_OX(c.s_true(Xbig))
    diag = dict(beta=beta, seed=seed, method="_diag", iox=iox,
                cover_blind_std_shat=float(np.std(s_hat_slice)),
                unc_slice=float(np.mean(unc_slice)),
                unc_core=float(np.mean(unc_core)),
                method_reject=val.score(Xm)["reject_rate"],
                obs_frac=c.obs_frac, n_obs=int(len(c.X_obs)))
    rows.append(diag)
    return rows


def run():
    rows = []
    for beta in BETAS:
        for seed in SEEDS:
            rows += one(beta, seed)
        def m(name, key):
            return np.nanmean([r[key] for r in rows if r["beta"] == beta
                               and r["method"] == name])
        dg = [r for r in rows if r["beta"] == beta and r["method"] == "_diag"]
        print(f"beta={beta} IOX={np.mean([x['iox'] for x in dg]):.3f} | slice "
              f"B0={m('B0_obs','acc_slice'):.3f} B3={m('B3_reweight','acc_slice'):.3f} "
              f"method={m('method','acc_slice'):.3f} decoy={m('decoy_rotate','acc_slice'):.3f}"
              f" | cover_blind std_shat={np.mean([x['cover_blind_std_shat'] for x in dg]):.3f}")
    C.save("exp5_real.json", {"rows": rows, "stamp": C.stamp()})


if __name__ == "__main__":
    run()
