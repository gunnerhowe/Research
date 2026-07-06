"""E3 -- PERCEPTUAL TESTBED-B (MNIST subpopulation censoring).

Confirms the mechanism survives structured data, not just 2D density. Binary
MNIST {3,8}; the censored subpopulation is the naturally THICK-STROKE digits
(phi = total ink), s_beta = sigmoid(-beta * ink_std). The whole
LOCATE->GENERATE->characterize pipeline runs in a PCA(16) latent of the images
(a genuinely higher-dim learned manifold); synthesized latents are decoded back
to pixels for the visual off-manifold check. The matched-strength MISDIRECTED
control is the identical bridge with the selector composed with a fixed random
ORTHOGONAL rotation of the latent (the high-dim analog of the 2D 90-degree
decoy).

Three curves + independent PCA-space density/NN validators. Emits
results/exp3_mnist.json (+ a decoded-sample grid for the paper figure).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from torchvision import datasets

import common as C
from selamp.bridge import RewardConfig, generate_labeled
from selamp.diffusion import Diffusion
from selamp.entropy import mutual_info_OX
from selamp.selection import SelectionEstimator
from selamp.validate import GateKDE, IndependentValidator

BETAS = [0.0, 2.0, 8.0]
SEEDS = [0, 1, 2]
LATENT = 16
NPC = 600
DIGITS = (3, 8)
DATA_ROOT = str(C.ROOT / "data")


def _load_mnist():
    ds = datasets.MNIST(DATA_ROOT, train=True, download=True)
    X = ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y = ds.targets.numpy()
    m = (y == DIGITS[0]) | (y == DIGITS[1])
    X = X[m]
    ylab = (y[m] == DIGITS[1]).astype(int)
    ink = X.sum(1)
    ink = (ink - ink.mean()) / ink.std()               # phi = standardized ink
    return X, ylab, ink


class RotatedSelector:
    """Matched-strength misdirected control: the fitted selector composed with a
    fixed random ORTHOGONAL rotation of the latent (points guidance at a rotated,
    wrong region at identical sharpness). The high-dim analog of the 2D decoy."""

    def __init__(self, sel, seed, d):
        self.sel = sel
        self._mu = sel._mu
        g = torch.Generator().manual_seed(seed)
        Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
        self.Q = Q.to(sel.device)
        mu = torch.tensor(sel._mu, dtype=torch.float32, device=sel.device)
        self.mu_t = mu

    def _rot(self, X):
        return (X - self.mu_t) @ self.Q.T + self.mu_t

    def s_hat_torch(self, X):
        return self.sel.s_hat_torch(self._rot(X))

    def uncertainty(self, X):
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.sel.device)
        return self.sel.uncertainty(self._rot(Xt).detach().cpu().numpy())

    def proximity(self, X):
        return self.sel.proximity(X)


def one(beta, seed, save_grid=False):
    Xpix, ylab, ink = _load_mnist()
    rng = np.random.default_rng(1000 + seed)
    idx = rng.permutation(len(Xpix))
    Xpix, ylab, ink = Xpix[idx], ylab[idx], ink[idx]
    n = len(Xpix)
    a, b = int(0.45 * n), int(0.7 * n)
    # PCA fit on the raw reference pool (unlabeled full-support cover)
    pca = PCA(n_components=LATENT, random_state=seed).fit(Xpix[a:b])
    Z = pca.transform(Xpix).astype(np.float32)

    s = 1.0 / (1.0 + np.exp(beta * ink))                # sigmoid(-beta*ink)
    keep = rng.random(n) < s
    obs = np.zeros(n, bool); obs[:a] = keep[:a]
    Z_obs, y_obs = Z[obs], ylab[obs]
    Z_ref = Z[a:b]
    Z_test, y_test, ink_test = Z[b:], ylab[b:], ink[b:]
    slice_mask_arr = ink_test > 0.5                      # thick-stroke slice
    def slice_mask(Zx):                                  # by nearest-in-test ink
        return None                                      # (we score the array directly)
    obs_frac = float(s[:a].mean())

    sel = SelectionEstimator(**C.SELECTOR_KW).fit(Z_obs, Z_ref, obs_frac, seed=seed)
    dm = Diffusion(**C.DIFFUSION_KW, d=LATENT).fit(Z_obs, y_obs, seed=seed,
                                                   **C.DIFFUSION_FIT_KW)
    gate = GateKDE(Z_ref, bandwidth=0.6, seed=seed)
    val = IndependentValidator(Z[b:], bandwidth=0.8)
    from scipy.spatial import cKDTree
    d2, _ = cKDTree(Z_obs).query(Z_obs, k=2)
    cfg = RewardConfig(gamma=C.OP["gamma"],
                       tau_log=gate.quantile_threshold(Z_ref, C.OP["tau_q"]),
                       veto_log=gate.quantile_threshold(Z_ref, C.OP["veto_q"]),
                       u_max=sel.uncertainty_quantile(Z_obs, C.OP["u_q"]),
                       d_max=C.OP["dmax_factor"] * float(np.median(d2[:, 1])))

    Zm, ym, dgm = generate_labeled(dm, sel, gate, NPC, cfg, seed=seed)
    dec = RotatedSelector(sel, seed, LATENT)
    Zd, yd, _ = generate_labeled(dm, dec, gate, NPC, cfg, seed=seed)
    Zb2 = np.concatenate([dm.sample(NPC, k, seed=seed + k) for k in range(2)])
    yb2 = np.concatenate([np.full(NPC, k) for k in range(2)]).astype(int)

    def clf_acc(Ztr, ytr, w=None):
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                            max_iter=600, random_state=seed)
        if w is not None:
            p = np.clip(w, 1e-6, None); p = p / p.sum()
            ii = rng.choice(len(Ztr), len(Ztr), replace=True, p=p)
            Ztr, ytr = Ztr[ii], ytr[ii]
        clf.fit(Ztr, ytr)
        pred = clf.predict(Z_test)
        acc = (pred == y_test).astype(float)
        return float(acc.mean()), float(acc[slice_mask_arr].mean())

    s_hat_obs = sel.s_hat(Z_obs)
    def aug(Zs, ys): return np.concatenate([Z_obs, Zs]), np.concatenate([y_obs, ys])
    res = {}
    res["B0_obs"] = clf_acc(Z_obs, y_obs)
    res["B3_reweight"] = clf_acc(Z_obs, y_obs, w=1.0 / np.clip(s_hat_obs, 0.05, 1.0))
    res["B2_uncond"] = clf_acc(*aug(Zb2, yb2))
    res["method"] = clf_acc(*aug(Zm, ym))
    res["decoy_rotate"] = clf_acc(*aug(Zd, yd))

    Xbig, _, ink_big = _load_mnist()
    iox = mutual_info_OX(1.0 / (1.0 + np.exp(beta * ink_big)))
    row = dict(beta=beta, seed=seed, iox=iox, obs_frac=obs_frac,
               n_obs=int(len(Z_obs)),
               acc={k: v[0] for k, v in res.items()},
               acc_slice={k: v[1] for k, v in res.items()},
               method_reject=val.score(Zm)["reject_rate"],
               decoy_reject=val.score(Zd)["reject_rate"],
               b2_reject=val.score(Zb2)["reject_rate"],
               method_guide_norm=dgm[0].mean_guide_norm,
               method_base_norm=dgm[0].mean_base_norm)

    if save_grid:                                        # decode for visual check
        grid = pca.inverse_transform(Zm[:64]).reshape(-1, 28, 28)
        np.save(C.RESULTS / "exp3_sample_grid.npy", grid)
        np.save(C.RESULTS / "exp3_obs_grid.npy",
                Xpix[obs][:64].reshape(-1, 28, 28))
    return row


def run():
    rows = []
    for beta in BETAS:
        for seed in SEEDS:
            rows.append(one(beta, seed, save_grid=(beta == 8.0 and seed == 0)))
        def m(name, key):
            return np.nanmean([r[key][name] for r in rows if r["beta"] == beta])
        io = np.mean([r["iox"] for r in rows if r["beta"] == beta])
        print(f"beta={beta} IOX={io:.3f} | slice B0={m('B0_obs','acc_slice'):.3f} "
              f"method={m('method','acc_slice'):.3f} decoy={m('decoy_rotate','acc_slice'):.3f} "
              f"B3={m('B3_reweight','acc_slice'):.3f} | "
              f"gap={m('method','acc_slice')-m('decoy_rotate','acc_slice'):+.4f} "
              f"| method_reject={np.mean([r['method_reject'] for r in rows if r['beta']==beta]):.3f}")
    C.save("exp3_mnist.json", {"rows": rows, "stamp": C.stamp(),
                               "digits": DIGITS, "latent": LATENT})


if __name__ == "__main__":
    run()
