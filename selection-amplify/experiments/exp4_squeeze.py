"""E4 -- SCOPE + ROBUSTNESS.

(A) THE CENTRAL SQUEEZE: (method - B3 reweighting) accuracy against
    distance-from-observed-support and against s_hat epistemic uncertainty --
    the thesis that generation buys a BOUNDED collar of advantage beyond
    reweighting, width set by s_hat extrapolation reliability. Per-test-point
    correctness is logged so make_figures can bin it.
(B) K4 FOREIGN-STRUCTURE PROBE: a disconnected, non-recombinable satellite in
    the censored region the method MUST fail to recover (positive control for
    the interpolative-only scope claim).
(C) ROBUSTNESS: inject noise into s_hat (graceful degradation); ablate gamma and
    the guidance-norm cap.

Emits results/exp4_squeeze.json.
"""
from __future__ import annotations

import numpy as np
import torch

import common as C
from selamp import data
from selamp.bridge import RewardConfig, generate_labeled
from selamp.downstream import train_eval

SQUEEZE_BETAS = [4.0, 8.0]
SEEDS = [0, 1, 2, 3, 4]
MAX_PTS = 2500                    # per-seed test points logged for the squeeze


class NoisySelector:
    """Wrap a fitted selector, adding a smooth spatial noise field to s_hat to
    model an unreliable estimator (differentiable, structured)."""

    def __init__(self, sel, eta, seed):
        self.sel = sel
        self.eta = eta
        rng = np.random.default_rng(seed)
        self.w = torch.tensor(rng.normal(0, 1.5, (3, 2)), dtype=torch.float32,
                              device=sel.device)
        self.b = torch.tensor(rng.uniform(0, 6.28, 3), dtype=torch.float32,
                              device=sel.device)
        self._mu = sel._mu

    def s_hat_torch(self, X):
        s = self.sel.s_hat_torch(X)
        field = torch.sin(X @ self.w.T + self.b).mean(1)
        return (s + self.eta * field).clamp(1e-4, 1 - 1e-4)

    def uncertainty(self, X):
        return self.sel.uncertainty(X)

    def proximity(self, X):
        return self.sel.proximity(X)


def _squeeze_records(st):
    c, sel = st.c, st.sel
    Xm, ym, _ = generate_labeled(st.dm, sel, st.gate, C.N_SYNTH_PER_CLASS,
                                 st.cfg, seed=st.seed)
    s_hat_obs = sel.s_hat(c.X_obs)
    Xtr_m = np.concatenate([c.X_obs, Xm])
    ytr_m = np.concatenate([c.y_obs, ym])
    from sklearn.neural_network import MLPClassifier

    def fit_pred(Xtr, ytr, w=None):
        clf = MLPClassifier(hidden_layer_sizes=(64, 64), alpha=1e-3,
                            max_iter=800, random_state=st.seed)
        if w is not None:
            rng = np.random.default_rng(st.seed)
            p = np.clip(w, 1e-6, None); p = p / p.sum()
            idx = rng.choice(len(Xtr), len(Xtr), replace=True, p=p)
            Xtr, ytr = Xtr[idx], ytr[idx]
        clf.fit(Xtr, ytr)
        return clf.predict(c.X_test)

    pm = fit_pred(Xtr_m, ytr_m)
    pb3 = fit_pred(c.X_obs, c.y_obs, w=1.0 / np.clip(s_hat_obs, 0.05, 1.0))
    pb0 = fit_pred(c.X_obs, c.y_obs)

    prox = sel.proximity(c.X_test)
    unc = sel.uncertainty(c.X_test)
    s_true = data.selection_prob(c.X_test, st.beta, st.testbed)
    n = min(MAX_PTS, len(c.X_test))
    sl = slice(0, n)
    return dict(
        beta=st.beta, seed=st.seed,
        prox=prox[sl], unc=unc[sl], s_true=s_true[sl],
        correct_method=(pm == c.y_test).astype(int)[sl],
        correct_b3=(pb3 == c.y_test).astype(int)[sl],
        correct_b0=(pb0 == c.y_test).astype(int)[sl],
    )


def _foreign_probe(seed, beta=4.0):
    st = C.Stack("two_moons_foreign", beta, seed)
    c = st.c
    Xm, ym, _ = generate_labeled(st.dm, st.sel, st.gate, C.N_SYNTH_PER_CLASS,
                                 st.cfg, seed=seed)
    fmask = data.foreign_mask(c.X_test)
    collar = c.slice_mask(c.X_test) & ~fmask
    ev0 = train_eval(c.X_obs, c.y_obs, c.X_test, c.y_test, c.slice_mask, seed=seed)
    Xa, ya = np.concatenate([c.X_obs, Xm]), np.concatenate([c.y_obs, ym])
    evm = train_eval(Xa, ya, c.X_test, c.y_test, c.slice_mask, seed=seed)

    from sklearn.neural_network import MLPClassifier

    def acc_on(mask, Xtr, ytr):
        clf = MLPClassifier(hidden_layer_sizes=(64, 64), alpha=1e-3,
                            max_iter=800, random_state=seed).fit(Xtr, ytr)
        p = clf.predict(c.X_test[mask])
        return float((p == c.y_test[mask]).mean())
    # how much of the method's synthesis even lands in the foreign region?
    synth_in_foreign = float(data.foreign_mask(Xm).mean())
    return dict(
        beta=beta, seed=seed,
        foreign_acc_b0=acc_on(fmask, c.X_obs, c.y_obs),
        foreign_acc_method=acc_on(fmask, Xa, ya),
        collar_acc_b0=acc_on(collar, c.X_obs, c.y_obs),
        collar_acc_method=acc_on(collar, Xa, ya),
        synth_in_foreign=synth_in_foreign,
        n_foreign=int(fmask.sum()),
    )


def _robustness(seed, beta=4.0):
    st = C.Stack(C.PRIMARY, beta, seed)
    c = st.c
    out = {"beta": beta, "seed": seed, "noise": [], "gamma": [], "cap": []}

    def slice_gain(Xs, ys):
        ev0 = train_eval(c.X_obs, c.y_obs, c.X_test, c.y_test, c.slice_mask, seed=seed)
        Xa, ya = np.concatenate([c.X_obs, Xs]), np.concatenate([c.y_obs, ys])
        evm = train_eval(Xa, ya, c.X_test, c.y_test, c.slice_mask, seed=seed)
        return evm["acc_slice"] - ev0["acc_slice"]

    for eta in (0.0, 0.1, 0.2, 0.4):
        nsel = NoisySelector(st.sel, eta, seed) if eta > 0 else st.sel
        Xs, ys, _ = generate_labeled(st.dm, nsel, st.gate, C.N_SYNTH_PER_CLASS,
                                     st.cfg, seed=seed)
        out["noise"].append({"eta": eta, "slice_gain": slice_gain(Xs, ys)})
    for gamma in (0.0, 3.0, 6.0, 10.0):
        cfg = RewardConfig(**{**st.cfg.__dict__, "gamma": gamma})
        Xs, ys, _ = generate_labeled(st.dm, st.sel, st.gate,
                                     C.N_SYNTH_PER_CLASS, cfg, seed=seed)
        out["gamma"].append({"gamma": gamma, "slice_gain": slice_gain(Xs, ys)})
    for cap in (1.0, 1e9):
        cfg = RewardConfig(**{**st.cfg.__dict__, "cap_ratio": cap})
        Xs, ys, dg = generate_labeled(st.dm, st.sel, st.gate,
                                      C.N_SYNTH_PER_CLASS, cfg, seed=seed)
        rej = st.val.score(Xs)["reject_rate"]
        out["cap"].append({"cap": cap, "slice_gain": slice_gain(Xs, ys),
                           "reject": rej})
    return out


def run():
    squeeze, foreign, robust = [], [], []
    for beta in SQUEEZE_BETAS:
        for seed in SEEDS:
            st = C.Stack(C.PRIMARY, beta, seed)
            squeeze.append(_squeeze_records(st))
        print(f"[squeeze] beta={beta} done")
    for seed in SEEDS:
        foreign.append(_foreign_probe(seed))
    fa = np.mean([f["foreign_acc_method"] - f["foreign_acc_b0"] for f in foreign])
    ca = np.mean([f["collar_acc_method"] - f["collar_acc_b0"] for f in foreign])
    print(f"[K4] foreign gain={fa:+.4f} (must ~0) | collar gain={ca:+.4f} "
          f"| synth_in_foreign={np.mean([f['synth_in_foreign'] for f in foreign]):.3f}")
    for seed in SEEDS:
        robust.append(_robustness(seed))
    C.save("exp4_squeeze.json", {"squeeze": squeeze, "foreign": foreign,
                                 "robust": robust, "stamp": C.stamp()})


if __name__ == "__main__":
    run()
