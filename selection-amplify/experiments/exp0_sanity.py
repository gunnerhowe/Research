"""E0 -- AFTERNOON SANITY (estimation only, the FIRST go/no-go).

Can a calibrated density-ratio deep ensemble recover a KNOWN 2D selector, and
does the selection-entropy axis computed from s_hat agree with analytic and move
with beta? The region-split gate (K3b fix) demands the recovery hold on the
COMPLEMENT, not just globally -- catching the silent false-GO where the pipeline
passes on the easy observed region and is pure extrapolation in the censored one.

No generation here. GATE (pre-registered, PLAN.md):
  global Spearman(s_hat,s_beta) > 0.7 AND collar-complement Spearman > 0.5,
  ECE < 0.10, and plug-in I(O;X) tracks analytic (corr > 0.95 across beta).
"""
from __future__ import annotations

import numpy as np

import common as C
from selamp import data
from selamp.entropy import d_comp, mutual_info_OX
from selamp.selection import SelectionEstimator
from selamp.stats import spearman, spearman_onesided_pos

TESTBEDS = ["two_moons", "eight_gaussians", "pinwheel"]
SEEDS = [0, 1, 2, 3, 4]


def one(testbed, beta, seed):
    c = data.make_corpora(testbed, beta, seed, C.N_POP, C.N_REF, C.N_TEST)
    sel = SelectionEstimator(**C.SELECTOR_KW).fit(
        c.X_obs, c.X_ref, c.obs_frac, seed=seed)

    X = c.X_test
    s_true = data.selection_prob(X, beta, testbed)
    s_hat = sel.s_hat(X)
    prox = sel.proximity(X)
    # proximity collar radius by the same rule as the bridge (median obs self-NN)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(c.X_obs).query(c.X_obs, k=2)
    d_max = C.OP["dmax_factor"] * float(np.median(d[:, 1]))

    comp = s_true < 0.5                       # censored-leaning complement
    collar = comp & (prox < d_max)            # recoverable collar-complement

    rho_g, _ = spearman(s_hat, s_true)
    rho_c, _ = spearman(s_hat[comp], s_true[comp]) if comp.sum() > 10 else (np.nan, np.nan)
    rho_k, pk = spearman_onesided_pos(s_hat[collar], s_true[collar]) \
        if collar.sum() > 10 else (np.nan, np.nan)

    # selection-entropy axis: analytic (from true s) vs plug-in (from s_hat),
    # both over the reference/population sample
    s_true_ref = data.selection_prob(c.X_ref, beta, testbed)
    s_hat_ref = sel.s_hat(c.X_ref)
    return {
        "testbed": testbed, "beta": beta, "seed": seed,
        "n_obs": int(len(c.X_obs)), "obs_frac": c.obs_frac,
        "spearman_global": rho_g,
        "spearman_complement": rho_c,
        "spearman_collar": rho_k, "p_collar": pk,
        "collar_frac": float(collar.mean()),
        "ece": sel.ece(),
        "IOX_analytic": mutual_info_OX(s_true_ref),
        "IOX_plugin": mutual_info_OX(s_hat_ref),
        "Dcomp_analytic": d_comp(s_true_ref),
        "Dcomp_plugin": d_comp(s_hat_ref),
    }


def run():
    rows = []
    for tb in TESTBEDS:
        for beta in C.BETAS:
            for seed in SEEDS:
                r = one(tb, beta, seed)
                rows.append(r)
            sub = [x for x in rows if x["testbed"] == tb and x["beta"] == beta]
            print(f"{tb:16s} b={beta:>3} | rho_g="
                  f"{np.nanmean([s['spearman_global'] for s in sub]):.3f} "
                  f"rho_comp={np.nanmean([s['spearman_complement'] for s in sub]):.3f} "
                  f"rho_collar={np.nanmean([s['spearman_collar'] for s in sub]):.3f} "
                  f"ece={np.nanmean([s['ece'] for s in sub]):.3f} "
                  f"IOX={np.nanmean([s['IOX_plugin'] for s in sub]):.3f}")
    C.save("exp0_sanity.json", {"rows": rows, "stamp": C.stamp()})


if __name__ == "__main__":
    run()
