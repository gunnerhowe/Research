"""A-E2/E3: semi-real tabular experiments with INDUCED MNAR selection.

Real regression datasets (California housing; UCI wine quality red), with
selection induced by the documented latent-correlated rule
(heckesel.synth.induce_mnar_selection): selection against high-predicted-y
regions, correlated with the outcome residual (strength rho), instrument z.
Ground truth for evaluation exists for ALL rows because we induced the
selection ourselves; test rows are held out before selection.

Methods: the full A-E3 baseline suite + a SKYLINE deep ensemble trained on
the complete (pre-selection) pool outcomes -- an oracle upper reference.

Headline: PI coverage in selected-against regions (prop_x <= 0.3),
rho = 0.8, 8 seeds; rho in {0, 0.5} at 3 seeds (house rule 3).

Run: python experiments/expA_e2.py [--smoke]
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_json, Timer, ROOT

from heckesel.synth import induce_mnar_selection
from heckesel.metrics import evaluate_predictive
from heckesel.deep import HeckmanEnsemble, HeckmanTwoStepEnsemble
from heckesel.uq import (DeepEnsembleUQ, IWDeepEnsembleUQ, MCDropoutUQ, GPUQ,
                         BlindTwoHeadUQ)

RHOS_SEEDS = [(0.0, 3), (0.5, 3), (0.8, 8)]
ALPHA = 1.0
TARGET_FRAC = 0.5


def load_california(n_pool=6000, n_test=4000, seed=0):
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing(data_home=str(ROOT / "data" / "sk"))
    x, y = d.data, d.target
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    return (x[idx[:n_pool]], y[idx[:n_pool]],
            x[idx[n_pool:n_pool + n_test]], y[idx[n_pool:n_pool + n_test]])


def load_wine(seed=0):
    path = ROOT / "data" / "winequality-red.csv"
    if not path.exists():
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "wine-quality/winequality-red.csv")
        path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    raw = np.genfromtxt(path, delimiter=";", skip_header=1)
    x, y = raw[:, :-1], raw[:, -1]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_pool = 1100
    return (x[idx[:n_pool]], y[idx[:n_pool]],
            x[idx[n_pool:]], y[idx[n_pool:]])


DATASETS = {"california": load_california, "wine": load_wine}


def standardize(x_pool, y_pool, x_test, y_test):
    mx, sx = x_pool.mean(0), x_pool.std(0) + 1e-9
    my, sy = y_pool.mean(), y_pool.std() + 1e-9
    return ((x_pool - mx) / sx, (y_pool - my) / sy,
            (x_test - mx) / sx, (y_test - my) / sy)


def run_cell(ds: str, rho: float, seed: int, epochs: int, k: int):
    x_pool, y_pool, x_test, y_test = DATASETS[ds](seed=seed)
    x_pool, y_pool, x_test, y_test = standardize(x_pool, y_pool, x_test,
                                                 y_test)
    s, z, prop, prop_x = induce_mnar_selection(
        x_pool, y_pool, rho, alpha=ALPHA, target_frac=TARGET_FRAC,
        seed=seed)
    # region definition for TEST rows: same documented rule applied with the
    # pool's linear fit (fresh z for test rows; prop_x needs only x)
    n_all = len(x_pool)
    X1 = np.column_stack([np.ones(n_all), x_pool])
    coef, *_ = np.linalg.lstsq(X1, y_pool, rcond=None)
    m_pool = X1 @ coef
    mm, ms = m_pool.mean(), m_pool.std() + 1e-12
    from scipy.special import ndtr
    m_test = (np.column_stack([np.ones(len(x_test)), x_test]) @ coef
              - mm) / ms
    idx_noise = -1.0 * m_pool  # for threshold c: reproduce rule's quantile
    # recompute c exactly as the rule did
    rng = np.random.default_rng(seed)
    r = y_pool - X1 @ coef
    r = (r - r.mean()) / (r.std() + 1e-12)
    z_rule = rng.normal(size=n_all)
    v = rng.normal(size=n_all)
    noise = rho * r + np.sqrt(max(1 - rho**2, 0.0)) * v
    m_std = (m_pool - mm) / ms
    idx_full = -m_std + ALPHA * z_rule + noise
    c = np.quantile(idx_full, 1.0 - TARGET_FRAC)
    prop_x_test = ndtr((-m_test - c) / np.sqrt(1.0 + ALPHA**2))

    sel = s > 0.5
    y_obs = np.where(sel, y_pool, np.nan)
    w_all = np.column_stack([x_pool, z])

    fits = {}
    fits["deep_ensemble"] = DeepEnsembleUQ.fit(x_pool[sel], y_pool[sel],
                                               k=k, seed=seed, epochs=epochs)
    fits["mc_dropout"] = MCDropoutUQ.fit(x_pool[sel], y_pool[sel],
                                         seed=seed, epochs=epochs)
    fits["gp"] = GPUQ.fit(x_pool[sel], y_pool[sel], seed=seed)
    fits["iw_oracle_ensemble"] = IWDeepEnsembleUQ.fit_iw(
        x_pool[sel], y_pool[sel], prop[sel], k=k, seed=seed, epochs=epochs)
    fits["blind_two_head"] = BlindTwoHeadUQ.fit(
        x_pool, w_all, y_obs, s, k=k, seed=seed, epochs=epochs)
    fits["heckman_ens"] = HeckmanEnsemble.fit(
        x_pool, w_all, y_obs, s, k=k, seed=seed, epochs=epochs)
    fits["heckman_2s_ens"] = HeckmanTwoStepEnsemble.fit(
        x_pool, w_all, y_obs, s, k=k, seed=seed, epochs=epochs)
    fits["skyline_ensemble"] = DeepEnsembleUQ.fit(x_pool, y_pool, k=k,
                                                  seed=seed, epochs=epochs)

    rows = []
    for name, model in fits.items():
        mu, var = model.predict(x_test)
        m = evaluate_predictive(y_test, mu, var, prop_x_test)
        row = {"dataset": ds, "rho": rho, "seed": seed, "method": name,
               "selected_frac": float(s.mean()), **m}
        if name == "heckman_ens":
            row["rho_hat"] = model.rho
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    grid = RHOS_SEEDS
    datasets = list(DATASETS)
    if args.smoke:
        grid = [(0.8, 1)]
        datasets = ["wine"]
        args.epochs = 600

    rows = []
    for ds in datasets:
        for rho, n_seeds in grid:
            for seed in range(n_seeds):
                with Timer(f"{ds} rho={rho} seed={seed}"):
                    rows += run_cell(ds, rho, seed, args.epochs, args.k)
    save_json("expA_e2_smoke.json" if args.smoke else "expA_e2.json", rows)

    print("\n===== coverage@90 selected-against (rho=0.8) =====")
    for name in ["deep_ensemble", "iw_oracle_ensemble", "blind_two_head",
                 "heckman_ens", "heckman_2s_ens", "skyline_ensemble"]:
        v = [r["picp90_against"] for r in rows
             if r["method"] == name and r["rho"] == 0.8
             and "picp90_against" in r]
        if v:
            print(f"{name:20s} {np.mean(v):.3f} +- {np.std(v):.3f}")


if __name__ == "__main__":
    main()
