"""From raw run records to fitted selection models and paper numbers.

Variable construction (all deterministic given the committed raw JSONL):
- V:      verbalization indicator from the lexical detector (hints.verbalized)
          applied to the reasoning portion of the hinted generation.
- R_TE:   total-effect latent reliance: restricted log-odds of the hinted
          letter at the answer position, hinted run minus unhinted run
          (input-level do(hint) contrast; includes CoT-mediated paths).
- R_NDE:  hint-excision splice: hinted run minus (unhinted prompt + hinted
          CoT). The direct, not-through-CoT-text path (a la 2512.23032).
- R_pre:  pre-verbalization commitment shift: mean upper-half-layer logit-lens
          log-odds difference (hinted vs unhinted prompt, no CoT, instrument
          sentence excluded by construction).
- followed / flip: behavioral hint adoption (answer equals hinted letter;
  flip additionally requires the unhinted answer to differ).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from . import selection as sel
from .hints import LETTERS, parse_answer, split_cot, verbalized


# ----------------------------------------------------------------- loading


def load_rows(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset="qid", keep="last").reset_index(drop=True)


def _rlogit(lp: np.ndarray, idx: int) -> float:
    """Restricted log-odds of letter idx from raw letter log-probs."""
    x = np.asarray(lp, dtype=float)
    x = x - np.logaddexp.reduce(x)
    p = np.exp(x[idx])
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


def _lens_logodds(lens: np.ndarray, idx: int) -> np.ndarray:
    """Per-layer restricted log-odds of letter idx from lens probabilities."""
    p = np.clip(np.asarray(lens, dtype=float)[:, idx], 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def augment(df: pd.DataFrame) -> pd.DataFrame:
    """Attach V, reliance measures, answers, and covariates."""
    out = df.copy()
    h_idx = out["hint_letter"].map(lambda L: LETTERS.index(L)).values

    cots = [split_cot(t) for t in out["gen_hinted"]]
    out["V"] = [int(verbalized(c, ht))
                for c, ht in zip(cots, out["hint_type"])]
    out["cot_len"] = [len(c) for c in cots]

    out["ans_h"] = [parse_answer(t) for t in out["gen_hinted"]]
    out["ans_u"] = [parse_answer(t) for t in out["gen_unhinted"]]

    if "lp_hh" in out.columns:
        r_te, r_nde = [], []
        for j in range(len(out)):
            hh = _rlogit(out["lp_hh"].iloc[j], h_idx[j])
            uu = _rlogit(out["lp_uu"].iloc[j], h_idx[j])
            uh = _rlogit(out["lp_uh"].iloc[j], h_idx[j])
            r_te.append(hh - uu)
            r_nde.append(hh - uh)
        out["R_TE"] = r_te
        out["R_NDE"] = r_nde

    if "lens_h" in out.columns and out["lens_h"].notna().all():
        r_pre, ent, direct_ok = [], [], []
        for j in range(len(out)):
            lh = _lens_logodds(out["lens_h"].iloc[j], h_idx[j])
            lu = _lens_logodds(out["lens_u"].iloc[j], h_idx[j])
            nl = len(lh)
            top = slice(nl // 2, nl)
            r_pre.append(float(np.mean(lh[top] - lu[top])))
            pu = np.asarray(out["lens_u"].iloc[j], dtype=float)[-1]
            pu = pu / pu.sum()
            ent.append(float(-(pu * np.log(np.clip(pu, 1e-12, 1))).sum()))
            direct_ok.append(int(LETTERS[int(np.argmax(pu))]
                                 == out["correct"].iloc[j]))
        out["R_pre"] = r_pre
        out["direct_entropy"] = ent
        out["direct_correct"] = direct_ok

    out["followed"] = (out["ans_h"] == out["hint_letter"]).astype(int)
    out["flip"] = ((out["ans_h"] == out["hint_letter"])
                   & (out["ans_u"] != out["hint_letter"])).astype(int)
    out["parse_ok"] = out["ans_h"].notna() & out["ans_u"].notna()
    return out


# ---------------------------------------------------------------- designs


def _zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 1e-12 else 1.0)


def design(df: pd.DataFrame, use_lens_covs: bool = True):
    """Outcome design X and selection design W (= X plus the instrument)."""
    n = len(df)
    cols, names = [np.ones(n)], ["const"]
    for ht in sorted(df["hint_type"].unique())[1:]:
        cols.append((df["hint_type"] == ht).astype(float).values)
        names.append(f"hint_{ht}")
    for ds in sorted(df["dataset"].unique())[1:]:
        cols.append((df["dataset"] == ds).astype(float).values)
        names.append(f"ds_{ds}")
    cols.append(_zscore(df["question_len"].values))
    names.append("qlen_z")
    if use_lens_covs and "direct_entropy" in df.columns:
        cols.append(_zscore(df["direct_entropy"].values))
        names.append("entropy_z")
        cols.append(df["direct_correct"].astype(float).values)
        names.append("direct_correct")
    X = np.column_stack(cols)
    W = np.column_stack(cols + [df["z"].astype(float).values])
    return X, W, names + ["z"]


# ------------------------------------------------------------ full report


def _gate(lr_p: float, est: dict, tgt: dict) -> dict:
    err_naive_sel = abs(est["naive_selected"] - tgt["true_pop"])
    err_naive_zero = abs(est["naive_zerofill"] - tgt["true_pop"])
    err_corr = abs(est["corrected_pop"] - tgt["true_pop"])
    err_hid_naive = abs(0.0 - tgt["true_hidden"])
    err_hid_corr = abs(est["corrected_hidden"] - tgt["true_hidden"])
    return {
        "rho_rejected_5pct": bool(lr_p < 0.05),
        "err_naive_selected": err_naive_sel,
        "err_naive_zerofill": err_naive_zero,
        "err_corrected_pop": err_corr,
        "corrected_beats_naive_selected": bool(err_corr < err_naive_sel),
        "corrected_beats_naive_zerofill": bool(err_corr < err_naive_zero),
        "err_hidden_naive0": err_hid_naive,
        "err_hidden_corrected": err_hid_corr,
        "corrected_beats_naive_hidden": bool(err_hid_corr < err_hid_naive),
    }


def fit_report(df: pd.DataFrame, outcome: str = "R_TE",
               n_boot: int = 1000, boot_seed: int = 0,
               do_sensitivity: bool = True,
               per_hint: bool = True) -> dict:
    """The full linear-outcome pipeline against open-model ground truth."""
    d = df[df["parse_ok"] & (df["hint_type"] != "placebo")].copy()
    X, W, names = design(d)
    y_full = d[outcome].values.astype(float)
    s = d["V"].values.astype(float)
    y = np.where(s > 0.5, y_full, np.nan)

    ts = sel.heckman_two_step(y, X, s, W)
    mle = sel.heckman_mle(y, X, s, W, start=ts)
    wald = sel.rho_wald_test(mle)
    lr = sel.rho_lr_test(y, X, s, W, fit_mle=mle)

    est_ts = sel.estimands(ts, y_full, s, X, W)
    est_mle = sel.estimands(mle, y_full, s, X, W)
    tgt = sel.ground_truth_targets(y_full, s)
    boot = sel.bootstrap_fit(y, X, s, W, y_full=y_full, n_boot=n_boot,
                             seed=boot_seed, method="two-step")

    report = {
        "outcome": outcome,
        "n_total": int(len(df)), "n_fit": int(len(d)),
        "n_parse_fail": int((~df["parse_ok"]).sum()),
        "design_names": names,
        "verbalization_rate": float(s.mean()),
        "flip_rate": float(d["flip"].mean()),
        "followed_rate": float(d["followed"].mean()),
        "turpin_naive_faithfulness_P(V|flip)":
            float(d.loc[d["flip"] == 1, "V"].mean())
            if (d["flip"] == 1).any() else np.nan,
        "P(V|followed)": float(d.loc[d["followed"] == 1, "V"].mean())
            if (d["followed"] == 1).any() else np.nan,
        "gamma_z_first_stage": _first_stage(W, s, names),
        "two_step": {"rho": ts.rho, "sigma": ts.sigma,
                     "beta": ts.beta.tolist(), "gamma": ts.gamma.tolist(),
                     "estimands": est_ts},
        "mle": {"rho": mle.rho, "sigma": mle.sigma,
                "beta": mle.beta.tolist(), "gamma": mle.gamma.tolist(),
                "loglik": mle.loglik, "converged": mle.converged,
                "estimands": est_mle},
        "rho_wald": wald, "rho_lr": lr,
        "targets": tgt,
        "gate": _gate(lr["p"], est_ts, tgt),
        "gate_mle": _gate(lr["p"], est_mle, tgt),
        "bootstrap": boot,
    }
    if do_sensitivity:
        report["rho_sensitivity"] = sel.rho_sensitivity(y, X, s, W)
    if per_hint:
        per = {}
        for ht in sorted(d["hint_type"].unique()):
            dh = d[d["hint_type"] == ht]
            try:
                per[ht] = _sub_fit(dh, outcome)
            except Exception as e:  # noqa: BLE001 - report, don't crash
                per[ht] = {"error": f"{type(e).__name__}: {e}"}
        report["per_hint"] = per
    return report


def _first_stage(W, s, names) -> dict:
    pr = sel.probit_fit(W, s)
    z_idx = names.index("z")
    se = float(np.sqrt(pr.cov[z_idx, z_idx]))
    coef = float(pr.params[z_idx])
    return {"coef": coef, "se": se, "z_stat": coef / se,
            "p": float(2 * (1 - stats.norm.cdf(abs(coef / se))))}


def _sub_fit(dh: pd.DataFrame, outcome: str) -> dict:
    X, W, names = design(dh)
    y_full = dh[outcome].values.astype(float)
    s = dh["V"].values.astype(float)
    y = np.where(s > 0.5, y_full, np.nan)
    ts = sel.heckman_two_step(y, X, s, W)
    mle = sel.heckman_mle(y, X, s, W, start=ts)
    lr = sel.rho_lr_test(y, X, s, W, fit_mle=mle)
    est = sel.estimands(ts, y_full, s, X, W)
    tgt = sel.ground_truth_targets(y_full, s)
    return {"n": int(len(dh)), "V_rate": float(s.mean()),
            "rho_two_step": ts.rho, "rho_mle": mle.rho,
            "rho_lr_p": lr["p"], "estimands": est, "targets": tgt,
            "gate": _gate(lr["p"], est, tgt)}


# ------------------------------------------------------------ E1 balance


def e1_balance(df: pd.DataFrame,
               outcomes=("R_TE", "R_NDE", "R_pre")) -> dict:
    """Exclusion-restriction validation: Z must move V, not the reliance."""
    d = df[df["parse_ok"] & (df["hint_type"] != "placebo")]
    z1, z0 = d[d["z"] == 1], d[d["z"] == 0]
    out = {"n_z1": int(len(z1)), "n_z0": int(len(z0)),
           "V_rate_z1": float(z1["V"].mean()),
           "V_rate_z0": float(z0["V"].mean())}
    X, W, names = design(d)
    out["first_stage"] = _first_stage(W, d["V"].values.astype(float), names)
    for oc in outcomes:
        if oc not in d.columns:
            continue
        a, b = z1[oc].values, z0[oc].values
        t, p = stats.ttest_ind(a, b, equal_var=False)
        pooled_sd = np.sqrt(0.5 * (a.var() + b.var()))
        out[oc] = {"mean_z1": float(a.mean()), "mean_z0": float(b.mean()),
                   "diff": float(a.mean() - b.mean()),
                   "welch_t": float(t), "p": float(p),
                   "std_diff": float((a.mean() - b.mean())
                                     / (pooled_sd + 1e-12))}
    return out


def placebo_report(df: pd.DataFrame, outcome: str = "R_TE") -> dict:
    """Placebo hint: correction should find nothing (rho ~ 0, R ~ 0)."""
    d = df[df["parse_ok"] & (df["hint_type"] == "placebo")].copy()
    if len(d) == 0:
        return {"n": 0}
    y_all = d[outcome].values.astype(float)
    ci = stats.t.interval(0.95, len(y_all) - 1, loc=y_all.mean(),
                          scale=stats.sem(y_all))
    out = {"n": int(len(d)), "V_rate": float(d["V"].mean()),
           "mean_R": float(y_all.mean()),
           "R_ci95": [float(ci[0]), float(ci[1])],
           "followed_rate": float(d["followed"].mean())}
    try:
        sub = _sub_fit(d, outcome)
        out["rho_two_step"] = sub["rho_two_step"]
        out["rho_mle"] = sub["rho_mle"]
        out["rho_lr_p"] = sub["rho_lr_p"]
    except Exception as e:  # noqa: BLE001
        out["fit_error"] = f"{type(e).__name__}: {e}"
    return out


# ------------------------------------------- observation-only (heckprob)


def heckprob_report(df: pd.DataFrame, unblind_outcome: str | None = "R_TE",
                    n_boot: int = 0) -> dict:
    """Observation-only pipeline: binary adopted-hint proxy, observed iff V=1.

    y_i = followed (answer equals the hinted letter), readable as reliance
    only when the trace verbalizes the hint. Fits heckprob, reports the
    corrected population/hidden adoption rates; when reliance ground truth
    exists (local models), unblinds against it.

    The FIT never uses white-box covariates (lens-derived entropy /
    direct-answer correctness), even when the raw file has them: the point
    is the regime where only text is observable. Ground truth enters only
    the clearly-labeled unblind fields.
    """
    d = df[df["parse_ok"] & (df["hint_type"] != "placebo")].copy()
    X, W, names = design(d, use_lens_covs=False)
    s = d["V"].values.astype(float)
    y_full = d["followed"].values.astype(float)
    y = np.where(s > 0.5, y_full, np.nan)

    fit = sel.heckprob_mle(y, X, s, W)
    tests = sel.heckprob_rho_tests(y, X, s, W, fit)
    p_pop = fit.predict_population(X)
    p_hidden = fit.predict_hidden(X, W)
    selm = s > 0.5

    report = {
        "n_fit": int(len(d)), "design_names": names,
        "verbalization_rate": float(s.mean()),
        "followed_rate_observed_all": float(y_full.mean()),
        "naive_adoption_among_verbalizers": float(y_full[selm].mean()),
        "naive_zerofill_adoption": float((y_full * s).mean()),
        "rho": fit.rho, "rho_tests": tests, "loglik": fit.loglik,
        "converged": fit.converged,
        "corrected_pop_adoption": float(p_pop.mean()),
        "corrected_hidden_adoption": float(p_hidden[~selm].mean())
            if (~selm).any() else np.nan,
        "true_hidden_adoption_unblind": float(y_full[~selm].mean())
            if (~selm).any() else np.nan,
        "true_pop_adoption_unblind": float(y_full.mean()),
    }
    if unblind_outcome and unblind_outcome in d.columns:
        r = d[unblind_outcome].values.astype(float)
        report["unblind"] = {
            "outcome": unblind_outcome,
            "true_mean_R_pop": float(r.mean()),
            "true_mean_R_hidden": float(r[~selm].mean()),
            "true_mean_R_selected": float(r[selm].mean()),
        }
    if n_boot > 0:
        rng = np.random.default_rng(0)
        draws = {"rho": [], "corrected_pop_adoption": [],
                 "corrected_hidden_adoption": []}
        nb_fail = 0
        for _ in range(n_boot):
            idx = rng.integers(0, len(d), size=len(d))
            try:
                f = sel.heckprob_mle(y[idx], X[idx], s[idx], W[idx])
                draws["rho"].append(f.rho)
                draws["corrected_pop_adoption"].append(
                    float(f.predict_population(X[idx]).mean()))
                hid = f.predict_hidden(X[idx], W[idx])
                m = s[idx] < 0.5
                draws["corrected_hidden_adoption"].append(
                    float(hid[m].mean()) if m.any() else np.nan)
            except Exception:  # noqa: BLE001
                nb_fail += 1
        report["bootstrap"] = {
            k: {"lo95": float(np.nanpercentile(v, 2.5)),
                "hi95": float(np.nanpercentile(v, 97.5)),
                "mean": float(np.nanmean(v))}
            for k, v in draws.items() if len(v)}
        report["bootstrap"]["n_fail"] = nb_fail
    return report


# ----------------------------------------------------------------- output


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, cls=NpEncoder, allow_nan=True)
