"""Estimators for the S x G crossed design (PLAN.md endpoints).

Primary: a mixed-effects fit  Y ~ S + G + S:G  with a random intercept per stem
(the four cells of a stem are matched, so stem is the cluster). Reported per
judge. Secondary: the hackability index P(Y[LH] > Y[HL]) over matched stems, and
cluster (stem-resampled) bootstrap CIs on every headline effect.

`s`, `g` are coded 0/1 (low/high). Effects are on the judge's 1-9 novelty scale.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class MixedFit:
    judge: str
    n_obs: int
    n_stems: int
    beta_S: float          # substantive-novelty main effect
    beta_G: float          # signal main effect (the bias)
    beta_SG: float         # interaction
    intercept: float
    ratio_S_over_G: float  # calibration: >>1 good, <1 signal-dominated
    # cluster-bootstrap 95% CIs (stem-resampled)
    ci_S: tuple[float, float]
    ci_G: tuple[float, float]
    ci_ratio: tuple[float, float]

    def as_dict(self) -> dict:
        return asdict(self)


def _ols_sg(df: pd.DataFrame) -> dict:
    """Closed-form OLS of y ~ 1 + s + g + s:g (stem-demeaning handled by caller
    for the bootstrap; here we fit the fixed-effects means, which for a balanced
    2x2 equal the cell-mean contrasts)."""
    x = np.column_stack([
        np.ones(len(df)),
        df["s"].to_numpy(float),
        df["g"].to_numpy(float),
        (df["s"].to_numpy(float) * df["g"].to_numpy(float)),
    ])
    y = df["y"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return {"intercept": beta[0], "beta_S": beta[1], "beta_G": beta[2], "beta_SG": beta[3]}


def _cluster_bootstrap(df: pd.DataFrame, stat_fn, B: int = 2000, seed: int = 0) -> np.ndarray:
    """Resample STEMS with replacement (cluster bootstrap). Returns (B, k) array."""
    rng = np.random.default_rng(seed)
    stems = df["stem"].unique()
    out = []
    by_stem = {s: d for s, d in df.groupby("stem")}
    for _ in range(B):
        pick = rng.choice(stems, size=len(stems), replace=True)
        boot = pd.concat([by_stem[s] for s in pick], ignore_index=True)
        try:
            out.append(stat_fn(boot))
        except Exception:
            out.append([np.nan] * len(stat_fn(df)))
    return np.asarray(out, float)


def fit_mixed(df: pd.DataFrame, judge: str = "", B: int = 2000, seed: int = 0) -> MixedFit:
    """Fit the crossed design for one judge.

    Prefers statsmodels MixedLM (random intercept per stem) for the point
    estimate; falls back to the balanced-2x2 OLS contrasts if MixedLM fails to
    converge. CIs always come from the stem cluster bootstrap (robust to the
    within-stem correlation and to non-normal Y).
    """
    d = df[df["judge"] == judge] if judge else df
    d = d.dropna(subset=["y", "s", "g", "stem"]).reset_index(drop=True)

    point = _ols_sg(d)
    try:
        import statsmodels.formula.api as smf
        m = smf.mixedlm("y ~ s * g", d, groups=d["stem"]).fit(reml=False, method="lbfgs")
        point = {
            "intercept": float(m.params.get("Intercept", point["intercept"])),
            "beta_S": float(m.params.get("s", point["beta_S"])),
            "beta_G": float(m.params.get("g", point["beta_G"])),
            "beta_SG": float(m.params.get("s:g", point["beta_SG"])),
        }
    except Exception:
        pass

    def _stat(frame):
        b = _ols_sg(frame)
        ratio = b["beta_S"] / b["beta_G"] if abs(b["beta_G"]) > 1e-9 else np.nan
        return [b["beta_S"], b["beta_G"], ratio]

    boot = _cluster_bootstrap(d, _stat, B=B, seed=seed)

    def _ci(col):
        v = boot[:, col]
        v = v[np.isfinite(v)]
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)

    ratio = point["beta_S"] / point["beta_G"] if abs(point["beta_G"]) > 1e-9 else np.nan
    return MixedFit(
        judge=judge, n_obs=len(d), n_stems=int(d["stem"].nunique()),
        beta_S=point["beta_S"], beta_G=point["beta_G"], beta_SG=point["beta_SG"],
        intercept=point["intercept"], ratio_S_over_G=float(ratio),
        ci_S=_ci(0), ci_G=_ci(1), ci_ratio=_ci(2),
    )


def hackability_index(df: pd.DataFrame, judge: str = "", B: int = 2000, seed: int = 0) -> dict:
    """P(Y[LH] > Y[HL]) over matched stems: signal (low-substance, high-signal)
    beating substance (high-substance, low-signal). >0.5 => signal wins."""
    d = df[df["judge"] == judge] if judge else df
    wide = d.pivot_table(index="stem", columns="cell", values="y", aggfunc="mean")
    wide = wide.dropna(subset=["LH", "HL"])
    diff = (wide["LH"] - wide["HL"]).to_numpy(float)
    idx = float(np.mean(diff > 0))

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        pick = rng.choice(len(diff), size=len(diff), replace=True)
        boots.append(np.mean(diff[pick] > 0))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if len(diff) else (np.nan, np.nan)
    return {
        "judge": judge, "n_stems": int(len(diff)), "hackability_index": idx,
        "mean_LH_minus_HL": float(np.mean(diff)) if len(diff) else np.nan,
        "ci": (float(lo), float(hi)),
    }


def make_synthetic(n_stems: int, beta_S: float, beta_G: float, beta_SG: float = 0.0,
                   stem_sd: float = 1.0, noise_sd: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """Ground-truth data generator for estimator recovery tests."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_stems):
        u = rng.normal(0, stem_sd)  # stem random intercept
        for (s, g), cell in [((0, 0), "LL"), ((0, 1), "LH"), ((1, 0), "HL"), ((1, 1), "HH")]:
            y = 5.0 + u + beta_S * s + beta_G * g + beta_SG * s * g + rng.normal(0, noise_sd)
            rows.append({"stem": f"s{k}", "cell": cell, "s": s, "g": g,
                         "y": y, "judge": "synthetic"})
    return pd.DataFrame(rows)
