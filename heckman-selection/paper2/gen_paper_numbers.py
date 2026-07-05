"""Generate paper2/numbers.tex from results/*.json (house rule 1).

Run:  python paper2/gen_paper_numbers.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper2" / "numbers.tex"

M = {}


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, digits=1):
    return f"{100.0 * x:.{digits}f}\\%"


def num(x, digits=2):
    return f"{x:.{digits}f}"


def setm(key, val):
    M[key] = val


# ---------------------------------------------------------------- E0

e0 = load("expB_e0.json")
if e0:
    rows = e0["rows"]

    def selbias(sigma, eta, rung="rung0"):
        v = [r[rung]["selection_bias"] for r in rows
             if abs(r["sigma"] - sigma) < 1e-9 and r["eta"] == eta]
        return float(np.mean(v)), float(np.std(v))

    sig_all = sorted({r["sigma"] for r in rows})
    SIGTEX = {0.005: "SigLo", 0.01: "SigMed", 0.02: "SigHi",
              0.04: "SigXhi"}
    for s, tex in SIGTEX.items():
        if any(abs(s - x) < 1e-9 for x in sig_all):
            m, sd = selbias(s, 3.0)
            setm(f"GateBias{tex}", num(m, 4))
            setm(f"GateBias{tex}SD", num(sd, 4))
    for eta, tex in ((2.0, "EtaTwo"), (3.0, "EtaThree"), (4.0, "EtaFour")):
        m, sd = selbias(0.02, eta)
        setm(f"GateBias{tex}", num(m, 4))
    cal = e0.get("calibrated_sigma") or {}
    if "sigma_median" in cal:
        setm("CalibSigDiff", f"{cal['sigma_median']:.4f}")
        m, _ = selbias(round(cal["sigma_median"], 4), 3.0)
        setm("GateBiasCalibDiff", num(m, 4))
    if "sigma_pow3_median" in cal:
        setm("CalibSigPow", f"{cal['sigma_pow3_median']:.4f}")
        m, _ = selbias(round(cal["sigma_pow3_median"], 4), 3.0)
        setm("GateBiasCalibPow", num(m, 4))
    # per-rung decay at sigma=0.02
    for k, tex in ((0, "RungOne"), (1, "RungTwo"), (2, "RungThree")):
        m, _ = selbias(0.02, 3.0, f"rung{k}")
        setm(f"GateBias{tex}", num(m, 4))
    # noise selection mechanism
    v = [r["rung0"]["eps_rung_surv"] for r in rows
         if r["sigma"] == 0.02 and r["eta"] == 3.0]
    setm("GateEpsSurv", num(float(np.mean(v)), 4))

# ---------------------------------------------------------------- E1

e1 = load("expB_e1.json")
if e1:
    rows = e1["rows"]
    sig0 = 0.01

    def pop(key, sigma=sig0):
        v = [r["pop"][key] for r in rows if r["sigma"] == sigma]
        return float(np.mean(v)), float(np.std(v))

    setm("PopTrueMuA", num(rows[0]["pop"]["true_mu_a"], 3))
    setm("PopTrueSdA", num(rows[0]["pop"]["true_sd_a"], 3))
    # overshoot triplet: multi-bracket Heckman mu_a vs noise (all sourced)
    sig_avail = sorted({r["sigma"] for r in rows})
    for s, tex in ((0.005, "Lo"), (0.01, "Med"), (0.02, "Hi")):
        if any(abs(s - x) < 1e-9 for x in sig_avail):
            m, _ = pop("eb_surv_heck_mb_mu_a", s)
            setm(f"PopMuAHeckMBSig{tex}", num(m, 3))
            ma, _ = pop("eb_all_mu_a", s)
            setm(f"PopMuAEBAllSig{tex}", num(ma, 3))
    for key, tex in (("naive_mu_a", "Naive"),
                     ("eb_surv_naive_mu_a", "EBSurv"),
                     ("eb_surv_heck_1b_mu_a", "HeckOneBr"),
                     ("eb_surv_heck_mb_mu_a", "HeckMB"),
                     ("eb_all_mu_a", "EBAll")):
        m, sd = pop(key)
        setm(f"PopMuA{tex}", num(m, 3))
        setm(f"PopMuA{tex}SD", num(sd, 3))
    m, sd = pop("eb_surv_heck_mb_sd_a")
    setm("PopSdAHeckMB", num(m, 3))
    v = [r["pop"]["n_surv"] for r in rows if r["sigma"] == sig0]
    setm("PopNSurv", str(int(np.mean(v))))

    def extrap(method, rung, key="bias", sigma=sig0):
        v = [r["extrap"][f"rung{rung}"][method][key] for r in rows
             if r["sigma"] == sigma]
        return float(np.mean(v)), float(np.std(v))

    for method, tex in (("naive_ls", "Naive"), ("tobit", "Tobit"),
                        ("ab_lag_iv", "ABIV"),
                        ("eb_surv_naive", "EBSurv"),
                        ("eb_surv_heck_mb", "HeckMB"),
                        ("eb_all", "EBAll")):
        m, sd = extrap(method, 0)
        setm(f"ExtrapBias{tex}", num(m, 4))
        setm(f"ExtrapBias{tex}SD", num(sd, 4))
        m, _ = extrap(method, 0, "rmse")
        setm(f"ExtrapRMSE{tex}", num(m, 4))
    # residual at higher noise (sigma=0.02): scopes "essentially removes"
    if any(abs(r["sigma"] - 0.02) < 1e-9 for r in rows):
        setm("ExtrapBiasHeckMBHi", num(extrap("eb_surv_heck_mb", 0,
                                              sigma=0.02)[0], 4))
        setm("ExtrapBiasEBAllHi", num(extrap("eb_all", 0,
                                             sigma=0.02)[0], 4))

    def fresh(method, key="bias"):
        v = [r["fresh"][method][key] for r in rows if r["sigma"] == sig0]
        return float(np.mean(v))

    for method, tex in (("naive_ls", "Naive"), ("eb_surv_naive", "EBSurv"),
                        ("eb_surv_heck_mb", "HeckMB"), ("eb_all", "EBAll")):
        setm(f"FreshBias{tex}", num(fresh(method), 4))

# ---------------------------------------------------------------- E2

e2 = load("expB_e2.json")
if e2:
    rows = e2["rows"]
    lcb = [r for r in rows if "__" not in r["dataset"]]
    pd1 = [r for r in rows if "__" in r["dataset"]]

    def predm(rows_, method, key):
        v = [r["pred@rung0"][method][key] for r in rows_]
        return float(np.mean(v)), float(np.std(v))

    METX = {"last_value": "Last", "naive_pow3": "Pow",
            "naive_pow3_vpen": "PowVpen", "eb_replay": "EB"}
    for fam, rows_ in (("LCB", lcb), ("PD", pd1)):
        if not rows_:
            continue
        for m, tex in METX.items():
            mu, sd = predm(rows_, m, "spearman")
            setm(f"Spear{fam}{tex}", num(mu, 2))
            mu, sd = predm(rows_, m, "bias_top_decile")
            setm(f"TopBias{fam}{tex}", num(mu, 3))
            mu, sd = predm(rows_, m, "rmse")
            setm(f"RMSE{fam}{tex}", num(mu, 3))
            for key, ktex in (("regret_completed", "Reg"),
                              ("regret_allpick", "RegAll")):
                v = [r[m][key] for r in rows_]
                setm(f"{ktex}{fam}{tex}", num(float(np.mean(v)), 4))
        # paired Wilcoxon naive vs eb on completed regret
        a = [r["naive_pow3"]["regret_completed"] for r in rows_]
        b = [r["eb_replay"]["regret_completed"] for r in rows_]
        if len(a) >= 6 and not np.allclose(a, b):
            setm(f"PRegNaiveVsEB{fam}",
                 f"{stats.wilcoxon(a, b).pvalue:.4f}")
        a = [r["last_value"]["regret_completed"] for r in rows_]
        b = [r["eb_replay"]["regret_completed"] for r in rows_]
        if len(a) >= 6 and not np.allclose(a, b):
            setm(f"PRegLastVsEB{fam}",
                 f"{stats.wilcoxon(a, b).pvalue:.4f}")
        v = [r["naive_pow3"]["allpick_was_killed"] for r in rows_]
        setm(f"AllpickKilledFracPow{fam}", pct(float(np.mean(v)), 0))

# -------------------------------------------------------------- write

lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
OUT.write_text("% AUTO-GENERATED by gen_paper_numbers.py -- do not edit\n"
               + "\n".join(lines) + "\n")
print(f"wrote {OUT} with {len(M)} macros")
