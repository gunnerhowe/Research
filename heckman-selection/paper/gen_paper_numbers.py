"""Generate paper/numbers.tex from results/*.json (house rule 1: every
number in the paper is machine-generated; prose ratios are macros).

Run:  python paper/gen_paper_numbers.py   (from repo root or paper/)
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "numbers.tex"

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

e0 = load("expA_e0.json")


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


if e0:
    r = e0["randhie"]
    setm("EZeroDevTS", sci(r["max_abs_dev_ts"]))
    setm("EZeroDevMLE", sci(r["max_abs_dev_mle"]))
    setm("EZeroLLDev", num(r["ll_dev"], 3))
    setm("EZeroRhoMLE", num(r["mle_rho"], 4))
    setm("EZeroSigmaMLE", num(r["mle_sigma"], 4))
    setm("EZeroRhoTS", num(r["ts_rho"], 4))
    setm("EZeroN", str(r["n"]))
    setm("EZeroNSel", str(r["n_selected"]))
    setm("MrozRhoMLE", num(e0["mroz"]["mle_rho"], 3))

# ---------------------------------------------------------------- E1

rows = load("expA_e1.json")


def agg(rows, method, rho, alpha, key):
    v = [r[key] for r in rows if r["method"] == method
         and r["rho"] == rho and r["alpha"] == alpha and key in r]
    return (float(np.mean(v)), float(np.std(v)), v)


def pairedp(rows, m1, m2, rho, alpha, key):
    v1 = {r["seed"]: r[key] for r in rows if r["method"] == m1
          and r["rho"] == rho and r["alpha"] == alpha and key in r}
    v2 = {r["seed"]: r[key] for r in rows if r["method"] == m2
          and r["rho"] == rho and r["alpha"] == alpha and key in r}
    seeds = sorted(set(v1) & set(v2))
    a = np.array([v1[s] for s in seeds])
    b = np.array([v2[s] for s in seeds])
    if len(seeds) < 5 or np.allclose(a, b):
        return float("nan")
    return float(stats.wilcoxon(a, b).pvalue)


TEXNAME = {"deep_ensemble": "Ens", "mc_dropout": "Drop", "gp": "GP",
           "iw_oracle_ensemble": "IW", "blind_two_head": "Blind",
           "heckman_ens": "Heck", "heckman_2s_ens": "HeckTS",
           "oracle": "Oracle"}

if rows:
    for m, tex in TEXNAME.items():
        for rho, rtex in ((0.6, "Mid"), (0.9, "Hi")):
            for alpha, atex in ((1.0, ""), (0.0, "NoInstr")):
                mu, sd, _ = agg(rows, m, rho, alpha, "picp90_against")
                setm(f"CovAg{tex}{rtex}{atex}", pct(mu))
                setm(f"CovAg{tex}{rtex}{atex}SD", pct(sd))
                mu, sd, _ = agg(rows, m, rho, alpha, "bias_f0_against")
                setm(f"BiasAg{tex}{rtex}{atex}", num(mu, 3))
    # rho recovery + kill-condition margin
    mu, sd, _ = agg(rows, "heckman_ens", 0.9, 1.0, "rho_hat")
    setm("RhoHatHi", num(mu, 2))
    setm("RhoHatHiSD", num(sd, 2))
    mu, sd, _ = agg(rows, "heckman_ens", 0.6, 1.0, "rho_hat")
    setm("RhoHatMid", num(mu, 2))
    mu, sd, _ = agg(rows, "heckman_ens", 0.9, 0.0, "rho_hat")
    setm("RhoHatHiNoInstr", num(mu, 2))
    p = pairedp(rows, "iw_oracle_ensemble", "heckman_2s_ens", 0.9, 1.0,
                "picp90_against")
    setm("PKillIWvsHeck", f"{p:.3f}" if p == p else "--")
    p = pairedp(rows, "deep_ensemble", "heckman_2s_ens", 0.9, 1.0,
                "picp90_against")
    setm("PEnsVsHeck", f"{p:.3f}" if p == p else "--")
    # ECE overall at high rho
    for m in ("deep_ensemble", "iw_oracle_ensemble", "heckman_2s_ens",
              "heckman_ens", "oracle"):
        mu, sd, _ = agg(rows, m, 0.9, 1.0, "ece_against")
        setm(f"ECEAg{TEXNAME[m]}Hi", num(mu, 3))
    meta = [r["selected_frac"] for r in rows if r["method"] == "_meta"]
    setm("EOneSelFrac", pct(float(np.mean(meta)), 0))

# ---------------------------------------------------------------- E2

rows2 = load("expA_e2.json")
if rows2:
    def agg2(ds, method, rho, key):
        v = [r[key] for r in rows2 if r["method"] == method
             and r["dataset"] == ds and r["rho"] == rho and key in r]
        return (float(np.mean(v)), float(np.std(v)))

    DSTEX = {"california": "Cal", "wine": "Wine"}
    for ds, dtex in DSTEX.items():
        for m, tex in list(TEXNAME.items()) + [("skyline_ensemble",
                                                "Sky")]:
            if m == "oracle":
                continue
            mu, sd = agg2(ds, m, 0.8, "picp90_against")
            if mu == mu:
                setm(f"RealCov{dtex}{tex}", pct(mu))
                setm(f"RealCov{dtex}{tex}SD", pct(sd))
            ec, _ = agg2(ds, m, 0.8, "ece_against")
            if ec == ec:
                setm(f"RealECE{dtex}{tex}", num(ec, 3))
            ecw, _ = agg2(ds, m, 0.8, "picp90_well")
            if ecw == ecw:
                setm(f"RealCovWell{dtex}{tex}", pct(ecw))
        for m, tex in (("heckman_2s_ens", "HeckTS"),
                       ("deep_ensemble", "Ens")):
            mu, sd = agg2(ds, m, 0.0, "picp90_against")
            if mu == mu:
                setm(f"RealCovZero{dtex}{tex}", pct(mu))

    # paired test at rho=0.8 pooling datasets by (ds, seed)
    def paired_real(m1, m2, key="ece_against", alt="less"):
        a, b = [], []
        for ds in DSTEX:
            v1 = {r["seed"]: r[key] for r in rows2 if r["method"] == m1
                  and r["dataset"] == ds and r["rho"] == 0.8 and key in r}
            v2 = {r["seed"]: r[key] for r in rows2 if r["method"] == m2
                  and r["dataset"] == ds and r["rho"] == 0.8 and key in r}
            for s in sorted(set(v1) & set(v2)):
                a.append(v1[s])
                b.append(v2[s])
        if len(a) < 6:
            return float("nan")
        return float(stats.wilcoxon(a, b, alternative=alt).pvalue)

    # one-sided: is Heckman ECE lower than the deep ensemble / IW / dropout?
    setm("PRealHeckVsEns",
         f"{paired_real('heckman_2s_ens', 'deep_ensemble'):.4f}")
    setm("PRealHeckVsIW",
         f"{paired_real('heckman_2s_ens', 'iw_oracle_ensemble'):.4f}")
    setm("PRealHeckVsDrop",
         f"{paired_real('heckman_2s_ens', 'mc_dropout'):.4f}")

# ---------------------------------------------------------------- E4

e4 = load("expA_e4.json")
if e4:
    n_bound, n_tot, interior = 0, 0, []
    for task, res in e4.items():
        if "pairwise" not in res:
            continue
        if res.get("boundary"):
            n_bound += 1
        n_tot += 1
        for pw in res["pairwise"]:
            if "rho_mle" in pw:
                n_tot += 1
                if pw["boundary"]:
                    n_bound += 1
                else:
                    interior.append(pw)
    setm("VigNBoundary", str(n_bound))
    setm("VigNTotal", str(n_tot))
    imnet = [p for p in interior
             if p.get("target", "").startswith("ImageNet ReaL")]
    if imnet:
        setm("VigImageNetReaLRho", num(imnet[0]["rho_mle"], 2))
        se = imnet[0].get("rho_mle_se")
        setm("VigImageNetReaLSE", num(se, 2) if se else "--")
    mt = e4.get("Machine Translation", {})
    if "rho_mle" in mt:
        setm("VigMTRho", num(mt["rho_mle"], 2))

# -------------------------------------------------------------- write

lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
OUT.write_text("% AUTO-GENERATED by gen_paper_numbers.py -- do not edit\n"
               + "\n".join(lines) + "\n")
print(f"wrote {OUT} with {len(M)} macros")
