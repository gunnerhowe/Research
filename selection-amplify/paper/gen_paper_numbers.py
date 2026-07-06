"""Generate paper/numbers.tex from results/*.json (house rule 1: every number in
the paper is a machine-generated macro; every prose ratio is a macro).

Run:  python paper/gen_paper_numbers.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "numbers.tex"

import sys
sys.path.insert(0, str(ROOT / "src"))
from selamp.stats import (bootstrap_ci, mean_sd, paired_wilcoxon,  # noqa: E402
                          spearman_onesided_pos)

M = {}


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def setm(k, v):
    M[k] = v


def num(x, d=3):
    return f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def pp(x, d=2):                       # percentage POINTS (already a difference)
    return f"{100 * x:.{d}f}"


def pval(p):
    if not np.isfinite(p):
        return "--"
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


BETAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

# --------------------------------------------------------------- E0
e0 = load("exp0_sanity.json")
if e0:
    rows = e0["rows"]
    def agg0(tb, beta, key):
        v = [r[key] for r in rows if r["testbed"] == tb and r["beta"] == beta
             and r.get(key) is not None and np.isfinite(r[key])]
        return float(np.mean(v)) if v else float("nan")
    rg = [agg0("two_moons", b, "spearman_global") for b in BETAS if b > 0]
    rc = [agg0("two_moons", b, "spearman_complement") for b in BETAS if b > 0]
    setm("EZeroRhoGlobalMin", num(np.nanmin(
        [agg0(tb, b, "spearman_global") for tb in
         ["two_moons", "eight_gaussians", "pinwheel"] for b in BETAS if b > 0]), 2))
    setm("EZeroRhoCompMin", num(np.nanmin(
        [agg0(tb, b, "spearman_complement") for tb in
         ["two_moons", "eight_gaussians", "pinwheel"] for b in BETAS if b > 0]), 2))
    setm("EZeroEceMax", num(np.nanmax(
        [agg0(tb, b, "ece") for tb in
         ["two_moons", "eight_gaussians", "pinwheel"] for b in BETAS if b > 0]), 3))
    # I(O;X) plugin-vs-analytic correlation (two_moons)
    ia = [agg0("two_moons", b, "IOX_analytic") for b in BETAS]
    ip = [agg0("two_moons", b, "IOX_plugin") for b in BETAS]
    setm("EZeroIOXCorr", num(float(np.corrcoef(ia, ip)[0, 1]), 4))
    setm("IOXBetaZero", num(agg0("two_moons", 0.0, "IOX_analytic"), 3))
    setm("IOXBetaMax", num(agg0("two_moons", 8.0, "IOX_analytic"), 3))

# --------------------------------------------------------------- E2
e2 = load("exp2_curve.json")


def e2_slice(rows, beta, method):
    return {r["seed"]: r["acc_slice"] for r in rows
            if r["beta"] == beta and r["method"] == method
            and np.isfinite(r.get("acc_slice", float("nan")))}


def e2_full(rows, beta, method):
    return {r["seed"]: r["acc_full"] for r in rows
            if r["beta"] == beta and r["method"] == method}


BTOK = {0.0: "Zero", 0.5: "Half", 1.0: "One", 2.0: "Two", 4.0: "Four",
        8.0: "Eight"}
MTAG = {"B0_obs": "Bzero", "method": "Method", "decoy_rotate": "Decoy",
        "B3_reweight": "Brew", "B4_oracle": "Boracle",
        "B2div_matched": "Btwodiv", "B2_uncond": "Btwo"}

if e2:
    rows = e2["rows"]
    iox = {b: float(np.mean([r["iox"] for r in rows
                             if r["beta"] == b and r["method"] == "_diag"]))
           for b in BETAS}
    for b in BETAS:
        setm(f"IOX{BTOK[b]}", num(iox[b], 3))
    # per-beta method/decoy slice accuracy + gap
    gaps_by_beta = {}
    for b in BETAS:
        for meth, tag in MTAG.items():
            mu, sd = mean_sd(list(e2_slice(rows, b, meth).values()))
            setm(f"Slice{tag}{BTOK[b]}", pct(mu))
        m = e2_slice(rows, b, "method")
        d = e2_slice(rows, b, "decoy_rotate")
        seeds = sorted(set(m) & set(d))
        gap = np.array([m[s] - d[s] for s in seeds])
        gaps_by_beta[b] = gap
        gmu, gsd = mean_sd(gap)
        lo, hi = bootstrap_ci(gap)
        setm(f"Gap{BTOK[b]}", pp(gmu))
        setm(f"GapSD{BTOK[b]}", pp(gsd))
        setm(f"GapCI{BTOK[b]}", f"[{pp(lo)}, {pp(hi)}]")
    # left-end (beta=0): method slice gain over B0
    m0 = e2_slice(rows, 0.0, "method"); b0 = e2_slice(rows, 0.0, "B0_obs")
    ss = sorted(set(m0) & set(b0))
    g0 = np.array([m0[s] - b0[s] for s in ss])
    setm("LeftEndGain", pp(np.mean(g0)))
    lo, hi = bootstrap_ci(g0)
    setm("LeftEndGainCI", f"[{pp(lo)}, {pp(hi)}]")
    setm("LeftEndGap", pp(np.mean(gaps_by_beta[0.0])))

    # K1: Spearman(gap, I(O;X)) across beta on per-(beta,seed) points
    xs, ys = [], []
    for b in BETAS:
        for g in gaps_by_beta[b]:
            xs.append(iox[b]); ys.append(g)
    rho, p1 = spearman_onesided_pos(xs, ys)
    setm("KOneSpearman", num(rho, 3))
    setm("KOneSpearmanP", pval(p1))
    # K1 alt: paired Wilcoxon gap(beta=8) vs gap(beta=0)
    pw, eff = paired_wilcoxon(gaps_by_beta[8.0], gaps_by_beta[0.0], "greater")
    setm("KOneWilcoxonP", pval(pw))
    setm("KOneWilcoxonEff", num(eff, 2))

    # K2 (core falsifier): at beta=8, method>decoy and method>B2div on slice
    for hi_b in [8.0]:
        m = e2_slice(rows, hi_b, "method"); d = e2_slice(rows, hi_b, "decoy_rotate")
        ss = sorted(set(m) & set(d))
        p, eff = paired_wilcoxon([m[s] for s in ss], [d[s] for s in ss], "greater")
        setm("KTwoMethodVsDecoyP", pval(p))
        setm("KTwoMethodVsDecoyEff", num(eff, 2))
        b2d = e2_slice(rows, hi_b, "B2div_matched")
        ss2 = sorted(set(m) & set(b2d))
        p2, eff2 = paired_wilcoxon([m[s] for s in ss2], [b2d[s] for s in ss2], "greater")
        setm("KTwoMethodVsBTwoDivP", pval(p2))
        setm("KTwoMethodVsBTwoDivEff", num(eff2, 2))
    # method vs B3 at high beta (the squeeze headline)
    m = e2_slice(rows, 8.0, "method"); b3 = e2_slice(rows, 8.0, "B3_reweight")
    ss = sorted(set(m) & set(b3))
    p, eff = paired_wilcoxon([m[s] for s in ss], [b3[s] for s in ss], "greater")
    setm("MethodVsBThreeP", pval(p))
    # positive magnitude by which the method TRAILS B3 reweighting
    setm("MethodVsBThreeGap", pp(np.mean([b3[s] - m[s] for s in ss])))

    # diagnostics: off-manifold reject at high beta (K3)
    dg8 = [r for r in rows if r["beta"] == 8.0 and r["method"] == "_diag"]
    setm("MethodRejectHi", pct(float(np.mean([r["method_reject"] for r in dg8])), 1))

# --------------------------------------------------------------- E1
e1 = load("exp1_bridge.json")
if e1:
    r4 = [r for r in e1["rows"] if r["beta"] == 4.0]
    setm("EOneCompHitMethod", num(np.mean([r["method"]["complement_hit"] for r in r4]), 3))
    setm("EOneCompHitDecoy", num(np.mean([r["decoy"]["complement_hit"] for r in r4]), 3))
    setm("EOneCompHitBTwo", num(np.mean([r["b2"]["complement_hit"] for r in r4]), 3))
    setm("EOneRejectMethod", pct(np.mean([r["method"]["reject_rate"] for r in r4]), 1))

# --------------------------------------------------------------- E4
e4 = load("exp4_squeeze.json")
if e4:
    f = e4["foreign"]
    fg = np.mean([x["foreign_acc_method"] - x["foreign_acc_b0"] for x in f])
    cg = np.mean([x["collar_acc_method"] - x["collar_acc_b0"] for x in f])
    setm("KFourForeignGain", pp(fg))
    setm("KFourCollarGain", pp(cg))
    setm("KFourSynthInForeign", pct(np.mean([x["synth_in_foreign"] for x in f]), 1))
    # robustness: noise degradation
    rb = e4["robust"]
    def noise_gain(eta):
        return np.mean([[n["slice_gain"] for n in r["noise"] if n["eta"] == eta][0]
                        for r in rb])
    setm("NoiseGainClean", pp(noise_gain(0.0)))
    setm("NoiseGainHi", pp(noise_gain(0.4)))
    # cap effect on reject
    def cap_reject(cap):
        return np.mean([[c["reject"] for c in r["cap"] if c["cap"] == cap][0]
                        for r in rb])
    setm("CapOnReject", pct(cap_reject(1.0), 1))
    setm("CapOffReject", pct(cap_reject(1e9), 1))

# --------------------------------------------------------------- E3
e3 = load("exp3_mnist.json")
if e3:
    rows = e3["rows"]
    def e3m(beta, meth, key="acc_slice"):
        return {r["seed"]: r[key][meth] for r in rows if r["beta"] == beta}
    m = e3m(8.0, "method"); d = e3m(8.0, "decoy_rotate"); b0 = e3m(8.0, "B0_obs")
    ss = sorted(set(m) & set(d))
    p, eff = paired_wilcoxon([m[s] for s in ss], [d[s] for s in ss], "greater")
    setm("EThreeSliceGap", pp(np.mean([m[s] - d[s] for s in ss])))
    setm("EThreeGapP", pval(p))
    setm("EThreeMethodGain", pp(np.mean([m[s] - b0[s] for s in ss])))
    setm("EThreeReject", pct(np.mean([r["method_reject"] for r in rows if r["beta"] == 8.0]), 1))

# --------------------------------------------------------------- E5
e5 = load("exp5_real.json")
if e5:
    rows = e5["rows"]
    def e5m(beta, meth):
        return np.nanmean([r["acc_slice"] for r in rows if r["beta"] == beta
                           and r["method"] == meth])
    setm("EFiveSliceMethod", pct(e5m(6.0, "method")))
    setm("EFiveSliceBThree", pct(e5m(6.0, "B3_reweight")))
    setm("EFiveSliceBZero", pct(e5m(6.0, "B0_obs")))
    setm("EFiveSliceDecoy", pct(e5m(6.0, "decoy_rotate")))
    dg = [r for r in rows if r["beta"] == 6.0 and r["method"] == "_diag"]
    setm("EFiveCoverStd", num(np.mean([r["cover_blind_std_shat"] for r in dg]), 3))

# -------------------------------------------------------------- write
lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
OUT.write_text("% AUTO-GENERATED by gen_paper_numbers.py -- do not edit\n"
               + "\n".join(lines) + "\n")
print(f"wrote {OUT} with {len(M)} macros")
