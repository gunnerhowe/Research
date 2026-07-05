"""Generate paper/numbers.tex from results/*.json (house rule 1: every number
in the paper is machine-generated; prose ratios are macros).

Run:  python paper/gen_paper_numbers.py   (from repo root or paper/)
"""

import json
import math
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

gate_v = load("exp0_gate_v.json")
if gate_v:
    setm("GateVRiceGP", pct(gate_v["rice_gp_max_rel_err"]))
    setm("GateVOU", pct(gate_v["ou_discrete_max_rel_err"]))
    setm("GateVMultisine", pct(gate_v["multisine_max_rel_err"], 2))

ou = load("exp0_ou.json")
if ou:
    # worst continuous-Rice underprediction at u=0 across rho
    ratios = {}
    for r in ou:
        ratios.setdefault(r["rho"], []).append(r["rice_fd"][0] / r["measured"][0])
    worst = min(np.mean(v) for v in ratios.values())
    setm("OURiceFDWorst", pct(1 - worst))

gam = load("exp0_sod_gamma.json")
if gam:
    x = np.array(gam["x_grid"])
    sp = np.array(gam["process_spread"])
    pool = np.array(gam["pooled_gamma"])
    lo = x <= 0.5
    setm("GammaSpreadSmallX", pct(np.max(sp[lo] / pool[lo])))
    setm("GammaAtHalf", num(float(np.interp(0.5, x, pool)), 2))

eps = load("exp0_eps_sweep.json")
if eps:
    setm("SplitHalfFloorE0", pct(eps["split_half_floor"][0]))
    best = min(eps["eps_sweep"], key=lambda r: abs(r["seg_bias"][0]))
    setm("EpsBest", num(best["eps_rel"], 2))

# ---------------------------------------------------------------- E1

gate_p = load("exp1_gate_p.json")
TASKNAME = {"sc2": "SCTwo", "psmnist": "PsMNIST", "enwik8": "Enwik"}
if gate_p:
    for t, tex in TASKNAME.items():
        if t in gate_p:
            setm(f"EmpMedian{tex}", pct(gate_p[t]["empirical_median"]))
            setm(f"EmpMax{tex}", pct(gate_p[t]["empirical_max"]))
            setm(f"RiceMedian{tex}", pct(gate_p[t]["rice_median"], 0))

for t, tex in TASKNAME.items():
    d = load(f"exp1_{t}.json")
    if not d:
        continue
    # iid range over streams
    iid = [p["iid"]["median_rel_err"] for s in d.values()
           for p in s["profiles"].values()]
    setm(f"IidMin{tex}", f"{min(iid):.1f}$\\times$")
    setm(f"IidMax{tex}", f"{max(iid):.1f}$\\times$")
    rice = [p["rice"]["median_rel_err"] for s in d.values()
            for p in s["profiles"].values()]
    setm(f"RiceMin{tex}", pct(min(rice), 0))
    setm(f"RiceMax{tex}", pct(max(rice), 0))
    # kurtosis correlation (per-channel rice err vs kurt, pooled)
    ku, er = [], []
    for s in d.values():
        for prof in s["profiles"].values():
            for a, b in zip(prof["kurt"], prof["rice"]["per_channel_median"]):
                if b is not None:
                    ku.append(a)
                    er.append(b)
    rho_s = stats.spearmanr(ku, er).statistic
    setm(f"KurtSpearman{tex}", num(rho_s, 2))
    # event prediction: openloop_cal err in deployable regime x<=0.75
    if t != "enwik8":
        errs, errs_all = [], []
        for s in d.values():
            for row in s["events"]:
                for st in ("input", "h1", "h2"):
                    e = abs(row[st]["pred_openloop_cal"] - row[st]["measured"]
                            ) / max(row[st]["measured"], 1e-9)
                    errs_all.append(e)
                    if row["x"] <= 0.75:
                        errs.append(e)
        setm(f"EventOpenloopMax{tex}", pct(max(errs)))
    else:
        errs = []
        for s in d.values():
            for row in s["events"]:
                if row["x"] <= 0.75:
                    for st in row["streams"].values():
                        errs.append(abs(st["pred_openloop_cal"] - st["measured"]
                                        ) / max(st["measured"], 1e-9))
        setm(f"EventOpenloopMax{tex}", pct(max(errs)))

# analytic gamma-transfer worst error on sc2 (h1, x<=1)
d = load("exp1_sc2.json")
if d:
    errs = [abs(row["h1"]["pred_analytic_emp_tv"] - row["h1"]["measured"])
            / row["h1"]["measured"]
            for s in d.values() for row in s["events"] if row["x"] <= 1.0]
    setm("AnalyticGammaWorstSCTwo", pct(max(errs), 0))

# ---------------------------------------------------------------- E2

for t, tex in (("sc2", "SCTwo"), ("psmnist", "PsMNIST")):
    d = load(f"exp2_{t}.json")
    if not d:
        continue
    cost = d["s0"]["cost_forwards"]
    setm(f"TuneCostRatio{tex}",
         f"{cost['random_search_total'] // cost['calibration']}$\\times$")
    # realized vs target accuracy of budget hitting (targets >= 0.2)
    rel = [abs(r["frac_of_dense"] - r["budget_frac"]) / r["budget_frac"]
           for s in d for r in d[s]["analytic"]["uniform_x"]
           if r["budget_frac"] >= 0.2]
    setm(f"AllocRealizedErr{tex}", pct(max(rel), 0))
    # analytic acc at target 0.2 (uniform_x)
    a02 = [r["acc"] for s in d for r in d[s]["analytic"]["uniform_x"]
           if r["budget_frac"] == 0.2]
    setm(f"AllocAccAtTwenty{tex}", pct(np.mean(a02)))

# events fraction at <=1 point acc drop: analytic vs random search
def frac_at_drop(points, dense_acc, drop=0.01):
    """points: sorted (frac, acc); smallest frac with acc >= dense - drop."""
    ok = [f for f, a in points if a >= dense_acc - drop]
    return min(ok) if ok else None


for t, tex in (("sc2", "SCTwo"), ("psmnist", "PsMNIST")):
    d2 = load(f"exp2_{t}.json")
    d1 = load(f"exp1_{t}.json")
    if not d2:
        continue
    base = load("base_training.json")
    per_seed = {"an": [], "k6": [], "k48": []}
    for s in d2:
        dense_acc = base[f"{t}_s{s[1:]}"]["test_acc"]
        an = sorted((r["frac_of_dense"], r["acc"])
                    for m in ("uniform_x", "prop_share", "single_theta")
                    for r in d2[s]["analytic"][m])
        per_seed["an"].append(frac_at_drop(an, dense_acc))
        for k in ("k6", "k48"):
            rs = sorted((r["frac_of_dense"], r["acc"])
                        for r in d2[s]["random_search"][k])
            per_seed[k].append(frac_at_drop(rs, dense_acc))
    for key, name in (("an", "Analytic"), ("k6", "RSSix"), ("k48", "RSFortyEight")):
        vals = [v for v in per_seed[key] if v is not None]
        if vals:
            setm(f"FracAtOnePt{name}{tex}", pct(np.mean(vals), 0))

# ---------------------------------------------------------------- E3/E4

REF_FRACS = {"sc2": 0.20, "psmnist": 0.40}


def envelope_at(d, prefix, frac):
    """Per-seed front accuracy at event fraction `frac` for arms matching
    prefix; np.nan when the front doesn't reach that fraction."""
    out = {}
    for s in d:
        pts = []
        for name, arm in d[s].items():
            if name == prefix or name.startswith(prefix):
                pts += [(r["frac_of_dense"], r["acc"]) for r in arm["front"]]
        if not pts:
            continue
        pts.sort()
        ef, ea, best = [], [], -1
        for f, a in pts:
            if a > best:
                ef.append(f)
                ea.append(a)
                best = a
        out[s] = float(np.interp(frac, ef, ea, left=np.nan, right=ea[-1]))
    return out


for t, tex in (("sc2", "SCTwo"), ("psmnist", "PsMNIST")):
    d3 = load(f"exp3_{t}.json")
    d4 = load(f"exp4_{t}.json")
    if not d3:
        continue
    frac = REF_FRACS[t]
    setm(f"RefFrac{tex}", pct(frac, 0))
    fam = {}
    for prefix, name in (("posthoc", "Posthoc"), ("budget_", "Budget")):
        fam[prefix] = envelope_at(d3, prefix, frac)
    if d4:
        for prefix, name in (("plain", "Plain"), ("l1delta_", "LOne"),
                             ("rate_", "Rate")):
            fam[prefix] = envelope_at(d4, prefix, frac)
    for prefix, name in (("posthoc", "Posthoc"), ("budget_", "Budget"),
                         ("plain", "Plain"), ("l1delta_", "LOne"),
                         ("rate_", "Rate")):
        if prefix in fam and fam[prefix]:
            v = np.array(list(fam[prefix].values()))
            setm(f"E3{name}{tex}", pct(np.nanmean(v)))
            setm(f"E3{name}Sd{tex}", pct(np.nanstd(v)))
    # paired budget - posthoc (shared seeds)
    if "budget_" in fam and fam["budget_"]:
        shared = sorted(set(fam["posthoc"]) & set(fam["budget_"]))
        a = np.array([fam["budget_"][s] for s in shared])
        b = np.array([fam["posthoc"][s] for s in shared])
        ok = ~(np.isnan(a) | np.isnan(b))
        diff = a[ok] - b[ok]
        setm(f"E3PairedDiff{tex}", pct(np.mean(diff)))
        setm(f"E3PairedN{tex}", str(int(ok.sum())))
        if ok.sum() >= 5 and np.any(diff != 0):
            p = stats.wilcoxon(diff).pvalue
            setm(f"E3WilcoxonP{tex}", num(p, 3))
    # dense-accuracy cost of the binding budget (rho=0.35)
    dense_cost = []
    for s in d3:
        if "budget_rho0.35" in d3[s]:
            dense_cost.append(d3[s]["posthoc"]["dense_acc"]
                              - d3[s]["budget_rho0.35"]["dense_acc"])
    if dense_cost:
        setm(f"BudgetDenseCost{tex}", pct(np.mean(dense_cost)))
    # sigma_delta shrinkage h1 (rho=0.2 if present else 0.35)
    for rho in ("0.2", "0.35"):
        key = f"budget_rho{rho}"
        if key in d3["s0"]:
            shr = [d3[s]["posthoc"]["sd"]["h1"] / d3[s][key]["sd"]["h1"]
                   for s in d3 if key in d3[s]]
            setm(f"SigmaShrink{tex}", f"{np.mean(shr):.1f}$\\times$")
            break

# ---------------------------------------------------------------- timing

tim = load("timing.json")
if tim:
    setm("OverheadPct", pct(tim["overhead_pct"] / 100.0))
    setm("TaskOnlyMs", num(tim["task_only"]["mean_ms"], 1))
    setm("WithBudgetMs", num(tim["with_budget"]["mean_ms"], 1))
    setm("EstimatorMs", num(tim["estimator_fwd_bwd_ms"]["mean_ms"], 1))

# ---------------------------------------------------------------- base

base = load("base_training.json")
if base:
    for t, tex in TASKNAME.items():
        if t == "enwik8":
            v = [base[k]["test_bpc"] for k in base if k.startswith("enwik8")]
            setm("BaseBpcEnwik", num(np.mean(v), 2))
        else:
            v = [base[k]["test_acc"] for k in base if k.startswith(t)]
            setm(f"BaseAcc{tex}", pct(np.mean(v)))
            setm(f"BaseAccSd{tex}", pct(np.std(v), 2))

# ---------------------------------------------------------------- emit

lines = ["% AUTO-GENERATED by paper/gen_paper_numbers.py -- do not edit"]
for k in sorted(M):
    lines.append(f"\\newcommand{{\\{k}}}{{{M[k]}}}")
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT} ({len(M)} macros)")
