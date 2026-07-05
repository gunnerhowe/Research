"""Generate paper/numbers.tex + paper/tables/*.tex from results/*.json.
Every number in the paper's prose or tables comes from here (house rule: no
hand-typed numbers; macro names contain no digits)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TABLES = ROOT / "paper" / "tables"
TABLES.mkdir(exist_ok=True)
M = {}


def macro(name, value):
    assert not any(ch.isdigit() for ch in name), f"digit in macro name: {name}"
    M[name] = value


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}\\%"


def num(x, nd=2):
    return f"{x:.{nd}f}"


def load(name):
    with open(RES / name) as f:
        return json.load(f)


exp0, exp1, exp2 = load("exp0_scaling.json"), load("exp1_learned.json"), load("exp2_flow.json")
exp3, exp4 = load("exp3_baselines.json"), load("exp4_boundary.json")

# ---------------------------------------------------------------- E0 (GATE S)
g = exp0["gate_s"]["per_quantity"]
macro("GateSHoldoutGamma", pct(g["gamma"]["holdout_median_rel_err"]))
macro("GateSHoldoutS", pct(g["s_density"]["holdout_median_rel_err"]))
macro("GateSSOneGamma", str(g["gamma"]["n_s1_pass"]))
macro("GateSSOneS", str(g["s_density"]["n_s1_pass"]))
macro("GateSNumK", str(g["gamma"]["n_k"]))
macro("AiccPowTwoGamma", str(g["gamma"]["aicc_prefers_power2"]))
macro("AiccPowTwoS", str(g["s_density"]["aicc_prefers_power2"]))
ps = exp0["per_size"]
macro("MedRSquaredWorst", num(min(v["median_r2_edmd_acf"] for v in ps.values()), 2))
macro("MedRSquaredBest", num(max(v["median_r2_edmd_acf"] for v in ps.values()), 2))
macro("SectorsSmallest", str(ps["22"]["n_sectors"]))
macro("SectorsHoldout", str(ps["176"]["n_sectors"]))
macro("SimMinutesSmall", num(sum(v["wall_s"] for v in ps.values()) / 60.0, 0))
r_low = g["gamma"]["rows"][1]   # k = 2*2pi/22 ~ 0.571
macro("GammaSmallKSmallL", num(r_low["y"][0], 3))
macro("GammaSmallKLargeL", num(r_low["y"][-1], 3))
macro("GammaGrowthFactor", num(r_low["y"][-1] / max(r_low["y"][0], 1e-9), 1))

# ---------------------------------------------------------------- E1 (K3)
k3 = exp1["k3"]
macro("KThreeSMedLogTen", num(k3["statics"]["median_abs_log10_S"], 3))
macro("KThreeSMedPct", pct(10 ** k3["statics"]["median_abs_log10_S"] - 1))
macro("KThreeCRelLTwo", pct(k3["statics"]["c_rel_l2"]))
macro("KThreeLamMedRel", pct(k3["resonance"]["median_rel_lam"]))
macro("KThreeGammaMedRel", pct(k3["resonance"]["median_rel_gamma"]))
macro("KThreeOmegaMedRel", pct(k3["resonance"]["median_rel_omega"]))
macro("KThreeTrainMinutes",
      num(float(np.median([r["train_wall_s"] for r in exp1["per_seed"]])) / 60.0, 1))

# ---------------------------------------------------------------- E2 (the flow)
def med(entry, m):
    return entry["median"].get(m, float("nan"))


for Lt, tex in (("176", "Holdout"), ("1408", "Target")):
    e = exp2["targets"][Lt]["methods"]["fitted_flow"]
    macro(f"Flow{tex}Gamma", pct(med(e, "gamma_med_rel")))
    macro(f"Flow{tex}SLogTen", num(med(e, "s_med_log10"), 3))
    macro(f"Flow{tex}SPct", pct(10 ** med(e, "s_med_log10") - 1))
    macro(f"Flow{tex}C", pct(med(e, "c_rel_l2")))
    macro(f"Flow{tex}Tau", pct(med(e, "tau_med_rel")))
    macro(f"Flow{tex}Slow", num(med(e, "slow_overlap"), 2))
    nb = e.get("new_band") or {}
    if nb.get("new_mode"):
        macro(f"Flow{tex}NewGamma", pct(nb["new_mode"]["gamma_med_rel"]))
        macro(f"Flow{tex}NewSPct", pct(10 ** nb["new_mode"]["s_med_log10"] - 1))
        macro(f"Flow{tex}NewModes", str(nb["new_mode"]["n_modes"]))
macro("TargetSectors", str(exp2["targets"]["1408"]["n_sectors"]))
macro("TargetFactor", "64")

# ---------------------------------------------------------------- E3 (K2 fires)
t3 = exp3["targets"]["1408"]["methods"]
for name, mac in (("interp22", "InterpTwoTwo"), ("interp44", "InterpFourFour"),
                  ("interp88", "InterpEightEight")):
    macro(f"{mac}Gamma", pct(med(t3[name], "gamma_med_rel")))
    macro(f"{mac}SPct", pct(10 ** med(t3[name], "s_med_log10") - 1))
    macro(f"{mac}C", pct(med(t3[name], "c_rel_l2")))
    macro(f"{mac}Tau", pct(med(t3[name], "tau_med_rel")))
macro("TilingC", pct(t3["strict_tiling"]["median"]["c_rel_l2"]))
macro("TilingBandPower", pct(t3["strict_tiling"]["median"]["band_power_med_rel"]))
macro("SmallBaseGamma", pct(med(t3["fitted_flow_smallbase"], "gamma_med_rel")))
macro("SmallBaseSPct", pct(10 ** med(t3["fitted_flow_smallbase"], "s_med_log10") - 1))
macro("EdmdLimShortGamma", pct(med(t3["edmd_limited_T2000"], "gamma_med_rel")))
macro("EdmdLimLongGamma", pct(med(t3["edmd_limited_T10000"], "gamma_med_rel")))
macro("EdmdLimLongS", pct(10 ** med(t3["edmd_limited_T10000"], "s_med_log10") - 1))
k2 = exp3["k2"]
macro("KTwoFlowWins", str(k2["n_wins"]))
macro("KTwoNumMetrics", "5")
macro("KTwoBestNull", {"interp88": "interp-88", "interp44": "interp-44",
                       "interp22": "interp-22"}[k2["metrics"]["gamma_med_rel"]["best_null"]])
macro("KTwoFires", "fires" if k2["k2_fires"] else "does not fire")
macro("SmallBaseWins", str(k2["smallbase_vs_interp44"]["n_wins"]))
comp = exp3["compute"]
macro("OracleTargetMultiple", num(comp["oracle1408_multiple"], 1))
macro("LimitedTargetMultiple", num(comp["limited1408_multiple"], 2))

# ---------------------------------------------------------------- E4 (boundary)
conv = exp4["convergence"]
c88 = next(r for r in conv if r["L"] == 88.0)
c44 = next(r for r in conv if r["L"] == 44.0)
macro("ConvEightEightGamma", pct(c88["gamma_rel_to_limit"]))
macro("ConvEightEightS", pct(c88["s_rel_to_limit"]))
macro("ConvFourFourGamma", pct(c44["gamma_rel_to_limit"]))
lad = exp4["ladder"]


def lad_val(L, method, m="gamma_med_rel"):
    return next(r[m] for r in lad if r["L"] == L and r["method"] == method)


macro("LadderFlowFinal", pct(lad_val(2816.0, "fitted_flow")))
macro("LadderFlowTarget", pct(lad_val(1408.0, "fitted_flow")))
macro("LadderInterpFinal", pct(lad_val(2816.0, "interp88")))
macro("LadderInterpTarget", pct(lad_val(1408.0, "interp88")))
od = exp4["odd_parity"]
macro("OddCVsPeriodic", pct(od["c_odd_vs_periodic_rel_l2"]))
macro("OddCVsFlow", pct(od["c_odd_vs_flow_rel_l2"]))
macro("OddAcfVsFlow", pct(od["acf_pt_odd_vs_flow_rel_l2"]))
macro("OddHealingLength", num(od["healing_length"], 1))
b4 = exp4["bands_1408"]["fitted_flow"]
macro("BandEnergyGamma", pct(b4["energy"]["gamma_med_rel"]))
macro("BandMicroGamma", pct(b4["microscale"]["gamma_med_rel"]))

# ---------------------------------------------------------------- tables
NAME = {"fitted_flow": "FSS flow (ours)", "fitted_flow_smallbase": "FSS flow, base $\\le 44$",
        "interp88": "interp-88 (null)", "interp44": "interp-44 (null)",
        "interp22": "interp-22 (null)", "edmd_limited_T2000": "EDMD@$L$, $T{=}2000$",
        "edmd_limited_T10000": "EDMD@$L$, $T{=}10000$"}


def cell(e, m):
    v = med(e, m)
    if not np.isfinite(v):
        return "--"
    if m == "s_med_log10":
        return f"{v:.3f}"
    if m == "slow_overlap":
        return f"{v:.2f}"
    return f"{100 * v:.1f}"


def write_tabular(name, colspec, header, rows):
    """Write a complete \\begin{tabular}...\\end{tabular} (no \\input boundary
    inside the tabular, which trips booktabs' rule macros)."""
    body = "\n".join(rows)
    tex = (f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
           f"{body}\n\\bottomrule\n\\end{{tabular}}\n")
    (TABLES / f"{name}.tex").write_text(tex)


def table_baselines():
    order = ["fitted_flow", "fitted_flow_smallbase", "interp88", "interp44",
             "interp22", "edmd_limited_T2000", "edmd_limited_T10000"]
    rows = []
    for nm in order:
        e = t3[nm]
        cells = [NAME[nm]] + [cell(e, m) for m in
                              ("gamma_med_rel", "s_med_log10", "c_rel_l2",
                               "tau_med_rel", "slow_overlap")]
        rows.append(" & ".join(cells) + r" \\")
    tile = t3["strict_tiling"]["median"]
    rows.append("strict tiling (null) & -- & -- & "
                f"{100 * tile['c_rel_l2']:.1f} & -- & -- " + r"\\")
    write_tabular("baselines", "lccccc",
                  r"method & $\gc$ & $\Shat\,|\log_{10}|$ & $C(r)$ & $\tau_e$ & slow",
                  rows)


def table_ladder():
    Ls = sorted({r["L"] for r in lad})
    rows = []
    for nm, lab in (("fitted_flow", "FSS flow"), ("interp88", "interp-88")):
        rows.append(" & ".join([lab] + [f"{100 * lad_val(L, nm):.1f}" for L in Ls]) + r" \\")
    write_tabular("ladder", "l" + "c" * len(Ls),
                  "method & " + " & ".join(f"{L:g}" for L in Ls), rows)


def table_convergence():
    rows = [f"{r['L']:g} & {100 * r['gamma_rel_to_limit']:.1f} & "
            f"{100 * r['s_rel_to_limit']:.1f}" + r" \\" for r in conv]
    write_tabular("convergence", "ccc",
                  r"$L$ & $\gc$ to limit (\%) & $\Shat$ to limit (\%)", rows)


table_baselines()
table_ladder()
table_convergence()

out = ROOT / "paper" / "numbers.tex"
with open(out, "w") as f:
    f.write("% AUTO-GENERATED by gen_paper_numbers.py -- do not edit\n")
    for k in sorted(M):
        f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
print(f"wrote {out} ({len(M)} macros) + 3 tables")
