"""Generate paper/numbers.tex from results/*.json. Every number in the paper's
prose or figures caption comes from here (house rule: no hand-typed result
numbers; macro names contain no digits). verify_regen.py checks byte-identity."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper" / "numbers.tex"
M = {}


def macro(name, value):
    assert not any(ch.isdigit() for ch in name), f"digit in macro name: {name}"
    M[name] = value


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}\\%"


def pts(x, nd=1):
    return f"{100 * x:.{nd}f}"


def num(x, nd=2):
    return f"{x:.{nd}f}"


def g(x):
    s = f"{x:g}"
    return s


def load(name):
    with open(RES / name) as f:
        return json.load(f)


BASELINE_LABELS = {"ou": "unconditioned OU", "mesu": "MESU", "ewc_best": "EWC",
                   "benna_fusi": "Benna--Fusi", "replay": "replay", "none": "naive"}

e0 = load("exp0_falsifier.json")
e1 = load("exp1_isolation.json")
e2 = load("exp2_bss2.json")
e3 = load("exp3_baselines.json")
e4 = load("exp4_modality.json")

# ------------------------------------------------------------------ E0 GATE F
macro("NumSeeds", str(len(e0["config"]["seeds"])))
macro("NumSigma", str(len(e0["config"]["sigmas"])))
iu = e0["methods"]["doob"]["inverted_u"]
macro("GateFSigStar", g(iu["sigma_star"]))
macro("GateFLiftPts", pts(iu["lift_over_zero"]))
macro("GateFPzero", num(iu["p_peak_gt_zero"], 3))
macro("GateFPhi", num(iu["p_peak_gt_hi"], 3))
macro("GateFZeroPct", pct(iu["ret_at_zero"]))
macro("GateFPeakPct", pct(iu["ret_at_peak"]))
ctrl_mono = [e0["methods"][m]["monotone_dec_frac"] for m in ("ou", "ewc", "mesu", "none")]
macro("ControlMonoWorst", num(min(ctrl_mono), 2))

# ------------------------------------------------------------------ E1 isolation
macro("KappaZeroLiftPts", pts(e1["kappa_scan"]["0.0"]["inverted_u"]["lift_over_zero"]))
macro("KappaOneLiftPts", pts(e1["kappa_scan"]["1.0"]["inverted_u"]["lift_over_zero"]))

# ------------------------------------------------------------------ E2 BSS-2
b_iu = e2["device_faithful"]["inverted_u"]
macro("BssLiftPts", pts(b_iu["lift_over_zero"]))
macro("BssSigStar", g(b_iu["sigma_star"]))
macro("BssSurviveWord", "survives" if b_iu["inverted_u"] else "does not survive")

# ------------------------------------------------------------------ E3 baselines
Mm = e3["methods"]
macro("DoobStarRetPct", pct(Mm["doob*"]["retention_mean"]))
macro("MesuRetPct", pct(Mm["mesu"]["retention_mean"]))
macro("EwcRetPct", pct(Mm["ewc_best"]["retention_mean"]))
macro("OuRetPct", pct(Mm["ou"]["retention_mean"]))
macro("BennaRetPct", pct(Mm["benna_fusi"]["retention_mean"]))
macro("ReplayRetPct", pct(Mm["replay"]["retention_mean"]))
macro("NaiveRetPct", pct(Mm["none"]["retention_mean"]))
# honest framing: rehearsal-FREE consolidation methods (store no data) vs replay.
REHEARSAL_FREE = ["ou", "mesu", "ewc_best", "benna_fusi", "none"]
best_rf = max(REHEARSAL_FREE, key=lambda k: Mm[k]["retention_mean"])
macro("BestRFName", BASELINE_LABELS.get(best_rf, best_rf))
macro("BestRFRetPct", pct(Mm[best_rf]["retention_mean"]))
macro("BestRFP", num(Mm[best_rf]["wilcoxon_vs_doob_p"], 2))
macro("OursMinusRFPts", pts(Mm["doob*"]["retention_mean"] - Mm[best_rf]["retention_mean"]))
# the anchored methods ours SIGNIFICANTLY beats: report the worst (largest) p among them
_beaten = [Mm[k]["wilcoxon_vs_doob_p"] for k in REHEARSAL_FREE
           if k != best_rf and Mm[k]["wilcoxon_vs_doob_p"] < 0.05]
macro("AnchoredMaxP", num(max(_beaten), 3) if _beaten else "n/a")
# replay stores data and does better -> reported honestly, not as a loss we hide
macro("ReplayMinusOursPts", pts(Mm["replay"]["retention_mean"] - Mm["doob*"]["retention_mean"]))
macro("ReplayBuffer", str(e3["config"]["replay_buffer"]))
macro("NoiseTaxPct", pct(Mm["doob*"]["noise_tax_gpu"], 0))

# ------------------------------------------------------------------ E4 modality
yy = e4["yin_yang"]["doob"]["inverted_u"]
macro("YYLiftPts", pts(yy["lift_over_zero"]))
macro("YYSigStar", g(yy["sigma_star"]))
macro("YYSurviveWord", "reproduces" if yy["inverted_u"] else "does not reproduce")

# ------------------------------------------------------------------ emit
lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
OUT.write_text("\n".join(lines) + "\n")
print(f"[wrote] {OUT}  ({len(M)} macros)")
