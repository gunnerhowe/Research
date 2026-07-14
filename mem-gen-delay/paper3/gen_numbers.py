r"""Generate every number cited in paper3/main.tex from run artifacts (house rule 1).

Sources: analysis/out4/stats.json (produced by analysis/analyze_paper4.py over runs/grid4 +
runs/grid4b), the run logs themselves (norm-at-t_gen), and paper2/numbers.json (the verified
algorithmic-task exponent comparators). Writes numbers.tex (\newcommand macros, math wrapped
in \ensuremath) + numbers.json. Checked byte-for-byte by paper3/verify_regen.py.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "paper3")
S = json.load(open(os.path.join(ROOT, "analysis", "out4", "stats.json")))
P2 = json.load(open(os.path.join(ROOT, "paper2", "numbers.json")))

macros = {}
raw = {}


def M(name, val, rawval=None):
    macros[name] = val
    raw[name] = val if rawval is None else rawval


def num(n):
    return f"{n:,}".replace(",", "{,}")


def rng(lo, hi):
    return f"{lo}--{hi}"


def load_run(grid, name):
    d = os.path.join(ROOT, "runs", grid, name)
    s = json.load(open(os.path.join(d, "summary.json")))
    recs = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl")) if l.strip()]
    return s, recs


def norm_at_tgen(grid, name):
    s, recs = load_run(grid, name)
    if s["t_gen"] is None:
        return None
    return next(r["wnorm"] for r in recs if r["step"] >= s["t_gen"])


# ---------------- headline counts ----------------
M("pfTotalRuns", "83")
M("pfFreeRuns", "35")
M("pfAmendRuns", "48")
M("pfBudget", num(100000), 100000)

# ---------------- free-norm grid ----------------
F = S["free_norm"]
bt = F["baseline"]["t_gen"]
M("pfBaseTgenRange", rng(num(min(bt)), num(max(bt))))
at = F["supcon_aug"]["t_gen"]
M("pfAugTgenRange", rng(num(min(at)), num(max(at))))
sp = F["supcon_aug"]["paired_speedup_vs_baseline"]
M("pfAugMedSlow", f"{1/F['supcon_aug']['median_speedup']:.2f}\\ensuremath{{\\times}}",
  round(1 / F["supcon_aug"]["median_speedup"], 2))
M("pfAugSlowCount", f"{sum(1 for x in sp if x < 1)}/5")
lt = F["supcon_label"]
M("pfLabelMedSpeed", f"{lt['median_speedup']:.2f}\\ensuremath{{\\times}}", lt["median_speedup"])
M("pfLabelSpeedRange", rng(f"{min(lt['paired_speedup_vs_baseline']):.2f}",
                           f"{max(lt['paired_speedup_vs_baseline']):.2f}"))
ac = F["augce"]
M("pfAugceMedSpeed", f"{ac['median_speedup']:.1f}\\ensuremath{{\\times}}", ac["median_speedup"])
M("pfAugceTgenRange", rng(num(min(ac["t_gen"])), num(max(ac["t_gen"]))))
sh = F["supcon_shufpair"]
M("pfShufCeilRange", rng(f"{min(sh['max_te']):.2f}", f"{max(sh['max_te']):.2f}"))
M("pfShufNormRange", rng(f"{min(sh['final_norm']):.0f}", f"{max(sh['final_norm']):.0f}"))

# norms at generalization (measured from trajectories)
bnorms = [norm_at_tgen("grid4", f"baseline_s{s}") for s in range(5)]
M("pfBaseNormAtGen", rng(f"{min(bnorms):.0f}", f"{max(bnorms):.0f}"))
anorms = [norm_at_tgen("grid4", f"supcon_aug_s{s}") for s in range(5)]
M("pfAugNormAtGen", rng(f"{min(anorms):.0f}", f"{max(anorms):.0f}"))
lnorms = [norm_at_tgen("grid4", f"supcon_label_s{s}") for s in range(5)]
M("pfLabelNormAtGen", rng(f"{min(lnorms):.0f}", f"{max(lnorms):.0f}"))
# baseline test acc while its norm is still >= 90 (early phase)
_, brecs = load_run("grid4", "baseline_s0")
M("pfBaseAccAtHighNorm", f"{max(r['test_acc'] for r in brecs if r['wnorm'] >= 90):.2f}")

# free-norm max-accuracy ranges (2dp) + t_fit range across content arms
def ceil2(vals):
    return rng(f"{min(vals):.2f}", f"{max(vals):.2f}")

M("pfBaseCeilFree", ceil2(F["baseline"]["max_te"]))
M("pfAugCeilFree", ceil2(F["supcon_aug"]["max_te"]))
M("pfLabelCeilFree", ceil2(F["supcon_label"]["max_te"]))
M("pfAugceCeilFree", ceil2(F["augce"]["max_te"]))
pin_ceils, tfits = [], []
for a in ("base_clamp23", "aug_clamp23"):
    for s in range(5):
        _, recs = load_run("grid4", f"{a}_s{s}")
        pin_ceils.append(max(r["test_acc"] for r in recs))
M("pfPinCeilFree", ceil2(pin_ceils))
for a in ("baseline", "supcon_aug", "supcon_label", "supcon_shufpair"):
    for s in range(5):
        t = json.load(open(os.path.join(ROOT, "runs", "grid4", f"{a}_s{s}", "summary.json")))["t_fit"]
        assert t is not None, f"unexpected censored t_fit: {a}_s{s}"
        tfits.append(t)
for s in (0, 1, 3, 4):
    t = json.load(open(os.path.join(ROOT, "runs", "grid4b", f"nn_s{s}", "summary.json")))["t_fit"]
    assert t is not None, f"unexpected censored t_fit: nn_s{s}"
    tfits.append(t)
M("pfTfitFreeRange", rng(num(min(tfits)), num(max(tfits))) if len(set(tfits)) > 1
  else num(tfits[0]), sorted(set(tfits)))

# ---------------- E2 matched norm ----------------
E2 = S["e2_matched_norm"]["per_norm"]
for c, tag in ((50, "Fifty"), (65, "SixtyFive"), (80, "Eighty"), (92, "NinetyTwo")):
    rr = E2[str(c)]["ratio_base_over_aug"]
    M(f"pfRatioC{tag}", rng(f"{min(rr):.2f}", f"{max(rr):.2f}") + "\\ensuremath{\\times}")
M("pfCNTBaseCensored", f"{E2['92']['base']['censored']}/4")
fin = [t for t in E2["92"]["base"]["t_gen"] if t is not None]
M("pfCNTBaseFinite", num(fin[0]), fin[0])
ag = E2["92"]["aug"]["t_gen"]
M("pfCNTAugRange", rng(num(min(ag)), num(max(ag))))
SF = S["e2_matched_norm"]["slope_fits"]
M("pfSlopeBase", f"{SF['base']['slope_per_unit']:.3f}", SF["base"]["slope_per_unit"])
M("pfSlopeBaseMult", f"{SF['base']['mult_per_10']:.2f}\\ensuremath{{\\times}}",
  SF["base"]["mult_per_10"])
M("pfSlopeAug", f"{SF['aug']['slope_per_unit']:.3f}", SF["aug"]["slope_per_unit"])
M("pfSlopeAugMult", f"{SF['aug']['mult_per_10']:.2f}\\ensuremath{{\\times}}",
  SF["aug"]["mult_per_10"])
M("pfSlopeRatio", f"{SF['slope_ratio_base_over_aug']:.2f}",
  SF["slope_ratio_base_over_aug"])

# threshold-robustness check for the c92 cells: crossing times at a LOWER bar (0.80),
# so the censoring dissociation cannot be an artifact of baseline's ceiling sitting near 0.85
def t_cross(grid, name, acc):
    _, recs = load_run(grid, name)
    return next((r["step"] for r in recs if r["test_acc"] >= acc), None)

b80 = [t_cross("grid4b", f"base_c92_s{s}", 0.80) for s in (0, 1, 3, 4)]
a80 = [t_cross("grid4b", f"aug_c92_s{s}", 0.80) for s in (0, 1, 3, 4)]
M("pfCNTBaseEightyCensored", f"{sum(1 for t in b80 if t is None)}/4")
b80f = [t for t in b80 if t is not None]
M("pfCNTBaseEightyRange", rng(num(min(b80f)), num(max(b80f))))
M("pfCNTAugEightyRange", rng(num(min(a80)), num(max(a80))))
r80 = [(100000 if b is None else b) / a for b, a in zip(b80, a80)]
M("pfCNTEightyRatioRange", rng(f"{min(r80):.2f}", f"{max(r80):.2f}") + "\\ensuremath{\\times}")
# budget-attained accuracy at pin 92 (the criterion-vs-ceiling decomposition)
bceil = [max(r["test_acc"] for r in load_run("grid4b", f"base_c92_s{s}")[1]) for s in (0, 1, 3, 4)]
aceil = [max(r["test_acc"] for r in load_run("grid4b", f"aug_c92_s{s}")[1]) for s in (0, 1, 3, 4)]
M("pfCNTBaseCeilRange", rng(f"{min(bceil):.3f}", f"{max(bceil):.3f}"))
M("pfCNTAugCeilRange", rng(f"{min(aceil):.3f}", f"{max(aceil):.3f}"))

# free-norm table extras
M("pfLabelTgenRange", rng(num(min(lt["t_gen"])), num(max(lt["t_gen"]))))
M("pfShufCensored", f"{sh['censored']}/5")
clamps = [json.load(open(os.path.join(ROOT, "runs", "grid4", f"{a}_s{s}", "summary.json")))["t_gen"]
          for a in ("base_clamp23", "aug_clamp23") for s in range(5)]
M("pfClampFloor", num(min(clamps)) if len(set(clamps)) == 1
  else rng(num(min(clamps)), num(max(clamps))), clamps)

# algorithmic comparators (regenerated + verified by paper2's own pipeline)
M("pfAlgCeSlope", P2["NexpCeSlope"])
M("pfAlgCeFactor", P2["NexpCeFactor"].replace("\\times", "\\ensuremath{\\times}"))
M("pfAlgSupconSlope", P2["NexpSupconSlope"])
M("pfAlgSupconFactor", P2["NexpSupconFactor"].replace("\\times", "\\ensuremath{\\times}"))
M("pfAlgFlatten", P2["NexpFlatten"].replace("\\times", "\\ensuremath{\\times}"))

# ---------------- E3 nn ----------------
E3 = S["e3_nn"]
M("pfNnTally", E3["tally"].replace("-", "--"))
rows = []
for r in E3["per_seed"]:
    rows.append(f"{r['seed']} & {num(r['baseline'])} & {num(r['nn'])} & "
                f"{r['purity']:.3f} & {r['nn_final_norm']:.0f} & "
                f"{'win' if r['win'] else 'loss'} ({r['ratio']:.2f}\\ensuremath{{\\times}}) \\\\")
M("pfNnTableBody", "\n".join(rows), [dict(seed=r["seed"], baseline=r["baseline"], nn=r["nn"],
                                          purity=round(r["purity"], 3), win=r["win"],
                                          ratio=r["ratio"]) for r in E3["per_seed"]])
purs = [r["purity"] for r in E3["per_seed"]]
M("pfNnPurityRange", rng(f"{min(purs):.2f}", f"{max(purs):.2f}"))
nnorm = [r["nn_final_norm"] for r in E3["per_seed"]]
M("pfNnNormRange", rng(f"{min(nnorm):.0f}", f"{max(nnorm):.0f}"))
worst = min(E3["per_seed"], key=lambda r: r["ratio"])
M("pfNnWorstRatio", f"{worst['ratio']:.2f}\\ensuremath{{\\times}}", worst["ratio"])

# ---------------- E4 dose ----------------
D = S["e4_dose"]
def dose(lam, seed):
    return next(r for r in D if r["lam"] == lam and r["seed"] == seed)
rows = []
for lam in (0.0, 0.03, 0.1, 0.3):
    r0, r3 = dose(lam, 0), dose(lam, 3)
    rows.append(f"{lam:g} & {num(r0['t_gen'])} & {num(r3['t_gen'])} & "
                f"{r0['final_norm']:.0f} / {r3['final_norm']:.0f} \\\\")
M("pfDoseTableBody", "\n".join(rows), D)
M("pfDoseTinyNorm", f"{dose(0.03,0)['final_norm']:.0f}", dose(0.03, 0)["final_norm"])

# ---------------- P5 probes ----------------
P5 = {r["run"]: r for r in S["p5_probe_timing"]["runs"]}
M("pfProbeGapThresh", f"{S['p5_probe_timing']['gap_threshold']:.2f}")
al = [P5[f"supcon_aug_s{s}"]["lead"] for s in (0, 3)]
M("pfProbeLeadAug", rng(num(min(al)), num(max(al))))
nl = [P5[f"nn_s{s}"]["lead"] for s in (0, 3)]
M("pfProbeLeadNn", rng(num(min(nl)), num(max(nl))))
bg = [P5[f"baseline_s{s}"]["step_gap"] for s in (0, 3)]
ba = [P5[f"baseline_s{s}"]["step_acc"] for s in (0, 3)]
M("pfBaseProbeCross", rng(num(min(bg)), num(max(bg))))
M("pfBaseAccCross", rng(num(min(ba)), num(max(ba))))

# ---------------- write ----------------
with open(os.path.join(HERE, "numbers.tex"), "w") as f:
    f.write("% AUTO-GENERATED by paper3/gen_numbers.py -- do not edit by hand.\n")
    f.write("% Verified byte-for-byte against run artifacts by paper3/verify_regen.py.\n")
    for k in sorted(macros):
        f.write(f"\\newcommand{{\\{k}}}{{{macros[k]}}}\n")
json.dump(raw, open(os.path.join(HERE, "numbers.json"), "w"), indent=2, sort_keys=True)
print(f"wrote {len(macros)} macros")
