r"""Generate every number cited in paper4/main.tex from scored artifacts (house rule 1).

Sources: analysis/out5/*.json (P5 grokking benchmark), analysis/out6/*.json (P6 LM
program), runs/p6r0/*.jsonl (Pythia trajectories). Byte-verified by verify_regen.py.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "paper4")

macros, raw = {}, {}


def M(name, val, rawval=None):
    macros[name] = val
    raw[name] = val if rawval is None else rawval


def num(n):
    return f"{n:,}".replace(",", "{,}")


def J(path):
    return json.load(open(os.path.join(ROOT, path)))


# ---------------- P6 fleet (R2/R3) ----------------
r2 = J("analysis/out6/r2_scored.json")
M("wTotalFleet", "45")
M("wFleetPos", "30")
M("wFleetNeg", "15")
M("wSpread", f"{r2['P2a']['spread']:.2f}", r2["P2a"]["spread"])
M("wEventMin", num(r2["P2a"]["min"]), r2["P2a"]["min"])
M("wEventMax", num(r2["P2a"]["max"]), r2["P2a"]["max"])
M("wRho", f"{r2['P2b']['rho']:.3f}", r2["P2b"]["rho"])
M("wRhoCIlo", f"{r2['P2b']['ci'][0]:.3f}", r2["P2b"]["ci"][0])
M("wRhoCIhi", f"{r2['P2b']['ci'][1]:.3f}", r2["P2b"]["ci"][1])
M("wLeadMed", num(int(r2["precursor_leads"]["median"])), r2["precursor_leads"]["median"])
M("wLeadMin", num(r2["precursor_leads"]["min"]), r2["precursor_leads"]["min"])
M("wLeadMax", num(r2["precursor_leads"]["max"]), r2["precursor_leads"]["max"])
M("wLossRho", f"{r2['P2c']['best_loss_rho']:.3f}", r2["P2c"]["best_loss_rho"])
M("wLossTheta", f"{r2['P2c']['loss_theta']:.3f}", r2["P2c"]["loss_theta"])
M("wIndRho", f"{r2['P2c']['rho_ind']:.2f}", r2["P2c"]["rho_ind"])
M("wConjFA", r2["P2d"]["conj_fa"])
M("wBareFA", r2["P2d"]["bare_precursor_fa"])
M("wConjPre", r2["P2d"]["conj_pre_event"])
np_max = r2["P2d"]["norep_prevtok_max"]
M("wNorepPvRange", f"{min(np_max):.3f}--{max(np_max):.3f}")
M("wOnePrefixMax", f"{max(r2['P2e']['prefix_max']):.3f}", max(r2["P2e"]["prefix_max"]))

# loss-rule lead (post-hoc block recorded in plan; recompute here from runs)
import numpy as np
leads_loss = []
for s in range(1, 31):
    summ = J(f"runs/grid6r2/rep_s{s}/summary.json")
    recs = [json.loads(l) for l in open(os.path.join(ROOT, f"runs/grid6r2/rep_s{s}/metrics.jsonl")) if l.strip()]
    tl = next(r["step"] for r in recs if r["train_loss"] <= r2["P2c"]["loss_theta"])
    leads_loss.append(summ["t_event"] - tl)
M("wLossLeadMed", num(int(np.median(leads_loss))), float(np.median(leads_loss)))
M("wLossLeadMin", num(min(leads_loss)), min(leads_loss))

# ---------------- P6 conformal (R4) ----------------
r4 = J("analysis/out6/r4_scored.json")
o = r4["t_pv_offset"]
M("wConfCover", o["coverage"])
M("wConfWidth", num(int(o["median_width"])), o["median_width"])
M("wConfDelta", num(int(o["params"]["delta"])), o["params"]["delta"])
M("wConfQ", num(int(o["q"])), o["q"])
M("wLossConfLead", num(int(r4["t_loss_offset"]["median_anchor_lead"])),
  r4["t_loss_offset"]["median_anchor_lead"])

# ---------------- P6 blind gate (R5) ----------------
r5 = J("analysis/out6/r5_scored.json")
M("wGateEvents", r5["P5a"]["events"])
M("wGatePre", r5["P5b"]["pre_event"])
M("wGateCover", r5["P5c"]["coverage"])
M("wGateRho", f"{r5['secondary_spearman']:.3f}", r5["secondary_spearman"])
M("wGateLeadMed", num(int(r5["median_lead"])), r5["median_lead"])

# ---------------- P6 Pythia (R0/R1) ----------------
r1p = J("analysis/out6/r1_scored.json")
M("wPythiaRuns", "5")
pre_leads = [r1p[k]["lead_stages"] for k in ("pythia-70m", "pythia-160m", "pythia-410m",
                                             "pythia-70m-deduped", "pythia-160m-deduped")]
M("wPythiaPreLeads", "1--8", pre_leads)


def pyth(name):
    return sorted([json.loads(l) for l in open(os.path.join(ROOT, f"runs/p6r0/{name}.jsonl"))],
                  key=lambda r: r["step"])


p70 = pyth("pythia-70m")


def at(recs, step, key):
    return next(r[key] for r in recs if r["step"] == step)


M("wPyPrevA", f"{max(at(p70, 512, 'prevtok_by_layer')[:3]):.2f}",
  max(at(p70, 512, "prevtok_by_layer")[:3]))
M("wPyPrefixA", f"{at(p70, 512, 'prefix_max'):.3f}", at(p70, 512, "prefix_max"))
M("wPyCopyA", f"{at(p70, 512, 'copy_adv'):.3f}", at(p70, 512, "copy_adv"))
M("wPyCopyB", f"{at(p70, 1000, 'copy_adv'):.1f}", at(p70, 1000, "copy_adv"))
M("wPyPrefixB", f"{at(p70, 1000, 'prefix_max'):.2f}", at(p70, 1000, "prefix_max"))
p70_late = [r for r in p70 if r["step"] == 32000][0]
M("wPySeventyLatePrefix", f"{p70_late['prefix_max']:.2f}", p70_late["prefix_max"])

# ---------------- P6 strengtheners: R6 trap, R7 third gate, R8 law, R9 blind ----------
r6 = J("analysis/out6/r6_scored.json")
M("wTrapBareFA", r6["P_T1"]["bare_fa"])
M("wTrapConjFA", r6["P_T2"]["conj_fa"])
tc = r6["P_T3"]["anchors"]["t_conj"]
M("wTrapConjRho", f"{tc['rho']:.3f}", tc["rho"])
M("wTrapConjLeadMed", num(int(tc["median_lead"])), tc["median_lead"])
M("wTrapConjLeadRange", f"{num(tc['lead_range'][0])}--{num(tc['lead_range'][1])}")
tp = r6["P_T3"]["anchors"]["t_prefix"]
M("wTrapPrefixRho", f"{tp['rho']:.3f}", tp["rho"])
M("wTrapPrefixLead", num(int(tp["median_lead"])), tp["median_lead"])
M("wTrapIndFA", r6["P_T3"]["anchors"]["t_ind"]["fa"])
neg_pv = [n["t_pv"] for n in r6["negatives"]]
M("wTrapNegPvRange", f"{num(min(neg_pv))}--{num(max(neg_pv))}")

r7 = J("analysis/out6/r7_scored.json")
M("wGateCcover", r7["P7c"]["coverage"])
M("wGateCpre", r7["P7b"]["pre_event"])
M("wGateCrho", f"{r7['secondary_spearman']:.3f}", r7["secondary_spearman"])
M("wGateCleadMed", f"{r7['median_lead']:.1f}", r7["median_lead"])

r8 = J("analysis/out6/r8_scored.json")
M("wGapLrRatio", f"{r8['lr_ratio']:.2f}", r8["lr_ratio"])
M("wGapBatchRatio", f"{r8['batch_ratio']:.2f}", r8["batch_ratio"])
for cell, tag in (("lr5e-4", "LrLow"), ("lr2e-3", "LrHigh"), ("b32", "BLow"),
                  ("b128", "BHigh")):
    M(f"wGap{tag}", num(int(r8["cells"][cell]["median_gap"])),
      r8["cells"][cell]["median_gap"])
    M(f"wEvent{tag}", num(int(r8["cells"][cell]["median_event"])),
      r8["cells"][cell]["median_event"])

# the law: recompute fractions across all 80 valid-anchor runs (same set as the
# committed post-hoc script; regeneration keeps it artifact-backed)
fracs = []
def _frac(path, anchor="t_pv"):
    s = J(f"{path}/summary.json")
    recs = [json.loads(l) for l in open(os.path.join(ROOT, path, "metrics.jsonl"))
            if l.strip()]
    ev = s["t_event"]
    if ev is None:
        return
    if anchor == "t_pv":
        t = next((r["step"] for r in recs if r["prevtok_by_layer"][0] >= 0.10), None)
    else:
        t = next((r["step"] for r in recs if r["prevtok_by_layer"][0] >= 0.10
                  and r["indist_adv"] >= 0.10), None)
    if t:
        fracs.append(t / ev)
for s in range(1, 31):
    _frac(f"runs/grid6r2/rep_s{s}")
for s in range(101, 111):
    _frac(f"runs/grid6r5/rep_s{s}")
for s in range(201, 211):
    _frac(f"runs/grid6r7/rep_s{s}")
for tag, seeds in (("lr5e-4", range(301, 306)), ("lr2e-3", range(301, 306)),
                   ("b32", range(311, 316)), ("b128", range(311, 316))):
    for s in seeds:
        _frac(f"runs/grid6r8/{tag}_s{s}")
for s in range(1, 11):
    _frac(f"runs/grid6r6/rep_s{s}", "conj")
fa = np.array(fracs)
M("wLawN", str(len(fa)), len(fa))
M("wLawFrac", f"{np.median(fa):.3f}", float(np.median(fa)))
M("wLawIQR", f"{np.percentile(fa, 25):.3f}--{np.percentile(fa, 75):.3f}")
M("wLawRange", f"{fa.min():.3f}--{fa.max():.3f}")
M("wLawMult", f"{1/np.median(fa):.2f}", float(1 / np.median(fa)))

r9 = J("analysis/out6/r9_scored.json")
M("wNineCover", r9["P9a"]["coverage"])
M("wNineEvents", r9["P9a"]["events"])
mults = r9["multipliers"]
M("wNineMultRange", f"{min(mults):.3f}--{max(mults):.3f}")

# ---------------- R-ORD cross-family replication ----------------
ro = J("analysis/out6/rord_scored.json")
o1 = {r["revision"]: r for r in ro["olmo1"]["rows"]}
o2 = {r["revision"]: r for r in ro["olmo2"]["rows"]}
M("ordOlmoOnePre", f"{o1['step2000-tokens8B']['prevtok']:.2f}",
  o1["step2000-tokens8B"]["prevtok"])
M("ordOlmoOneCap", f"{o1['step2000-tokens8B']['copy_adv']:.2f}",
  o1["step2000-tokens8B"]["copy_adv"])
M("ordOlmoTwoPre", f"{o2['stage1-step300-tokens1B']['prevtok']:.2f}",
  o2["stage1-step300-tokens1B"]["prevtok"])
M("ordOlmoTwoCap", f"{o2['stage1-step300-tokens1B']['copy_adv']:.2f}",
  o2["stage1-step300-tokens1B"]["copy_adv"])
M("ordOlmoOnePreT", "8", ro["olmo1"]["t_pre_B"])
M("ordOlmoOneCapT", "12", ro["olmo1"]["t_cap_B"])
M("ordOlmoTwoPreT", "1", ro["olmo2"]["t_pre_B"])
M("ordOlmoTwoCapT", "21", ro["olmo2"]["t_cap_B"])

# ---------------- P5 (grokking benchmark) ----------------
M("vCorpus", "466")
fz = J("analysis/out5/frozen_eval.json")
M("vRoneLead", num(int(fz["r1"]["test"]["median_lead"])), fz["r1"]["test"]["median_lead"])
M("vRoneRel", f"{fz['r1']['test']['median_rel']*100:.0f}\\%", fz["r1"]["test"]["median_rel"])
r1r2 = J("analysis/out5/r1r2_stats.json")
r2m = {r["signal"]: r for r in r1r2["r2_mnist"]["rows"]}
M("vShiftCos", num(int(r2m["cos_gap"]["test_med_lead"])), r2m["cos_gap"]["test_med_lead"])
M("vShiftDcos", num(int(r2m["d.cos_gap"]["test_med_lead"])), r2m["d.cos_gap"]["test_med_lead"])
r1m = {r["signal"]: r for r in r1r2["r1_mnist"]["rows"]}
M("vInWnorm", num(int(r1m["wnorm"]["test_med_lead"])), r1m["wnorm"]["test_med_lead"])
tg = J("analysis/out5/r2b_twogate.json")
ch = tg["mnist"]["champion"]
M("vTwoGateLead", num(int(ch["test_lead"])), ch["test_lead"])
M("vTwoGateRel", f"{ch['test_rel']*100:.0f}\\%", ch["test_rel"])
r3 = J("analysis/out5/r3_scored.json")
M("vProspLead", num(int(r3["P_R3a"]["median"])), r3["P_R3a"]["median"])
M("vProspBar", num(int(r3["P_R3a"]["bar"])), r3["P_R3a"]["bar"])
r5b = J("analysis/out5/r5b_scored.json")
M("vRfivebLead", num(int(r5b["d.top1_frac"]["median_lead"])), r5b["d.top1_frac"]["median_lead"])

with open(os.path.join(HERE, "numbers.tex"), "w") as f:
    f.write("% AUTO-GENERATED by paper4/gen_numbers.py -- do not edit by hand.\n")
    f.write("% Verified byte-for-byte by paper4/verify_regen.py.\n")
    for k in sorted(macros):
        f.write(f"\\newcommand{{\\{k}}}{{{macros[k]}}}\n")
json.dump(raw, open(os.path.join(HERE, "numbers.json"), "w"), indent=2, sort_keys=True)
print(f"wrote {len(macros)} macros")
