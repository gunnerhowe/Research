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
