"""Generate paper/numbers.tex from results JSONs — every number in the paper comes
from here, never hand-typed. Macros expand to BARE math content (no $); prose uses
them inside $...$; generated table rows wrap their own cells."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).parent / "numbers.tex"

E0 = json.loads((RES / "exp0_existence.json").read_text())
AM = json.loads((RES / "amended_gate.json").read_text())

L = []


def cmd(name, body):
    L.append(f"\\newcommand{{\\{name}}}{{{body}}}")


def pm(vals, d=3):
    vals = np.asarray(vals, dtype=float)
    return f"{vals.mean():.{d}f} \\pm {vals.std():.{d}f}"


def f(v, d=3):
    return f"{float(v):.{d}f}"


R = E0["results"]
PAIRS = ["null", "seed", "noise", "distill", "prune", "difftask"]
PLABEL = {"null": "recoded (null)", "seed": "independent seed",
          "noise": "noise twin ($\\sigma{=}0.2$)", "distill": "distilled twin",
          "prune": "pruned twin (50\\%)", "difftask": "different task (ctrl)"}


def amended_win(r, curve="dbar_curve", wall="k_wall"):
    ok = [row for row in r[curve]
          if 2 <= row["n"] <= r[wall] and row["dbar"] >= 2 * row["floor"]]
    return max(ok, key=lambda x: x["delta"]) if ok else None


def ratios(pair, curve="dbar_curve", wall="k_wall"):
    out = []
    for r in R[pair]:
        w = amended_win(r, curve, wall)
        out.append(w["dbar"] / max(w["floor"], 1e-12) if w else 1.0)
    return np.array(out)


# ---- global setup ----------------------------------------------------------
cmd("taskH", f(E0["meta"]["task_h"], 4))
gen = E0["meta"]["gen_kw"]
n_run = gen["B"] * gen["T"]
cmd("genB", str(gen["B"]))
cmd("genT", f"{gen['T']:,}".replace(",", "\\,"))
cmd("nRun", f"{n_run/1e6:.1f}\\times10^{{6}}")
cmd("nSeeds", str(len(R["null"])))
cmd("valCEmin", f(min(E0["meta"]["val_ce"].values()), 4))
cmd("valCEmax", f(max(E0["meta"]["val_ce"].values()), 4))
cmd("sigmaStar", f(list(E0["meta"]["sigma_star"].values())[0], 1))
walls = [r["k_wall"] for r in R["noise"]]
cmd("kWall", f(np.mean(walls), 1))
cmd("beliefWall", f(np.mean([r["belief_k_wall"] for r in R["noise"]]), 1))

# noise calibration: worst-case TV and CE increase at sigma*
cal = [E0["meta"]["calib"][k] for k in E0["meta"]["calib"]]
tv_at_star = [row["tv"] for c in cal for row in c["grid"] if row["sigma"] == 0.2]
ce_rel = [(row["ce"] / c["ce0"] - 1) * 100
          for c in cal for row in c["grid"] if row["sigma"] == 0.2]
cmd("calibTVmax", f(max(tv_at_star), 4))
cmd("calibCEmaxPct", f(max(ce_rel), 1))

# ---- per-pair table (three metrics side by side) ---------------------------
rows = []
for p in PAIRS:
    cka = [r["cka"] for r in R[p]]
    dsa = [r["dsa"] for r in R[p]]
    tv1 = [r["dbar_1"] for r in R[p]]
    rat = ratios(p)
    brat = (ratios(p, "belief_curve", "belief_k_wall")
            if "belief_curve" in R[p][0] else None)
    sep = AM["pairs"][p]["n_seeds_separating"] if p in AM["pairs"] else None
    rows.append(
        f"{PLABEL[p]} & ${pm(cka)}$ & ${pm(dsa)}$ & ${pm(tv1, 4)}$ & "
        f"${pm(rat, 1)}$ & " + (f"${pm(brat, 0)}$" if brat is not None else "---")
        + " \\\\")
cmd("eZeroTableRows", "\n".join(rows))

# ---- headline per-pair macros ----------------------------------------------
for p in PAIRS:
    tag = p.capitalize()
    cmd(f"cka{tag}", pm([r["cka"] for r in R[p]]))
    cmd(f"dsa{tag}", pm([r["dsa"] for r in R[p]]))
    cmd(f"ratio{tag}", pm(ratios(p), 1))
    cmd(f"ratio{tag}Min", f(ratios(p).min(), 1))
    cmd(f"ratio{tag}Max", f(ratios(p).max(), 1))
cmd("beliefNoiseMin", f"{ratios('noise', 'belief_curve', 'belief_k_wall').min():.0f}")
cmd("beliefNoiseMax", f"{ratios('noise', 'belief_curve', 'belief_k_wall').max():.0f}")
cmd("beliefNullMax", f(ratios('null', 'belief_curve', 'belief_k_wall').max(), 2))
cmd("tvNoise", pm([r["dbar_1"] for r in R["noise"]], 4))
cmd("tvDifftask", pm([r["dbar_1"] for r in R["difftask"]], 4))
cmd("dbarDifftaskTwo", pm([next(row["dbar"] for row in r["dbar_curve"]
                                if row["n"] == 2) for r in R["difftask"]]))
cmd("fanoNoiseMax", f"{max(r['fano_lb'] for r in R['noise']):.5f}")
hs = [r["diag_a"]["h_cond"] for r in R["noise"]] + \
     [r["diag_b"]["h_cond"] for r in R["noise"]]
cmd("hHatRange", f"[{min(hs):.3f}, {max(hs):.3f}]")

cmd("tvSameTaskMax",
    f(max(r["dbar_1"] for p in ("null", "seed", "noise", "distill", "prune")
          for r in R[p]), 4))
cmd("beliefDistill", pm(ratios("distill", "belief_curve", "belief_k_wall"), 0))
cmd("beliefSeed", pm(ratios("seed", "belief_curve", "belief_k_wall"), 0))

# amended-gate numbers
cmd("nuDSA", f(AM["nu_dsa"]))
cmd("stdNullDSA", f(AM["std_null_dsa"]))
cmd("dsaSimThresh", f(AM["dsa_similar_threshold"]))
pr = AM["pairs"]["prune"]
cmd("pruneDeltaMean", f"{pr['delta_mean']:.4f}")
cmd("pruneDeltaStd", f"{pr['delta_std']:.4f}")
no = AM["pairs"]["noise"]
cmd("noiseDeltaMean", f"{no['delta_mean']:.5f}")
cmd("noiseDeltaStd", f"{no['delta_std']:.5f}")
cmd("noiseSepTwoSigma", "passes" if no["sep_2sigma"] else "fails")
kappa = float(np.mean([r["dsa"] for r in R["difftask"]]))
nu = AM["nu_dsa"]
cmd("preregDSAThresh", f(nu + 0.25 * (kappa - nu)))
cmd("kappaDSA", f(kappa))
cmd("dsaNullMin", f(min(r["dsa"] for r in R["null"])))
cmd("dsaNullMax", f(max(r["dsa"] for r in R["null"])))
cmd("dsaDifftaskMin", f(min(r["dsa"] for r in R["difftask"])))
cmd("dsaDifftaskMax", f(max(r["dsa"] for r in R["difftask"])))

# sampled-regime bias level (n=16 floors) for the amendment discussion
fl16 = [next(row["floor"] for row in r["dbar_curve"] if row["n"] == 16)
        for p in PAIRS for r in R[p]]
cmd("biasSixteen", pm(fl16))
fl8 = [next(row["floor"] for row in r["dbar_curve"] if row["n"] == 8)
       for p in PAIRS for r in R[p]]
cmd("floorEight", f"{np.mean(fl8):.4f}")

# convergence study (V3)
conv = E0["convergence"]
cmd("convPair", conv["pair"])
last = conv["curves"][-1]
n8 = next(r for r in last["rows"] if r["n"] == 8)
cmd("convNEightDelta", f"{n8['delta']:.4f}")
n32 = next(r for r in last["rows"] if r["n"] == 32)
cmd("convNThirtytwoDelta", f"{n32['delta']:.4f}")

# ---- E1 (guarded) -----------------------------------------------------------
E1P = RES / "exp1_generality.json"
if E1P.exists():
    E1 = json.loads(E1P.read_text())

    def bykey(rows, knob, k):
        return [r for r in rows if r[knob] == k]

    def am_ratio(r, curve="dbar_curve", wall="k_wall"):
        w = amended_win(r, curve, wall)
        return w["dbar"] / max(w["floor"], 1e-12) if w else 1.0

    nrows, prows = [], []
    for sig in E1["sigmas"]:
        rs = bykey(E1["noise"], "sigma", sig)
        nrows.append(
            f"${sig}$ & ${pm([r['cka'] for r in rs])}$ & "
            f"${pm([r['dsa'] for r in rs])}$ & "
            f"${pm([am_ratio(r) for r in rs], 1)}$ & "
            f"${pm([am_ratio(r, 'belief_curve', 'belief_k_wall') for r in rs], 0)}$ \\\\")
    for fr in E1["fracs"]:
        rs = bykey(E1["prune"], "frac", fr)
        prows.append(
            f"${fr}$ & ${pm([r['cka'] for r in rs])}$ & "
            f"${pm([r['dsa'] for r in rs])}$ & "
            f"${pm([am_ratio(r) for r in rs], 1)}$ & "
            f"${pm([r['ce'] for r in rs], 4)}$ \\\\")
    cmd("eOneNoiseRows", "\n".join(nrows))
    cmd("eOnePruneRows", "\n".join(prows))

    # smallest sigma with all-seed emitted separation; sigma where DSA leaves null
    thr = AM["dsa_similar_threshold"]
    sig_sep = [s for s in E1["sigmas"] if s > 0 and all(
        amended_win(r) is not None for r in bykey(E1["noise"], "sigma", s))]
    cmd("sigmaSepMin", str(min(sig_sep)) if sig_sep else "none")
    sig_dsa = [s for s in E1["sigmas"] if s > 0 and
               np.mean([r["dsa"] for r in bykey(E1["noise"], "sigma", s)]) > thr]
    cmd("sigmaDSAwake", str(min(sig_dsa)) if sig_dsa else "none")
    # belief ratio at the smallest separating sigma
    if sig_sep:
        s0 = min(sig_sep)
        rs = bykey(E1["noise"], "sigma", s0)
        cmd("beliefAtSigmaSepMin",
            f"{np.mean([am_ratio(r, 'belief_curve', 'belief_k_wall') for r in rs]):.0f}")
        cmd("dsaAtSigmaSepMin", pm([r["dsa"] for r in rs]))
        cmd("emittedAtSigmaSepMin", pm([am_ratio(r) for r in rs], 1))

# ---- E2 (guarded) -----------------------------------------------------------
E2P = RES / "exp2_transformer.json"
if E2P.exists():
    E2 = json.loads(E2P.read_text())
    trows = []
    T2LABEL = {("gm", "noise"): "GM, noise twin",
               ("gm", "distill"): "GM, distilled twin",
               ("mess3", "noise"): "MESS3, noise twin",
               ("mess3", "distill"): "MESS3, distilled twin"}
    for t, p in [("gm", "noise"), ("gm", "distill"),
                 ("mess3", "noise"), ("mess3", "distill")]:
        rs = E2[t][p]
        w = [amended_win(r) for r in rs]
        rat = [x["dbar"] / max(x["floor"], 1e-12) if x else 1.0 for x in w]
        trows.append(
            f"{T2LABEL[(t, p)]} & ${pm([r['cka'] for r in rs])}$ & "
            f"${pm([r['dsa'] for r in rs])}$ & ${pm([r['dbar_1'] for r in rs], 4)}$ & "
            f"${pm(rat, 1)}$ \\\\")
    cmd("eTwoTableRows", "\n".join(trows))
    cmd("crossCKA", pm([c["cka"] for c in E2["cross"]]))
    cmd("crossDSA", pm([c["dsa"] for c in E2["cross"]]))
    sigs = {k: v["sigma"] for k, v in E2["meta"]["sigma_star"].items()}
    cmd("tfSigmaStarGM", f(sigs.get("gm_s0", 0), 2))
    cmd("tfSigmaStarMessThree", f(sigs.get("mess3_s0", 0), 2))
    e2rat_noise = []
    for r in E2["gm"]["noise"]:
        w = amended_win(r)
        e2rat_noise.append(w["dbar"] / max(w["floor"], 1e-12) if w else 1.0)
    cmd("tfNoiseRatioGM", pm(e2rat_noise, 1))

OUT.write_text("% AUTO-GENERATED by gen_paper_numbers.py -- do not edit by hand\n"
               + "\n".join(L) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({len(L)} macros)")
