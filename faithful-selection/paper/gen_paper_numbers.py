"""Generate paper/numbers.tex and paper/tables/*.tex from results/*.json
(house rule: every number in the paper is machine-generated)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "numbers.tex"
TABLES = ROOT / "paper" / "tables"
sys.path.insert(0, str(ROOT / "src"))

M = {}


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def detector_vs_judge(raw_name, vjudge_name):
    """Detector V-rate vs validated judge V-rate on the same hinted instances,
    plus the detector's precision against the judge. Machine-generated so the
    over-count claim is reproducible."""
    import warnings
    warnings.filterwarnings("ignore")
    from faithsel.analysis import load_rows, augment
    raw = RESULTS / "raw" / raw_name
    vj = RESULTS / vjudge_name
    if not raw.exists() or not vj.exists():
        return None
    judge = {k: int(v) for k, v in json.loads(vj.read_text()).items()}
    df = augment(load_rows(str(raw)))               # detector V
    d = df[df["parse_ok"] & (df["hint_type"] != "placebo")]
    d = d[d["qid"].isin(judge)]
    det = d.set_index("qid")["V"].to_dict()
    common = [q for q in det if q in judge]
    det_rate = sum(det[q] for q in common) / len(common)
    jud_rate = sum(judge[q] for q in common) / len(common)
    tp = sum(1 for q in common if det[q] == 1 and judge[q] == 1)
    fp = sum(1 for q in common if det[q] == 1 and judge[q] == 0)
    fn = sum(1 for q in common if det[q] == 0 and judge[q] == 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(common), "det_rate": det_rate, "judge_rate": jud_rate,
            "overcount": det_rate / jud_rate if jud_rate else float("inf"),
            "precision": prec, "recall": rec}


def num(x, d=2):
    return f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100.0 * x:.{d}f}\\%"


def pval(p):
    if p < 1e-4:
        return "p<10^{-4}"
    return f"p={p:.3g}"


def put(key, val):
    M[key] = str(val)


MODEL_LABELS = {
    "qwen7b": "Qwen2.5-7B-Instruct",
    "mistral7b": "Mistral-7B-Instruct-v0.3",
    "phi35": "Phi-3.5-mini-instruct",
    "nemotron8b": "Nemotron-Nano-8B",
}
HINT_LABELS = {
    "sycophancy": "Sycophancy", "authority": "Authority",
    "metadata": "Metadata", "consistency": "Consistency",
    "placebo": "Placebo",
}


def emit_main(tag, res, prefix, outcome="R_TE"):
    rep = res["outcomes"][outcome]
    est = rep["two_step"]["estimands"]
    tgt = rep["targets"]
    boot = rep["bootstrap"]
    put(prefix + "N", rep["n_fit"])
    put(prefix + "Vrate", pct(rep["verbalization_rate"]))
    put(prefix + "FlipRate", pct(rep["flip_rate"]))
    put(prefix + "TurpinFaith",
        pct(rep["turpin_naive_faithfulness_P(V|flip)"]))
    put(prefix + "RhoTS", num(rep["two_step"]["rho"]))
    put(prefix + "Rho", num(rep["mle"]["rho"]))
    put(prefix + "RhoLo", num(rep["rho_wald"]["rho_ci95"][0]))
    put(prefix + "RhoHi", num(rep["rho_wald"]["rho_ci95"][1]))
    put(prefix + "RhoLRp", pval(rep["rho_lr"]["p"]))
    put(prefix + "RhoWaldp", pval(rep["rho_wald"]["p"]))
    put(prefix + "NaiveSel", num(est["naive_selected"]))
    put(prefix + "NaiveZero", num(est["naive_zerofill"]))
    put(prefix + "CorrPop", num(est["corrected_pop"]))
    put(prefix + "CorrHidden", num(est["corrected_hidden"]))
    put(prefix + "TruePop", num(tgt["true_pop"]))
    put(prefix + "TrueHidden", num(tgt["true_hidden"]))
    put(prefix + "TrueSel", num(tgt["true_selected"]))
    g = rep["gate"]
    put(prefix + "ErrNaiveSel", num(g["err_naive_selected"]))
    put(prefix + "ErrNaiveZero", num(g["err_naive_zerofill"]))
    put(prefix + "ErrCorr", num(g["err_corrected_pop"]))
    put(prefix + "ErrHidNaive", num(g["err_hidden_naive0"]))
    put(prefix + "ErrHidCorr", num(g["err_hidden_corrected"]))
    for bk, mk in [("rho", "RhoB"), ("corrected_pop", "CorrPopB"),
                   ("corrected_hidden", "CorrHiddenB"),
                   ("naive_selected", "NaiveSelB")]:
        if bk in boot:
            put(prefix + mk + "Lo", num(boot[bk]["lo95"]))
            put(prefix + mk + "Hi", num(boot[bk]["hi95"]))
    fs = rep["gamma_z_first_stage"]
    put(prefix + "FirstStageZ", num(fs["z_stat"], 1))
    put(prefix + "FirstStageP", pval(fs["p"]))


def emit_e1(res, prefix):
    b = res["e1_balance"]
    put(prefix + "VzOne", pct(b["V_rate_z1"]))
    put(prefix + "VzZero", pct(b["V_rate_z0"]))
    put(prefix + "FsZ", num(b["first_stage"]["z_stat"], 1))
    for oc in ("R_TE", "R_NDE", "R_pre"):
        if oc in b:
            key = oc.replace("_", "")
            put(prefix + f"Bal{key}SD", num(b[oc]["std_diff"], 3))
            put(prefix + f"Bal{key}P", pval(b[oc]["p"]))
            put(prefix + f"Bal{key}T", num(b[oc]["welch_t"], 2))


def emit_placebo(res, prefix):
    p = res.get("placebo", {})
    if not p or p.get("n", 0) == 0:
        return
    put(prefix + "PlacN", p["n"])
    put(prefix + "PlacV", pct(p["V_rate"]))
    put(prefix + "PlacR", num(p["mean_R"], 3))
    put(prefix + "PlacRLo", num(p["R_ci95"][0], 3))
    put(prefix + "PlacRHi", num(p["R_ci95"][1], 3))
    if "rho_lr_p" in p:
        put(prefix + "PlacRhoP", pval(p["rho_lr_p"]))
        put(prefix + "PlacRho", num(p["rho_mle"]))


def emit_heckprob(res, prefix):
    hp = res.get("heckprob")
    if not hp:
        return
    put(prefix + "HpN", hp["n_fit"])
    put(prefix + "HpV", pct(hp["verbalization_rate"]))
    put(prefix + "HpRho", num(hp["rho"]))
    put(prefix + "HpRhoLo", num(hp["rho_tests"]["rho_ci95"][0]))
    put(prefix + "HpRhoHi", num(hp["rho_tests"]["rho_ci95"][1]))
    put(prefix + "HpLRp", pval(hp["rho_tests"]["lr_p"]))
    put(prefix + "HpNaive", pct(hp["naive_adoption_among_verbalizers"]))
    put(prefix + "HpCorrPop", pct(hp["corrected_pop_adoption"]))
    put(prefix + "HpCorrHidden", pct(hp["corrected_hidden_adoption"]))
    put(prefix + "HpTrueHidden", pct(hp["true_hidden_adoption_unblind"]))
    put(prefix + "HpTruePop", pct(hp["true_pop_adoption_unblind"]))


MEAS = {"nemotron8b": ("nemotron8b_e0.jsonl", "vjudge_nemotron8b_e0.json"),
        "qwen7b": ("qwen7b_e3.jsonl", "vjudge_qwen7b_e3.json"),
        "phi35": ("phi35_e3.jsonl", "vjudge_phi35_e3.json")}


def table_models(tags, outcome="R_NDE"):
    rows = []
    for tag, fname in tags:
        res = load(fname) if fname else None
        lab = MODEL_LABELS.get(tag, tag)
        dj = detector_vs_judge(*MEAS[tag]) if tag in MEAS else None
        meas = (f"{pct(dj['det_rate'])} & {pct(dj['judge_rate'])} & "
                f"{num(dj['overcount'], 1)}$\\times$" if dj else " & & ")
        if res is None:
            rows.append(f"{lab} & --- & --- & --- & --- & --- & --- & {meas} \\\\")
            continue
        rep = res["outcomes"].get(outcome) or res["outcomes"]["R_TE"]
        est = rep["two_step"]["estimands"]
        tgt = rep["targets"]
        g = rep["gate"]
        rows.append(
            f"{lab} & {num(rep['mle']['rho'])} & {pval(rep['rho_lr']['p'])} & "
            f"{num(est['naive_selected'])} & {num(est['corrected_pop'])} & "
            f"{num(tgt['true_pop'])} & "
            f"{'\\checkmark' if g['corrected_beats_naive_selected'] else '$\\times$'} & "
            f"{meas} \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{lrrrrrc|rrr}\n\\toprule\n"
        "& \\multicolumn{6}{c}{estimation (judge-$V$, $R^{\\mathrm{NDE}}$)} & "
        "\\multicolumn{3}{c}{verbalization measurement} \\\\\n"
        "Model & $\\hat\\rho$ & LR & "
        "$\\hat\\mu_{\\text{nv}}$ & $\\hat\\mu_{\\text{cr}}$ & "
        "$\\mu_{\\text{tr}}$ & wins & "
        "det.\\ & judge & over \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")


def table_hints(res, outcome="R_TE"):
    rep = res["outcomes"].get(outcome) or res["outcomes"]["R_TE"]
    rows = []
    for ht, sub in sorted(rep.get("per_hint", {}).items()):
        if "error" in sub:
            continue
        est = sub["estimands"]
        tgt = sub["targets"]
        rows.append(
            f"{HINT_LABELS.get(ht, ht)} & {sub['n']} & {pct(sub['V_rate'])} & "
            f"{num(sub['rho_mle'])} & {pval(sub['rho_lr_p'])} & "
            f"{num(est['naive_selected'])} & {num(est['corrected_pop'])} & "
            f"{num(tgt['true_pop'])} \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
        "Hint type & $n$ & $\\Pr(V)$ & $\\hat\\rho$ & LR test & "
        "$\\hat\\mu_{\\text{naive}}$ & $\\hat\\mu_{\\text{corr}}$ & "
        "$\\mu_{\\text{true}}$ \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")


def table_sensitivity(res, outcome="R_TE"):
    rep = res["outcomes"].get(outcome) or res["outcomes"]["R_TE"]
    tgt = rep["targets"]
    if not rep.get("rho_sensitivity"):
        return STUB_TABLE
    rows = []
    for r in rep["rho_sensitivity"]:
        rows.append(f"{num(r['rho_fixed'], 1)} & {num(r['corrected_pop'])} & "
                    f"{num(r['corrected_hidden'])} \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{rrr}\n\\toprule\n"
        "fixed $\\rho$ & $\\hat\\mu_{\\text{corr}}$ & "
        "$\\hat\\mu_{0,\\text{corr}}$ \\\\\n\\midrule\n"
        + body +
        f"\n\\midrule\ntruth & {num(tgt['true_pop'])} & "
        f"{num(tgt['true_hidden'])} \\\\\n\\bottomrule\n\\end{{tabular}}\n")


STUB_TABLE = "\\begin{tabular}{l}\\toprule (results pending) \\\\\\bottomrule\\end{tabular}\n"


def main():
    TABLES.mkdir(exist_ok=True)
    for t in ("models.tex", "hints.tex", "sensitivity.tex"):
        p = TABLES / t
        if not p.exists():
            p.write_text(STUB_TABLE)

    # PRIMARY = Nemotron (PLAN amendment A5). The identified primary OUTCOME is
    # R_NDE (direct effect): the disclosure instrument satisfies the exclusion
    # restriction there (balance test) but not for the total effect R_TE.
    # Prefix "Q" = primary/R_NDE namespace; "Qte" = R_TE (exclusion-violated).
    primary = load("nemotron8b_e0.json")
    if primary:
        emit_main("nemotron8b", primary, "Q", outcome="R_NDE")
        emit_e1(primary, "Q")
        emit_placebo(primary, "Q")
        # observation-only heckprob on the primary (the unblind bridge)
        emit_heckprob(primary, "Ne")
        # the primary's own full R_NDE fit doubles as the NeFull cross-check
        emit_main("nemotron8b", primary, "NeFull", outcome="R_NDE")
        # total effect (exclusion-violated) and pre-commitment outcomes
        for oc, pre in (("R_TE", "Qte"), ("R_pre", "Qpre")):
            if oc in primary["outcomes"]:
                rep = primary["outcomes"][oc]
                put(pre + "Rho", num(rep["mle"]["rho"]))
                put(pre + "RhoLRp", pval(rep["rho_lr"]["p"]))
                put(pre + "CorrPop",
                    num(rep["two_step"]["estimands"]["corrected_pop"]))
                put(pre + "TruePop", num(rep["targets"]["true_pop"]))
                put(pre + "NaiveSel",
                    num(rep["two_step"]["estimands"]["naive_selected"]))

    phi = load("phi35_e3.json")
    if phi:
        emit_main("phi35", phi, "Ph", outcome="R_NDE")
        emit_placebo(phi, "Ph")

    # detector-vs-judge measurement block (the over-count finding)
    for tag, raw, vj in (("Ne", "nemotron8b_e0.jsonl", "vjudge_nemotron8b_e0.json"),
                         ("Qw", "qwen7b_e3.jsonl", "vjudge_qwen7b_e3.json"),
                         ("Ph", "phi35_e3.jsonl", "vjudge_phi35_e3.json")):
        dj = detector_vs_judge(raw, vj)
        if dj:
            put(tag + "DetRate", pct(dj["det_rate"]))
            put(tag + "JudgeRate", pct(dj["judge_rate"]))
            put(tag + "Overcount", num(dj["overcount"], 1))
            put(tag + "DetPrec", num(dj["precision"], 2))
            put(tag + "MeasN", dj["n"])
    # Qwen judge-V rate (unfittable: too few verbalizers)
    qwen_vj = load("vjudge_qwen7b_e3.json")
    if qwen_vj:
        vals = list(qwen_vj.values())
        put("QwJudgeVerb", sum(vals))
        put("QwJudgeN", len(vals))

    (TABLES / "models.tex").write_text(table_models([
        ("nemotron8b", "nemotron8b_e0.json"),
        ("phi35", "phi35_e3.json"),
        ("qwen7b", None)]))

    claude = load("claude_e2.json")
    if claude:
        emit_heckprob(claude, "Cl")

    judge = load("judge_v_nemotron8b.json")
    if judge:
        put("judgeAgree", pct(judge["agreement"]))
        put("judgeKappa", num(judge["kappa"]))
        put("judgeN", judge["n_judged"])
    else:
        put("judgeAgree", "TBD")
        put("judgeKappa", "TBD")

    # aliases used in the abstract
    if "QRho" in M:
        put("rhoMain", M["QRho"])
        put("rhoLRpMain", "$" + M["QRhoLRp"] + "$")

    lines = ["% AUTO-GENERATED by gen_paper_numbers.py -- do not edit"]
    for k in sorted(M):
        lines.append(f"\\newcommand{{\\Q{k}}}{{{M[k]}}}")

    # Safety net: back-fill a visible placeholder for any \Q<name> macro the
    # paper references but that this run did not produce (e.g. a results file
    # not yet present). Guarantees the paper compiles and makes gaps obvious.
    import re
    main_tex = (ROOT / "paper" / "main.tex").read_text()
    referenced = set(re.findall(r"\\Q([A-Za-z]+)", main_tex))
    missing = sorted(referenced - set(M))
    for k in missing:
        lines.append(f"\\newcommand{{\\Q{k}}}{{\\textbf{{??}}}}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} with {len(M)} macros + tables"
          + (f"; {len(missing)} placeholders: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
