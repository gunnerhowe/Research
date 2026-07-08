"""Generate paper/numbers.tex and paper/tables/*.tex from results/*.json
(house rule: every number in the paper is machine-generated)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "numbers.tex"
TABLES = ROOT / "paper" / "tables"

M = {}


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


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


def emit_main(tag, res, prefix):
    rep = res["outcomes"]["R_TE"]
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


def table_models(tags):
    rows = []
    for tag, fname in tags:
        res = load(fname)
        if res is None:
            continue
        rep = res["outcomes"]["R_TE"]
        est = rep["two_step"]["estimands"]
        tgt = rep["targets"]
        g = rep["gate"]
        lab = MODEL_LABELS.get(tag, tag)
        rows.append(
            f"{lab} & {rep['n_fit']} & {pct(rep['verbalization_rate'])} & "
            f"{num(rep['mle']['rho'])} "
            f"[{num(rep['rho_wald']['rho_ci95'][0])}, "
            f"{num(rep['rho_wald']['rho_ci95'][1])}] & "
            f"{pval(rep['rho_lr']['p'])} & "
            f"{num(est['naive_selected'])} & {num(est['corrected_pop'])} & "
            f"{num(tgt['true_pop'])} & "
            f"{'\\checkmark' if g['corrected_beats_naive_selected'] else '$\\times$'} \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{lrrrrrrrc}\n\\toprule\n"
        "Model & $n$ & $\\Pr(V)$ & $\\hat\\rho$ [95\\% CI] & LR test & "
        "$\\hat\\mu_{\\text{naive}}$ & $\\hat\\mu_{\\text{corr}}$ & "
        "$\\mu_{\\text{true}}$ & corr.\\ wins \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")


def table_hints(res):
    rep = res["outcomes"]["R_TE"]
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


def table_sensitivity(res):
    rep = res["outcomes"]["R_TE"]
    tgt = rep["targets"]
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


def main():
    TABLES.mkdir(exist_ok=True)

    qwen = load("qwen7b_e3.json") or load("qwen7b_e0.json")
    if qwen:
        emit_main("qwen7b", qwen, "Q")
        emit_e1(qwen, "Q")
        emit_placebo(qwen, "Q")
        # secondary outcomes on primary model
        for oc, pre in (("R_NDE", "Qnde"), ("R_pre", "Qpre")):
            if oc in qwen["outcomes"]:
                rep = qwen["outcomes"][oc]
                put(pre + "Rho", num(rep["mle"]["rho"]))
                put(pre + "RhoLRp", pval(rep["rho_lr"]["p"]))
                put(pre + "CorrPop",
                    num(rep["two_step"]["estimands"]["corrected_pop"]))
                put(pre + "TruePop", num(rep["targets"]["true_pop"]))
                put(pre + "NaiveSel",
                    num(rep["two_step"]["estimands"]["naive_selected"]))
        (TABLES / "hints.tex").write_text(table_hints(qwen))
        (TABLES / "sensitivity.tex").write_text(table_sensitivity(qwen))

    mist = load("mistral7b_e3.json")
    if mist:
        emit_main("mistral7b", mist, "Mi")
        emit_placebo(mist, "Mi")
    phi = load("phi35_e3.json")
    if phi:
        emit_main("phi35", phi, "Ph")
        emit_placebo(phi, "Ph")

    (TABLES / "models.tex").write_text(table_models([
        ("qwen7b", "qwen7b_e3.json"), ("mistral7b", "mistral7b_e3.json"),
        ("phi35", "phi35_e3.json")]))

    nemo = load("nemotron8b_e2.json")
    if nemo:
        emit_heckprob(nemo, "Ne")
        if "outcomes" in nemo and "R_TE" in nemo.get("outcomes", {}):
            emit_main("nemotron8b", nemo, "NeFull")
    claude = load("claude_e2.json")
    if claude:
        emit_heckprob(claude, "Cl")

    judge = load("judge_v_qwen7b.json")
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
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} with {len(M)} macros + tables")


if __name__ == "__main__":
    main()
