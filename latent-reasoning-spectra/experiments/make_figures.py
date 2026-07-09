"""Generate all paper figures (paper/figs/*.pdf) and paper/numbers.tex from
results/*.json and runs/*.npz.  Idempotent; skips figures whose inputs are missing."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrspec.paths import DATA, FIGS, RESULTS, ROOT  # noqa: E402

import os as _os
RUNS = Path(_os.environ.get("LRSPEC_RUNS", ROOT / "runs"))
NUM_PATH = Path(_os.environ.get("LRSPEC_NUMBERS", ROOT / "paper" / "numbers.tex"))
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

C_BRANCH = "#c0392b"
C_NON = "#2980b9"
C_M2 = "#8e44ad"
C_M3 = "#16a085"
C_M4 = "#f39c12"


def _load(name):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else None


def _npz(name):
    p = RUNS / name
    return np.load(p) if p.exists() else None


def _inv(cap, key):
    return cap["inv"][:, :, [str(k) for k in cap["scalar_keys"]].index(key)]


# ---------------------------------------------------------------- numbers.tex

def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if isinstance(x, float):
        if abs(x) >= 100:
            return f"{x:.0f}"
        if abs(x) >= 10:
            return f"{x:.1f}"
        return f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return str(x)


def fmt_p(p):
    if p is None:
        return "--"
    if p < 1e-300:
        return "<10^{-300}"
    if p >= 0.001:
        return f"{p:.3f}".rstrip("0").rstrip(".")
    exp = int(np.floor(np.log10(p)))
    mant = p / 10 ** exp
    return f"{mant:.1f}\\times10^{{{exp}}}"


def fmt_ci(ci):
    return f"[{fmt(ci[0])}, {fmt(ci[1])}]"


def numbers():
    macros = {}

    # dataset facts
    from lrspec.prosqa import branch_stats, load_problems
    ps = load_problems(DATA / "prosqa_test.json")
    st = branch_stats(ps)
    macros["branchRatePct"] = fmt(100 * st["branch_rate"], 1)
    macros["nTestProblems"] = st["n_problems"]
    macros["nPathSteps"] = st["n_path_steps"]
    macros["hopsMin"] = st["hops_min"]
    macros["hopsMax"] = st["hops_max"]

    # descriptive means of the invariants per model (from capture arrays)
    for mk, mn in [("M2", "MTwo"), ("M3", "MThree"), ("M4", "MFour")]:
        cap = _npz(f"exp0_capture_{mk}.npz")
        if cap is None:
            continue
        nn = int(cap["n_done"][0])
        keys = [str(k) for k in cap["scalar_keys"]]
        for pk, pn in [("rho", "Rho"), ("sigma1", "Sigma"), ("kappa", "Kappa"),
                       ("henrici_norm", "Henrici")]:
            vals = cap["inv"][:nn, :, keys.index(pk)]
            macros[f"mean{pn}{mn}"] = fmt(float(np.nanmean(vals)), 2)
        if mk == "M2":
            hs = cap["hs"][:nn]
            lab = cap["labels"][:nn]
            dc = np.linalg.norm(np.diff(hs, axis=1), axis=2)[:, :6]
            macros["meanDcMTwo"] = fmt(float(dc[lab >= 0].mean()), 1)

    san = _load("sanity_accuracy.json")
    if san:
        for k in ["M1", "M2", "M3", "M4"]:
            if k in san:
                macros[f"acc{ {'M1':'MOne','M2':'MTwo','M3':'MThree','M4':'MFour'}[k] }Pct"] = \
                    fmt(100 * san[k]["accuracy"], 1)

    e0 = _load("exp0_gate.json")
    if e0:
        b = e0["branch"]
        for mk, mn in [("M2", "MTwo"), ("M3", "MThree"), ("M4", "MFour")]:
            for pk, pn in [("sigma1", "Sigma"), ("henrici_norm", "Henrici"),
                           ("unit_mass", "UnitMass"), ("kappa", "Kappa"),
                           ("n_expanding", "NExp"), ("rho", "Rho"),
                           ("baseline_step", "BaseStep"), ("baseline_cnorm", "BaseNorm"),
                           ("baseline_dc", "BaseDc")]:
                if pk in b[mk]:
                    macros[f"auroc{pn}{mn}"] = fmt(b[mk][pk]["auroc"])
                    macros[f"auroc{pn}{mn}CI"] = fmt_ci(b[mk][pk]["auroc_ci"])
        if "branch_stratified" in e0:
            macros["stratSigma"] = fmt(e0["branch_stratified"]["sigma1"]["auroc"])
            macros["stratDc"] = fmt(e0["branch_stratified"]["baseline_dc"]["auroc"])
            macros["stratHenrici"] = fmt(e0["branch_stratified"]["henrici_norm"]["auroc"])
        a = e0["anchor"]
        for pk, pn in [("unit_mass", "UnitMass"), ("rho", "Rho"), ("sigma1", "Sigma"),
                       ("baseline_cnorm", "BaseNorm"), ("baseline_step", "BaseStep"),
                       ("baseline_dc", "BaseDc"), ("henrici_norm", "Henrici"),
                       ("n_expanding", "NExp")]:
            macros[f"anchorSp{pn}"] = fmt(a[pk]["spearman"]["spearman"])
            macros[f"anchorSp{pn}CI"] = fmt_ci(a[pk]["spearman"]["ci"])
            macros[f"anchorAu{pn}"] = fmt(a[pk]["auroc_top_tercile"]["auroc"])
            macros[f"anchorAu{pn}CI"] = fmt_ci(a[pk]["auroc_top_tercile"]["auroc_ci"])
        s = e0["separation_sigma1"]
        macros["sepMTwo"] = fmt(s["M2_mean"])
        macros["sepMThree"] = fmt(s["M3_mean"])
        macros["sepMFour"] = fmt(s["M4_mean"])
        macros["sepWilcoxonMThreeP"] = fmt_p(s["wilcoxon_M2_vs_M3"]["p"])
        macros["sepWilcoxonMFourP"] = fmt_p(s["wilcoxon_M2_vs_M4"]["p"])
        if "pruned_control" in e0:
            pc = e0["pruned_control"]
            macros["nPruned"] = pc["n_pruned"]
            macros["accPrunedPct"] = fmt(100 * pc["accuracy_pruned"], 1)
            macros["prunedDropBranch"] = fmt(pc["mean_drop_at_branch_steps"])
            macros["prunedDropNonbranch"] = fmt(pc["mean_drop_at_nonbranch_steps"])
            macros["prunedWilcoxonP"] = fmt_p(pc["wilcoxon_branch_drop_p"])
            macros["prunedMWUP"] = fmt_p(pc["mwu_branch_vs_nonbranch_p"])
        macros["KOneFires"] = "fires" if e0["K1"]["fires"] else "does not fire"
        macros["KOneSepRatioMThree"] = fmt(e0["K1"]["sep_ratio_M3"], 2)
        macros["KOneSepRatioMFour"] = fmt(e0["K1"]["sep_ratio_M4"], 2)

    e1 = _load("exp1_causal.json")
    if e1:
        macros["eOneNProblems"] = e1["n_problems"]
        for e, en in [(0.1, "Lo"), (0.3, "Hi")]:
            d = e1.get(f"eps_{e}")
            if d:
                macros[f"eOneRatioVone{en}"] = fmt(d["ratio_v1_over_rand"], 2)
                macros[f"eOneRatioQone{en}"] = fmt(d["ratio_q1_over_rand"], 2)
                macros[f"eOneVoneP{en}"] = fmt_p(d["wilcoxon_v1_vs_rand"]["p"])
                macros[f"eOneQoneP{en}"] = fmt_p(d["wilcoxon_q1_vs_rand"]["p"])
                macros[f"eOneFlipVone{en}Pct"] = fmt(100 * d["flip_v1"], 1)
                macros[f"eOneFlipRand{en}Pct"] = fmt(100 * d["flip_rand"], 1)
                macros[f"eOneVoneBranch{en}"] = fmt(d["v1_mean_at_branch"], 2)
                macros[f"eOneVoneNonbranch{en}"] = fmt(d["v1_mean_at_nonbranch"], 2)
        macros["eOneSubRatio"] = fmt(e1["subspace"]["ratio"], 2)
        macros["eOneSubP"] = fmt_p(e1["subspace"]["wilcoxon"]["p"])
        macros["eOneSubFlipSpectralPct"] = fmt(100 * e1["subspace"]["flip_spectral"], 1)
        macros["eOneSubFlipRandomPct"] = fmt(100 * e1["subspace"]["flip_random"], 1)
        macros["KTwoFires"] = "fires" if e1["K2"]["fires"] else "does not fire"

    e2 = _load("exp2_koopman.json")
    if e2:
        rm = e2["part_a_routedness"]["R_median"]
        macros["routedMedianOneTwo"] = fmt(rm.get("1_2"), 2)
        macros["routedMedianTwoTwo"] = fmt(rm.get("2_2"), 2)
        macros["routedMedianThreeTwo"] = fmt(rm.get("3_2"), 2)
        macros["routedMedianOneThree"] = fmt(rm.get("1_3"), 2)
        macros["routedMedianTwoThree"] = fmt(rm.get("2_3"), 2)
        macros["routedMedianThreeThree"] = fmt(rm.get("3_3"), 2)
        macros["routedMedianMax"] = fmt(max(v for v in rm.values()), 2)
        macros["routedMedianMin"] = fmt(min(v for v in rm.values()), 2)
        ed = e2["part_b_edmd"]["edmd_r128"]
        macros["edmdExplainedPct"] = fmt(100 * ed["explained_var"], 1)
        macros["edmdResid"] = fmt(ed["fit_residual"], 3)
        macros["edmdRho"] = fmt(ed["spectral_radius"], 3)
        macros["edmdUnitCount"] = ed["n_unit_band"]
        macros["koopResidBranchAu"] = fmt(ed["branch_auroc_koopman_residual"]["auroc"])
        macros["koopResidBranchAuCI"] = fmt_ci(ed["branch_auroc_koopman_residual"]["auroc_ci"])
        macros["koopPartBranchAu"] = fmt(ed["branch_auroc_unit_participation"]["auroc"])
        macros["koopPartAnchorSp"] = fmt(ed["anchor_spearman_unit_participation"]["spearman"])
        macros["koopPartAnchorSpCI"] = fmt_ci(ed["anchor_spearman_unit_participation"]["ci"])
        macros["koopResidAnchorSp"] = fmt(ed["anchor_spearman_koopman_residual"]["spearman"])
        macros["KThreeFires"] = "fires" if e2["K3"]["fires"] else "does not fire"

    e3 = _load("exp3_robustness.json")
    if e3:
        if "valid_split" in e3:
            macros["validAurocSigma"] = fmt(e3["valid_split"]["branch_auroc_sigma1"]["auroc"])
            macros["validAurocSigmaCI"] = fmt_ci(e3["valid_split"]["branch_auroc_sigma1"]["auroc_ci"])
            macros["validAccPct"] = fmt(100 * e3["valid_split"]["accuracy"], 1)
            macros["nValidProblems"] = e3["valid_split"]["n"]
        if "natural_linear_chains" in e3:
            nl = e3["natural_linear_chains"]
            macros["natlinN"] = nl["n"]
            macros["natlinSigma"] = fmt(nl["sigma1_mean"], 2)
            macros["natlinSigmaBranch"] = fmt(nl["M2_sigma1_mean_branch"], 2)
            macros["natlinSigmaNonbranch"] = fmt(nl["M2_sigma1_mean_nonbranch"], 2)
        if "epoch_sweep" in e3:
            for ep, en in [("10", "Ten"), ("20", "Twenty"), ("30", "Thirty"),
                           ("40", "Forty"), ("best", "Best")]:
                if ep in e3["epoch_sweep"]:
                    d = e3["epoch_sweep"][ep]
                    macros[f"epoch{en}Auroc"] = fmt(d["branch_auroc_sigma1"]["auroc"])
                    macros[f"epoch{en}AccPct"] = fmt(100 * d["accuracy"], 1)

    lines = ["% AUTO-GENERATED by experiments/make_figures.py -- do not edit\n"]
    for k, v in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    NUM_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"numbers.tex: {len(macros)} macros")


# ---------------------------------------------------------------- figures

def fig1_spectra():
    cap = _npz("exp0_capture_M2.npz")
    if cap is None:
        return
    n = int(cap["n_done"][0])
    s1 = _inv(cap, "sigma1")[:n]
    lab = cap["labels"][:n]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    b_mask = lab == 1
    nb_mask = lab == 0
    steps = np.arange(1, 7)
    mb = [np.nanmean(s1[:, t][b_mask[:, t]]) for t in range(6)]
    sb = [np.nanstd(s1[:, t][b_mask[:, t]]) / np.sqrt(max(b_mask[:, t].sum(), 1))
          for t in range(6)]
    mn = [np.nanmean(s1[:, t][nb_mask[:, t]]) if nb_mask[:, t].sum() else np.nan
          for t in range(6)]
    sn = [np.nanstd(s1[:, t][nb_mask[:, t]]) / np.sqrt(max(nb_mask[:, t].sum(), 1))
          for t in range(6)]
    ax.errorbar(steps, mb, yerr=sb, color=C_BRANCH, label="branch steps", marker="o", ms=3)
    ax.errorbar(steps, mn, yerr=sn, color=C_NON, label="non-branch steps", marker="s", ms=3)
    ax.set_xlabel("thought step $t$")
    ax.set_ylabel(r"$\sigma_1(J_t)$")
    ax.legend(frameon=False)
    ax.set_title("(a) transient expansion by step type (M2)")

    ax = axes[1]
    eig = cap["eig_abs"][:n]
    m = lab >= 0
    for cls, col, name in [(1, C_BRANCH, "branch"), (0, C_NON, "non-branch")]:
        sel = (lab == cls)
        prof = np.nanmean(eig[sel], axis=0)
        ax.plot(np.arange(1, 17), prof, color=col, marker=".", label=name)
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("eigenvalue rank (by $|\\lambda|$)")
    ax.set_ylabel(r"mean $|\lambda_i|$")
    ax.set_title("(b) eigenvalue modulus profile")
    ax.legend(frameon=False)
    fig.savefig(FIGS / "fig1_spectra.pdf")
    plt.close(fig)


def fig2_branch():
    e0 = _load("exp0_gate.json")
    caps = {"M2": _npz("exp0_capture_M2.npz"), "M3": _npz("exp0_capture_M3.npz"),
            "M4": _npz("exp0_capture_M4.npz")}
    if e0 is None or any(v is None for v in caps.values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    data, ticks = [], []
    for mk, col in [("M2", C_M2), ("M3", C_M3), ("M4", C_M4)]:
        cap = caps[mk]
        n = int(cap["n_done"][0])
        s1 = _inv(cap, "sigma1")[:n]
        lab = cap["labels"][:n]
        data.append(s1[lab == 1])
        data.append(s1[lab == 0])
        ticks += [f"{mk}\nbranch", f"{mk}\nnon-br."]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.55)
    cols = [C_BRANCH, C_NON] * 3
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticklabels(ticks)
    ax.set_ylabel(r"$\sigma_1(J_t)$")
    ax.set_title("(a) expansion at branch vs non-branch steps")

    ax = axes[1]
    preds = ["sigma1", "henrici_norm", "unit_mass", "baseline_step",
             "baseline_cnorm", "baseline_dc"]
    names = [r"$\sigma_1$", r"$d_F/\|J\|_F$", "unit mass", "step $t$",
             r"$\|c_t\|$", r"$\|\Delta c_t\|$"]
    x = np.arange(len(preds))
    w = 0.26
    for off, (mk, col) in zip([-w, 0, w], [("M2", C_M2), ("M3", C_M3), ("M4", C_M4)]):
        au = [e0["branch"][mk][p]["auroc"] for p in preds]
        lo = [e0["branch"][mk][p]["auroc_ci"][0] for p in preds]
        hi = [e0["branch"][mk][p]["auroc_ci"][1] for p in preds]
        ax.bar(x + off, au, w, color=col, alpha=0.8, label=mk)
        ax.errorbar(x + off, au, yerr=[np.array(au) - lo, np.array(hi) - au],
                    fmt="none", ecolor="black", lw=0.7, capsize=1.5)
    ax.axhline(0.5, color="gray", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("branch AUROC")
    ax.set_ylim(0.3, 1.0)
    ax.legend(frameon=False, ncol=3)
    ax.set_title("(b) branch prediction, model vs pause nulls")
    fig.savefig(FIGS / "fig2_branch.pdf")
    plt.close(fig)


def fig3_pruned():
    capO = _npz("exp0_capture_M2.npz")
    capP = _npz("exp0_capture_M2_pruned.npz")
    if capO is None or capP is None:
        return
    orig_by_idx = {int(v): i for i, v in enumerate(capO["prob_idx"])}
    sO, sP = _inv(capO, "sigma1"), _inv(capP, "sigma1")
    lab = capO["labels"]
    nP = int(capP["n_done"][0])
    d_b, d_n = [], []
    pair_b = []
    for j in range(nP):
        i = orig_by_idx.get(int(capP["prob_idx"][j]))
        if i is None:
            continue
        for t in range(6):
            if lab[i, t] == 1:
                d_b.append(sO[i, t] - sP[j, t])
                pair_b.append((sO[i, t], sP[j, t]))
            elif lab[i, t] == 0:
                d_n.append(sO[i, t] - sP[j, t])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    pair_b = np.array(pair_b)
    lim = [min(pair_b.min(), 0), pair_b.max() * 1.05]
    ax.plot(lim, lim, color="gray", lw=0.7, ls=":")
    ax.scatter(pair_b[:, 0], pair_b[:, 1], s=6, alpha=0.4, color=C_BRANCH)
    ax.set_xlabel(r"$\sigma_1$ original (branch step)")
    ax.set_ylabel(r"$\sigma_1$ pruned twin (same step)")
    ax.set_title("(a) paired pruned-real control")
    ax = axes[1]
    ax.hist(d_b, bins=40, alpha=0.6, color=C_BRANCH, label="formerly-branch steps",
            density=True)
    ax.hist(d_n, bins=40, alpha=0.6, color=C_NON, label="non-branch steps",
            density=True)
    ax.axvline(0, color="gray", lw=0.7, ls=":")
    ax.set_xlabel(r"$\sigma_1$(original) $-$ $\sigma_1$(pruned)")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    ax.set_title("(b) expansion drop after de-branching")
    fig.savefig(FIGS / "fig3_pruned.pdf")
    plt.close(fig)


def fig4_anchor():
    cap = _npz("exp0_capture_M2.npz")
    abl = _npz("exp0_ablate_M2.npz")
    e0 = _load("exp0_gate.json")
    if cap is None or abl is None or e0 is None:
        return
    n = min(int(cap["n_done"][0]), int(abl["n_done"][0]))
    um = _inv(cap, "unit_mass")[:n].ravel()
    I = abl["I_mean"][:n].ravel()
    m = np.isfinite(um) & np.isfinite(I)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    ax.scatter(um[m], np.maximum(I[m], 1e-4), s=4, alpha=0.25, color=C_M2)
    ax.set_yscale("log")
    ax.set_xlabel("near-unit spectral mass of $J_t$")
    ax.set_ylabel(r"ablation influence $I_t$ ($|\Delta$margin$|$)")
    ax.set_title("(a) slow modes vs causal influence")
    ax = axes[1]
    preds = ["unit_mass", "rho", "sigma1", "henrici_norm", "baseline_step",
             "baseline_cnorm", "baseline_dc"]
    names = ["unit mass", r"$\rho$", r"$\sigma_1$", r"$d_F/\|J\|_F$", "step $t$",
             r"$\|c_t\|$", r"$\|\Delta c_t\|$"]
    au = [e0["anchor"][p]["auroc_top_tercile"]["auroc"] for p in preds]
    lo = [e0["anchor"][p]["auroc_top_tercile"]["auroc_ci"][0] for p in preds]
    hi = [e0["anchor"][p]["auroc_top_tercile"]["auroc_ci"][1] for p in preds]
    x = np.arange(len(preds))
    ax.bar(x, au, 0.6, color=C_M2, alpha=0.8)
    ax.errorbar(x, au, yerr=[np.array(au) - lo, np.array(hi) - au], fmt="none",
                ecolor="black", lw=0.7, capsize=1.5)
    ax.axhline(0.5, color="gray", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("anchor AUROC (top tercile)")
    ax.set_title("(b) anchor prediction (M2)")
    fig.savefig(FIGS / "fig4_anchor.pdf")
    plt.close(fig)


def fig5_causal():
    e1 = _load("exp1_causal.json")
    npz = _npz("exp1_causal.npz")
    if e1 is None or npz is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    eps = e1["eps"]
    arms = [("v1", r"$v_1(J_t)$", C_BRANCH), ("q1", "top eig dir", C_M2)]
    x = np.arange(len(eps))
    w = 0.25
    n_rand = e1["n_rand"]
    for off, (a, nm, col) in zip([-w, 0], arms):
        mus = [np.mean(npz[f"eff_{a}_{e}"]) for e in eps]
        ses = [np.std(npz[f"eff_{a}_{e}"]) / np.sqrt(len(npz[f"eff_{a}_{e}"]))
               for e in eps]
        ax.bar(x + off, mus, w, color=col, alpha=0.85, label=nm)
        ax.errorbar(x + off, mus, yerr=ses, fmt="none", ecolor="black", lw=0.7,
                    capsize=1.5)
    mus = [np.mean([npz[f"eff_rand{r}_{e}"] for r in range(n_rand)]) for e in eps]
    ax.bar(x + w, mus, w, color="gray", alpha=0.8, label="random dirs")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\epsilon={e}$" for e in eps])
    ax.set_ylabel(r"downstream $|\Delta$margin$|$")
    ax.legend(frameon=False)
    ax.set_title("(a) directional interventions (M2)")

    ax = axes[1]
    se, sr = npz["sub_spectral"], npz["sub_random"]
    bp = ax.boxplot([se, sr], showfliers=False, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], [C_M2, "gray"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticklabels(["top-4 eig subspace\nprojected out", "random 4-dim\nsubspace out"])
    ax.set_ylabel(r"$|\Delta$margin$|$")
    ax.set_title("(b) slow-mode subspace ablation")
    fig.savefig(FIGS / "fig5_causal.pdf")
    plt.close(fig)


def fig6_routed():
    npz = _npz("exp2_routed.npz")
    e2 = _load("exp2_koopman.json")
    if npz is None or e2 is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    keys = ["1_2", "2_2", "3_2", "1_3", "2_3", "3_3"]
    data = [npz[f"R_{k}"] for k in keys if f"R_{k}" in npz]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.55)
    for patch in bp["boxes"]:
        patch.set_facecolor(C_M3)
        patch.set_alpha(0.6)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticklabels([f"$t{{=}}{k[0]},k{{=}}{k[2]}$" for k in keys])
    ax.set_ylabel(r"routedness $R_{t,k}$")
    ax.axhline(0.5, color="gray", lw=0.6, ls=":")
    ax.set_title("(a) full influence vs chained local Jacobians")
    ax = axes[1]
    ed = e2["part_b_edmd"]["edmd_r128"]
    tops = ed["top_eigs_abs"]
    ax.plot(np.arange(1, len(tops) + 1), tops, marker="o", ms=3, color=C_M2,
            label="pooled EDMD ($r{=}128$)")
    ed2 = e2["part_b_edmd"].get("edmd_r128_delay2")
    if ed2:
        ax.plot(np.arange(1, len(ed2["top_eigs_abs"]) + 1), ed2["top_eigs_abs"],
                marker="s", ms=3, color=C_M4, label="delay-2 EDMD")
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("Koopman mode rank")
    ax.set_ylabel(r"$|\lambda|$")
    ax.legend(frameon=False)
    ax.set_title("(b) orbit-level Koopman spectrum")
    fig.savefig(FIGS / "fig6_routed.pdf")
    plt.close(fig)


def fig7_epochs():
    e3 = _load("exp3_robustness.json")
    if e3 is None or "epoch_sweep" not in e3:
        return
    sw = e3["epoch_sweep"]
    eps_order = [k for k in ["10", "20", "30", "40", "best"] if k in sw]
    xs = np.arange(len(eps_order))
    au = [sw[k]["branch_auroc_sigma1"]["auroc"] for k in eps_order]
    lo = [sw[k]["branch_auroc_sigma1"]["auroc_ci"][0] for k in eps_order]
    hi = [sw[k]["branch_auroc_sigma1"]["auroc_ci"][1] for k in eps_order]
    acc = [sw[k]["accuracy"] for k in eps_order]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.errorbar(xs, au, yerr=[np.array(au) - lo, np.array(hi) - au], marker="o",
                ms=4, color=C_M2, label="branch AUROC ($\\sigma_1$)", capsize=2)
    ax.axhline(0.5, color="gray", lw=0.6, ls=":")
    ax2 = ax.twinx()
    ax2.plot(xs, acc, marker="s", ms=4, color=C_M3, label="ProsQA accuracy")
    ax2.set_ylabel("accuracy", color=C_M3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"ep.{k}" if k != "best" else "best" for k in eps_order])
    ax.set_ylabel("branch AUROC", color=C_M2)
    ax.set_title("curriculum sweep")
    fig.savefig(FIGS / "fig7_epochs.pdf")
    plt.close(fig)


def main():
    numbers()
    for f in [fig1_spectra, fig2_branch, fig3_pruned, fig4_anchor, fig5_causal,
              fig6_routed, fig7_epochs]:
        try:
            f()
            print(f"{f.__name__}: ok")
        except Exception as e:  # noqa: BLE001
            print(f"{f.__name__}: SKIP/FAIL ({e})")


if __name__ == "__main__":
    main()
