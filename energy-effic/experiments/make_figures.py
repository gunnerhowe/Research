"""Generate ALL paper figures (paper/figs/) and tables (paper/tables/) from
results/*.json. Every number in the paper comes from here or from
gen_paper_numbers.py. Skips missing result files with a warning so it can run
incrementally; the final pre-submission run must emit everything.
"""

import numpy as np
import matplotlib.pyplot as plt

from common import ROOT, RESULTS, load_json, log

FIGS = ROOT / "paper" / "figs"
TABLES = ROOT / "paper" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150})

TASKS = ["sc2", "psmnist"]
TASK_LABEL = {"sc2": "SC2 keyword spotting", "psmnist": "row-sequential MNIST",
              "enwik8": "enwik8 char-LM"}
PRED_LABEL = {"empirical": "(b) empirical profile", "rice": "(a) Rice",
              "discrete": "(a$'$) discrete Gaussian", "iid": "(c) Gaussian iid"}
PRED_COLOR = {"empirical": "C0", "rice": "C1", "discrete": "C3", "iid": "C2"}


def _maybe(path):
    p = RESULTS / path
    if not p.exists():
        log(f"SKIP (missing): {path}")
        return None
    return load_json(p)


# ------------------------------------------------------------------- E0


def fig_e0_validation():
    ou = _maybe("exp0_ou.json")
    eps = _maybe("exp0_eps_sweep.json")
    if ou is None or eps is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

    # (a) OU: exact-discrete + rice-FD relative error vs rho at u=0 and u=1
    ax = axes[0]
    rhos = sorted({r["rho"] for r in ou})
    for li, ls in ((0, "-"), (2, "--")):
        exact = [np.mean([r["exact_discrete"][li] / r["measured"][li]
                          for r in ou if r["rho"] == rho]) for rho in rhos]
        rice = [np.mean([r["rice_fd"][li] / r["measured"][li]
                         for r in ou if r["rho"] == rho]) for rho in rhos]
        lab = f"$u={ou[0]['levels'][li]:g}\\sigma$"
        ax.plot(rhos, exact, "o" + ls, color="C3", label=f"discrete, {lab}")
        ax.plot(rhos, rice, "s" + ls, color="C1", label=f"Rice (FD), {lab}")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel(r"lag-1 autocorrelation $\rho_1$")
    ax.set_ylabel("predicted / measured")
    ax.set_title("(a) OU: sampled-sequence crossing rates")
    ax.legend(fontsize=6.5)

    # (b) eps sweep bias
    ax = axes[1]
    ep = [r["eps_rel"] for r in eps["eps_sweep"]]
    for est, c in (("seg", "C0"), ("mid", "C1")):
        bias0 = [abs(r[f"{est}_bias"][0]) for r in eps["eps_sweep"]]
        ax.plot(ep, bias0, "o-", color=c,
                label={"seg": "segment", "mid": "midpoint"}[est])
    ax.axhline(eps["split_half_floor"][0], color="k", ls=":",
               label="split-half floor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\epsilon / \sigma$")
    ax.set_ylabel(r"|relative bias| at $u=0$")
    ax.set_title("(b) smoothing bandwidth: bias")
    ax.legend(fontsize=7)

    # (c) eps sweep std
    ax = axes[2]
    for est, c in (("seg", "C0"), ("mid", "C1")):
        std0 = [r[f"{est}_std"][0] for r in eps["eps_sweep"]]
        ax.plot(ep, std0, "o-", color=c,
                label={"seg": "segment", "mid": "midpoint"}[est])
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon / \sigma$")
    ax.set_ylabel("relative std (8-trace batches)")
    ax.set_title("(c) smoothing bandwidth: variance")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "e0_validation.pdf")
    plt.close(fig)
    log("fig e0_validation")


def fig_e0_gamma():
    d = _maybe("exp0_sod_gamma.json")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = d["x_grid"]
    for name, g in d["per_process_gamma"].items():
        ax.plot(x, g, "-", lw=1.0, alpha=0.7, label=name)
    ax.plot(x, d["pooled_gamma"], "k-", lw=2.2, label="pooled")
    ax.set_xscale("log")
    ax.set_xlabel(r"$x = \theta / \sigma_\delta$")
    ax.set_ylabel(r"$\gamma$ = events / TV bound")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "e0_gamma.pdf")
    plt.close(fig)
    log("fig e0_gamma")


# ------------------------------------------------------------------- E1


def fig_e1_profiles():
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0))
    tasks = ["sc2", "psmnist", "enwik8"]
    for col, task in enumerate(tasks):
        d = _maybe(f"exp1_{task}.json")
        if d is None:
            continue
        prof = d["s0"]["profiles"]
        stream = "h1" if task != "enwik8" else "l0.fc2_in"
        exs = prof[stream]["examples"]
        # top: exemplar with median kurtosis; bottom: max-kurtosis exemplar
        for row, ex in ((0, exs[1]), (1, exs[2])):
            ax = axes[row, col]
            ks = ex["k_sigma"]
            ax.plot(ks, ex["measured"], "ko-", ms=3.5, label="measured")
            for p in ("empirical", "rice", "iid"):
                ax.plot(ks, ex[p], "--", color=PRED_COLOR[p], lw=1.3,
                        label=PRED_LABEL[p])
            ax.set_title(f"{TASK_LABEL[task]}, {stream} ch{ex['channel']} "
                         f"(kurt {ex['kurt']:.1f})", fontsize=8)
            ax.set_xlabel(r"level $(u-m)/\sigma$")
            if col == 0:
                ax.set_ylabel("crossings / step")
            if row == 0 and col == 0:
                ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGS / "e1_profiles.pdf")
    plt.close(fig)
    log("fig e1_profiles")


def fig_e1_kurtosis():
    """Per-channel Rice error vs excess kurtosis (where Gaussianity breaks)."""
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for task, marker in (("sc2", "o"), ("psmnist", "s"), ("enwik8", "^")):
        d = _maybe(f"exp1_{task}.json")
        if d is None:
            continue
        ku, er = [], []
        for s in d.values():
            for prof in s["profiles"].values():
                k = prof["kurt"]
                e = prof["rice"]["per_channel_median"]
                for a, b in zip(k, e):
                    if b is not None:
                        ku.append(a)
                        er.append(b)
        ax.scatter(ku, er, s=4, alpha=0.25, marker=marker,
                   label=f"{TASK_LABEL[task]}")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.set_xlabel("excess kurtosis (channel)")
    ax.set_ylabel("Rice median rel. error")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "e1_kurtosis.pdf")
    plt.close(fig)
    log("fig e1_kurtosis")


def fig_e1_events():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    tasks = ["sc2", "psmnist", "enwik8"]
    for col, task in enumerate(tasks):
        d = _maybe(f"exp1_{task}.json")
        if d is None:
            continue
        ax = axes[col]
        stream = "h1" if task != "enwik8" else "l0.qkv_in"
        xs = [r["x"] for r in d["s0"]["events"]]

        def get(field):
            rows = []
            for s in d.values():
                if task == "enwik8":
                    rows.append([r["streams"][stream][field]
                                 for r in s["events"]])
                else:
                    rows.append([r[stream][field] for r in s["events"]])
            return np.array(rows)

        meas = get("measured")
        ax.errorbar(xs, meas.mean(0), yerr=meas.std(0), fmt="ko-", ms=3.5,
                    label="measured (closed loop)")
        for f, lab, c in (("pred_analytic_emp_tv", r"TV/$\theta\cdot\gamma$ (cal)", "C1"),
                          ("pred_openloop_cal", "open-loop sim (cal)", "C0"),
                          ("openloop_eval", "open-loop (eval)", "C2")):
            v = get(f)
            ax.plot(xs, v.mean(0), "--", color=c, lw=1.3, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$x = \theta/\sigma_\delta$")
        if col == 0:
            ax.set_ylabel("event rate / component / step")
        ax.set_title(f"{TASK_LABEL[task]}, {stream}", fontsize=8.5)
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGS / "e1_events.pdf")
    plt.close(fig)
    log("fig e1_events")


def table_e1():
    rows = []
    for task in ("sc2", "psmnist", "enwik8"):
        d = _maybe(f"exp1_{task}.json")
        if d is None:
            continue
        streams = list(d["s0"]["profiles"].keys())
        for st in streams:
            vals = {}
            for p in ("empirical", "rice", "iid"):
                errs = [d[s]["profiles"][st][p]["median_rel_err"] for s in d]
                w10 = [d[s]["profiles"][st][p]["frac_within_10pct"] for s in d]
                vals[p] = (np.mean(errs), np.std(errs), np.mean(w10))
            floor = np.mean([d[s]["profiles"][st]["split_half_floor_median"]
                             for s in d])
            rows.append((task, st, vals, floor))
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"task & stream & \multicolumn{2}{c}{(b) empirical} & "
             r"\multicolumn{2}{c}{(a) Rice} & \multicolumn{2}{c}{(c) iid} & floor\\",
             r" & & err & $\le$10\% & err & $\le$10\% & err & $\le$10\% & \\",
             r"\midrule"]
    for task, st, vals, floor in rows:
        e, r_, i = vals["empirical"], vals["rice"], vals["iid"]
        lines.append(
            f"{TASK_LABEL[task].split()[0]} & {st.replace('_', r'\_')} & "
            f"{e[0]*100:.1f}\\% & {e[2]*100:.0f}\\% & "
            f"{r_[0]*100:.1f}\\% & {r_[2]*100:.0f}\\% & "
            f"{i[0]*100:.1f}\\% & {i[2]*100:.0f}\\% & {floor*100:.1f}\\%\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "e1_profiles.tex").write_text("\n".join(lines))
    log("table e1_profiles")


# ------------------------------------------------------------------- E2


def fig_e2_pareto():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for col, task in enumerate(TASKS):
        d = _maybe(f"exp2_{task}.json")
        if d is None:
            continue
        ax = axes[col]
        for m, lab, c in (("uniform_x", "analytic uniform-$x$", "C0"),
                          ("prop_share", "analytic prop-share", "C1"),
                          ("single_theta", "single abs. $\\theta$", "C2")):
            fr = np.array([[r["frac_of_dense"] for r in d[s]["analytic"][m]]
                           for s in d])
            ac = np.array([[r["acc"] for r in d[s]["analytic"][m]] for s in d])
            ax.errorbar(fr.mean(0), ac.mean(0), yerr=ac.std(0),
                        xerr=fr.std(0), fmt="o-", ms=3, color=c, lw=1.2,
                        label=lab)
        for k, c, ls in (("k6", "0.55", ":"), ("k48", "0.2", "--")):
            pts = []
            for s in d:
                pts += [(r["frac_of_dense"], r["acc"])
                        for r in d[s]["random_search"][k]]
            pts.sort()
            fr, ac = zip(*pts)
            ax.plot(fr, ac, ls, color=c, lw=1.4,
                    label=f"random search ({k[1:]} cfgs/seed)")
        ax.set_xscale("log")
        ax.set_xlabel("MAC-weighted events (fraction of dense)")
        ax.set_ylabel("test accuracy")
        ax.set_title(TASK_LABEL[task], fontsize=9)
        ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "e2_pareto.pdf")
    plt.close(fig)
    log("fig e2_pareto")


# ---------------------------------------------------------------- E3/E4


def _seed_envelope(d, prefix, fracs):
    """Per-seed upper envelope of accuracy at given event fractions, by
    interpolating each seed's pooled points of arms matching prefix."""
    out = []
    for s in d:
        pts = []
        for name, arm in d[s].items():
            if name == prefix or (name.startswith(prefix) and name != prefix):
                for r in arm["front"]:
                    pts.append((r["frac_of_dense"], r["acc"]))
        if not pts:
            continue
        pts.sort()
        f = np.array([p[0] for p in pts])
        a = np.array([p[1] for p in pts])
        # envelope: running max from the left is wrong; interpolate the
        # Pareto-filtered curve
        env_f, env_a, best = [], [], -1
        for fi, ai in sorted(zip(f, a), key=lambda p: p[0]):
            if ai > best:
                env_f.append(fi)
                env_a.append(ai)
                best = ai
        out.append(np.interp(fracs, env_f, env_a,
                             left=np.nan, right=env_a[-1]))
    return np.array(out)


FAMILIES = [("posthoc", "post-hoc thresholding (base)", "k"),
            ("budget_", "crossing budget (ours)", "C0"),
            ("l1delta_", "L1 on deltas", "C1"),
            ("rate_", "rate regularizer", "C2")]


def fig_e3_pareto():
    fracs = np.geomspace(0.02, 0.7, 25)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for col, task in enumerate(TASKS):
        d3 = _maybe(f"exp3_{task}.json")
        d4 = _maybe(f"exp4_{task}.json")
        if d3 is None:
            continue
        ax = axes[col]
        for prefix, lab, c in FAMILIES:
            src = d3 if prefix in ("posthoc", "budget_") else d4
            if src is None:
                continue
            env = _seed_envelope(src, prefix, fracs)
            if env.size == 0:
                continue
            mean, sd = np.nanmean(env, 0), np.nanstd(env, 0)
            ax.plot(fracs, mean, "-", color=c, lw=1.6, label=lab)
            ax.fill_between(fracs, mean - sd, mean + sd, color=c, alpha=0.15)
        ax.set_xscale("log")
        ax.set_xlabel("MAC-weighted events (fraction of dense)")
        ax.set_ylabel("test accuracy")
        ax.set_title(TASK_LABEL[task], fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "e3_pareto.pdf")
    plt.close(fig)
    log("fig e3_pareto")


def fig_e4_histograms():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for col, task in enumerate(TASKS):
        d = _maybe(f"exp4_{task}_histograms.json")
        if d is None:
            continue
        ax = axes[col]
        for name, lab, c in (("base", "base", "k"),
                             ("budget", "crossing budget", "C0"),
                             ("l1delta", "L1 on deltas", "C1")):
            cnts = np.mean([d[s][name]["counts"] for s in d], axis=0)
            bins = np.array(d["s0"][name]["bins"])
            centers = np.sqrt(bins[:-1] * bins[1:])
            dens = cnts / cnts.sum() / np.diff(np.log(bins))
            ax.plot(centers, dens, "-", color=c, lw=1.5, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-4, 2)
        ax.set_ylim(bottom=1e-4)
        ax.set_xlabel(r"$|h_t - h_{t-1}|$")
        ax.set_ylabel("density (per log bin)")
        ax.set_title(TASK_LABEL[task], fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "e4_histograms.pdf")
    plt.close(fig)
    log("fig e4_histograms")


def main():
    fig_e0_validation()
    fig_e0_gamma()
    fig_e1_profiles()
    fig_e1_kurtosis()
    fig_e1_events()
    table_e1()
    fig_e2_pareto()
    fig_e3_pareto()
    fig_e4_histograms()


if __name__ == "__main__":
    main()
