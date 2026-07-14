"""Paper 4 analysis: natural-data (MNIST/Omnigrok) test of representational priors.

Inputs:  runs/grid4 (free-norm grid, 7 conditions x 5 seeds)
         runs/grid4b (pre-registered amendment: E2 matched-norm sweep, E3 supcon_nn, E4 dose)
Outputs: analysis/out4/results.csv, stats.json, fig4_*.pdf/png

Conventions follow the house rules: censored runs scored at budget (conservative lower
bounds), open-triangle markers at budget in figures, per-seed pairing, no pooled claims
without labels.
"""
import csv
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out4")
BUDGET = 100000
SEEDS_FREE = [0, 1, 2, 3, 4]
SEEDS_E2 = [0, 1, 3, 4]
NORMS = [23, 35, 50, 65, 80, 92]  # c23 cells live in grid4 (base_clamp23 / aug_clamp23)
GAP_ABS = 0.09   # absolute cos_gap threshold for the probe-timing check (init ~0.06)
ACC_ABS = 0.5    # absolute test-acc threshold


def load(grid, name):
    d = os.path.join(ROOT, "runs", grid, name)
    sp = os.path.join(d, "summary.json")
    if not os.path.exists(sp):
        return None
    s = json.load(open(sp))
    recs = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl")) if l.strip()]
    return dict(summary=s, recs=recs)


def t_gen_scored(s):
    """t_gen with censoring scored at budget; returns (value, censored)."""
    tg = s["t_gen"]
    return (BUDGET, True) if tg is None else (tg, False)


def collect():
    rows = []
    for grid in ("grid4", "grid4b"):
        gdir = os.path.join(ROOT, "runs", grid)
        for name in sorted(os.listdir(gdir)):
            r = load(grid, name)
            if r is None:
                continue
            s, recs = r["summary"], r["recs"]
            m = re.match(r"(.+)_s(\d+)$", name)
            tg, cen = t_gen_scored(s)
            rows.append(dict(
                grid=grid, run=name, arm=m.group(1), seed=int(m.group(2)),
                condition=s["condition"], norm_clamp=s.get("norm_clamp", 0.0),
                lambda_con=s.get("lambda_con"), t_fit=s["t_fit"],
                t_gen=s["t_gen"], t_gen_scored=tg, censored=cen,
                max_te=round(max(x["test_acc"] for x in recs), 4),
                final_norm=round(recs[-1]["wnorm"], 2),
                nn_purity=s.get("nn_purity"),
            ))
    return rows


def by(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r[k] == v]
    return out


def one(rows, **kw):
    r = by(rows, **kw)
    assert len(r) == 1, f"expected 1 row for {kw}, got {len(r)}"
    return r[0]


def slope_fit(points):
    """OLS slope of ln(t_gen) on norm. points = [(norm, t, censored), ...]."""
    x = np.array([p[0] for p in points], float)
    y = np.log(np.array([p[1] for p in points], float))
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b), any(p[2] for p in points)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = collect()
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    stats = {}

    # ---------- free-norm grid ----------
    free = {}
    for arm in ("baseline", "augce", "supcon_aug", "supcon_label", "supcon_shufpair"):
        rs = sorted(by(rows, grid="grid4", arm=arm), key=lambda r: r["seed"])
        free[arm] = dict(
            t_gen=[r["t_gen"] for r in rs], censored=sum(r["censored"] for r in rs),
            max_te=[r["max_te"] for r in rs], final_norm=[r["final_norm"] for r in rs])
    # paired ratios vs baseline (scored)
    for arm in ("augce", "supcon_aug", "supcon_label"):
        rat = []
        for s in SEEDS_FREE:
            b = one(rows, grid="grid4", arm="baseline", seed=s)["t_gen_scored"]
            a = one(rows, grid="grid4", arm=arm, seed=s)["t_gen_scored"]
            rat.append(round(b / a, 3))
        free[arm]["paired_speedup_vs_baseline"] = rat
        free[arm]["median_speedup"] = round(float(np.median(rat)), 3)
    stats["free_norm"] = free

    # ---------- E2 matched-norm ----------
    e2 = {"per_norm": {}}
    for c in NORMS:
        cell = {}
        for arm_tag, (g, pref) in {
                "base": ("grid4b", "base_c"), "aug": ("grid4b", "aug_c")}.items():
            names = ([f"{pref}{c}_s{s}" for s in SEEDS_E2] if c != 23 else
                     [f"{'base_clamp23' if arm_tag=='base' else 'aug_clamp23'}_s{s}"
                      for s in SEEDS_E2])
            grid = "grid4b" if c != 23 else "grid4"
            rs = [one(rows, grid=grid, run=n) for n in names]
            cell[arm_tag] = dict(t_gen=[r["t_gen"] for r in rs],
                                 scored=[r["t_gen_scored"] for r in rs],
                                 censored=sum(r["censored"] for r in rs))
        cell["ratio_base_over_aug"] = [round(b / a, 3) for b, a in
                                       zip(cell["base"]["scored"], cell["aug"]["scored"])]
        cell["ratio_is_lower_bound"] = cell["base"]["censored"] > 0
        e2["per_norm"][c] = cell
    # slope fits on the ascending branch (50..92), per-seed then summarized
    fits = {}
    for arm in ("base", "aug"):
        pts = []
        for c in (50, 65, 80, 92):
            for i, s in enumerate(SEEDS_E2):
                cell = e2["per_norm"][c][arm]
                pts.append((c, cell["scored"][i], cell["t_gen"][i] is None))
        a, b, cen = slope_fit(pts)
        fits[arm] = dict(slope_per_unit=round(a, 5), mult_per_10=round(float(np.exp(10 * a)), 3),
                         includes_censored=cen)
    fits["slope_ratio_base_over_aug"] = round(fits["base"]["slope_per_unit"] /
                                              fits["aug"]["slope_per_unit"], 3)
    fits["note"] = ("baseline slope is a LOWER BOUND (censored cells scored at budget); "
                    "true ratio >= reported")
    e2["slope_fits"] = fits
    stats["e2_matched_norm"] = e2

    # ---------- E3 nn ----------
    e3 = {"per_seed": []}
    wins = 0
    for s in SEEDS_E2:
        b = one(rows, grid="grid4", arm="baseline", seed=s)
        n = one(rows, grid="grid4b", arm="nn", seed=s)
        win = n["t_gen_scored"] < b["t_gen_scored"]
        wins += win
        e3["per_seed"].append(dict(seed=s, baseline=b["t_gen"], nn=n["t_gen"],
                                   purity=n["nn_purity"], win=bool(win),
                                   ratio=round(b["t_gen_scored"] / n["t_gen_scored"], 3),
                                   nn_final_norm=n["final_norm"]))
    e3["tally"] = f"{wins}-{len(SEEDS_E2) - wins}"
    e3["verdict"] = "no evidence of free-norm transfer (pre-stated tally rule)"
    stats["e3_nn"] = e3

    # ---------- E4 dose ----------
    e4 = []
    for lam, grid, arm in [(0.0, "grid4", "baseline"), (0.03, "grid4b", "aug_lam0.03"),
                           (0.1, "grid4b", "aug_lam0.1"), (0.3, "grid4", "supcon_aug")]:
        for s in (0, 3):
            r = one(rows, grid=grid, arm=arm, seed=s)
            e4.append(dict(lam=lam, seed=s, t_gen=r["t_gen"], final_norm=r["final_norm"]))
    stats["e4_dose"] = e4

    # ---------- P5 probe timing (absolute thresholds) ----------
    p5 = []
    for grid, name in [("grid4", "baseline_s0"), ("grid4", "baseline_s3"),
                       ("grid4", "supcon_aug_s0"), ("grid4", "supcon_aug_s3"),
                       ("grid4b", "nn_s0"), ("grid4b", "nn_s3")]:
        recs = load(grid, name)["recs"]
        tg = next((r["step"] for r in recs if r["cos_gap"] >= GAP_ABS), None)
        ta = next((r["step"] for r in recs if r["test_acc"] >= ACC_ABS), None)
        p5.append(dict(run=name, step_gap=tg, step_acc=ta,
                       lead=None if tg is None or ta is None else ta - tg))
    stats["p5_probe_timing"] = dict(gap_threshold=GAP_ABS, acc_threshold=ACC_ABS, runs=p5)

    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=2)

    # ================= figures =================
    C = {"baseline": "#444444", "augce": "#1f77b4", "supcon_aug": "#d62728",
         "supcon_label": "#2ca02c", "supcon_shufpair": "#9467bd", "nn": "#ff7f0e"}

    # fig 1: free-norm test-acc curves (median across seeds)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for arm in ("baseline", "augce", "supcon_aug", "supcon_label", "supcon_shufpair"):
        curves = []
        for s in SEEDS_FREE:
            recs = load("grid4", f"{arm}_s{s}")["recs"]
            steps = [r["step"] for r in recs]
            accs = [r["test_acc"] for r in recs]
            curves.append((steps, accs))
        grid_steps = np.arange(0, BUDGET + 1, 200)
        interp = [np.interp(grid_steps, st, ac, right=ac[-1]) for st, ac in curves]
        med = np.median(np.vstack(interp), axis=0)
        ax.plot(grid_steps, med, color=C[arm], label=arm.replace("supcon_", "supcon-"), lw=1.8)
    ax.axhline(0.85, color="k", ls=":", lw=0.8)
    ax.set(xlabel="step", ylabel="test accuracy (median of 5 seeds)", xlim=(0, 60000),
           title="Free-norm race (MNIST, Omnigrok regime)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig4_1_freenorm.{ext}"), dpi=180)
    plt.close(fig)

    # fig 2: weight-norm trajectories (the confound picture), seed 0
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for arm in ("baseline", "augce", "supcon_aug", "supcon_label", "supcon_shufpair"):
        recs = load("grid4", f"{arm}_s0")["recs"]
        ax.plot([r["step"] for r in recs], [r["wnorm"] for r in recs],
                color=C[arm], label=arm.replace("supcon_", "supcon-"), lw=1.8)
    recs = load("grid4b", "nn_s0")["recs"]
    ax.plot([r["step"] for r in recs], [r["wnorm"] for r in recs],
            color=C["nn"], label="supcon-nn", lw=1.8)
    ax.set(xlabel="step", ylabel="total weight norm", xlim=(0, 60000),
           title="The confound: aux terms freeze or inflate the norm clock (seed 0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig4_2_normtraj.{ext}"), dpi=180)
    plt.close(fig)

    # fig 3: MONEY — delay vs pinned norm, censored as open triangles at budget
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for arm, col, lab in (("base", "#444444", "baseline (pinned)"),
                          ("aug", "#d62728", "supcon-aug (pinned)")):
        med = []
        for c in NORMS:
            cell = stats["e2_matched_norm"]["per_norm"][c][arm]
            n_cen = 0
            for tg, sc in zip(cell["t_gen"], cell["scored"]):
                if tg is None:
                    # jitter censored markers horizontally so overlapping seeds stay visible
                    ax.scatter([c - 1.2 + 1.2 * n_cen], [BUDGET], marker="^",
                               facecolors="none", edgecolors=col, s=46, zorder=3)
                    n_cen += 1
                else:
                    ax.scatter([c], [tg], marker="o", color=col, s=22, alpha=0.75, zorder=3)
            med.append(float(np.median(cell["scored"])))
        ax.plot(NORMS, med, color=col, lw=1.8, label=lab)
    ax.set_yscale("log")
    ax.axhline(BUDGET, color="k", ls=":", lw=0.8)
    ax.text(24, BUDGET * 0.72, "budget (open triangles = censored)", fontsize=7)
    ax.set(xlabel="pinned weight norm", ylabel="t_gen (steps, log scale)",
           title="Matched-norm delay law (n=4 seeds)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig4_3_delaylaw.{ext}"), dpi=180)
    plt.close(fig)

    # fig 4: probes lead behavior (supcon_aug seed 0)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    recs = load("grid4", "supcon_aug_s0")["recs"]
    st = [r["step"] for r in recs]
    ax.plot(st, [r["test_acc"] for r in recs], color="#d62728", lw=1.8, label="test accuracy")
    ax2 = ax.twinx()
    ax2.plot(st, [r["cos_gap"] for r in recs], color="#1f77b4", lw=1.8, ls="--",
             label="class cosine gap (probe)")
    ax.set(xlabel="step", ylabel="test accuracy", xlim=(0, 50000))
    ax2.set_ylabel("class cosine gap", color="#1f77b4")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax.set_title("Structure forms early, behavior follows late (supcon-aug, seed 0)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig4_4_probes.{ext}"), dpi=180)
    plt.close(fig)

    # fig 5: dose monotonicity
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    lams = [0.0, 0.03, 0.1, 0.3]
    for s, mk in ((0, "o"), (3, "s")):
        ys = [next(r["t_gen"] for r in stats["e4_dose"] if r["lam"] == l and r["seed"] == s)
              for l in lams]
        ax.plot(lams, ys, marker=mk, lw=1.5, label=f"seed {s}")
    ax.set(xlabel="lambda (aux weight)", ylabel="t_gen (steps)",
           title="Free-norm dose: no dose wins")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig4_5_dose.{ext}"), dpi=180)
    plt.close(fig)

    print(json.dumps(dict(
        n_runs=len(rows),
        e2_ratio_c92=stats["e2_matched_norm"]["per_norm"][92]["ratio_base_over_aug"],
        e2_base_c92_censored=stats["e2_matched_norm"]["per_norm"][92]["base"]["censored"],
        slope_ratio=stats["e2_matched_norm"]["slope_fits"]["slope_ratio_base_over_aug"],
        e3_tally=stats["e3_nn"]["tally"],
        free_aug_median_speedup=stats["free_norm"]["supcon_aug"]["median_speedup"],
        free_label_median_speedup=stats["free_norm"]["supcon_label"]["median_speedup"],
    ), indent=2))


if __name__ == "__main__":
    main()
