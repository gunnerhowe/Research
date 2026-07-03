"""Aggregate grid runs -> stats + publication figures.

Outputs into analysis/out/:
  results.csv           per-run summary table
  stats.json            per-condition medians/CIs + pairwise Mann-Whitney tests
  fig1_curves.pdf/png   test-accuracy trajectories per condition
  fig2_delay.pdf/png    epochs-to-generalization + delay per condition
  fig3_reps.pdf/png     representation-timing probes (Fourier, cluster, CKA)
  fig4_mediation.pdf/png weight norm + logit scale (norm-matched check)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, "runs", "grid")
OUT = os.path.join(ROOT, "analysis", "out")
os.makedirs(OUT, exist_ok=True)

PRIMARY_LAM = float(os.environ.get("PRIMARY_LAM", "0.3"))
COND_LABELS = {
    "baseline": "Baseline",
    f"supcon_true_lam{PRIMARY_LAM}": "SupCon-true",
    f"supcon_shuffled_lam{PRIMARY_LAM}": "SupCon-shuffled",
    "grokfast": "Grokfast",
    f"norm_matched_lam{PRIMARY_LAM}": "Norm-matched",
}
COLORS = {
    "Baseline": "#444444",
    "SupCon-true": "#d62728",
    "SupCon-shuffled": "#1f77b4",
    "Grokfast": "#2ca02c",
    "Norm-matched": "#9467bd",
}


def cond_key(run_name: str) -> str:
    return run_name.rsplit("_s", 1)[0]


def load_runs():
    rows, logs = [], {}
    for sdir in sorted(glob.glob(os.path.join(GRID, "*", "summary.json"))):
        run_dir = os.path.dirname(sdir)
        name = os.path.basename(run_dir)
        s = json.load(open(sdir))
        key = cond_key(name)
        censored = s["t_gen"] is None
        # Censored runs never generalized within budget: treat t_gen as the budget
        # (a conservative lower bound) and flag them.
        t_gen = s["epochs"] if censored else s["t_gen"]
        delay = (t_gen - s["t_fit"]) if s["t_fit"] is not None else None
        L = pd.DataFrame([json.loads(l) for l in
                          open(os.path.join(run_dir, "metrics.jsonl"))])
        # Earlier-threshold crossings: usable when t_gen (0.95) is censored.
        def crossing(thresh):
            hit = L[L.test_acc >= thresh]
            return int(hit.epoch.iloc[0]) if len(hit) else None
        rows.append(dict(run=name, cond=key, seed=s["seed"], t_fit=s["t_fit"],
                         t_gen=t_gen, delay=delay, censored=censored,
                         t10=crossing(0.10), t50=crossing(0.50),
                         final_test_acc=s["final_test_acc"], epochs_ran=s["epochs_ran"]))
        logs[name] = L
    return pd.DataFrame(rows), logs


def linear_cka_np(X, Y):
    X = X.astype(np.float64) - X.mean(0).astype(np.float64)
    Y = Y.astype(np.float64) - Y.mean(0).astype(np.float64)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    return hsic / max(np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y), 1e-12)


def cka_trajectory(run_dir):
    z = np.load(os.path.join(run_dir, "snaps.npz"))
    snaps, epochs = z["snaps"], z["epochs"]
    final = snaps[-1]
    return epochs[:-1], np.array([linear_cka_np(s, final) for s in snaps[:-1]])


def boot_ci(vals, n=10000, seed=0):
    vals = np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    if len(vals) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(vals, size=(n, len(vals))), axis=1)
    return float(np.median(vals)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main():
    df, logs = load_runs()
    df.to_csv(os.path.join(OUT, "results.csv"), index=False)
    print(df.to_string())

    conds = [c for c in COND_LABELS if c in set(df.cond)]
    stats = {}
    for c in conds:
        sub = df[df.cond == c]
        med, lo, hi = boot_ci(sub.t_gen.tolist())
        dmed, dlo, dhi = boot_ci(sub.delay.tolist())
        stats[COND_LABELS[c]] = dict(n=len(sub), n_censored=int(sub.censored.sum()),
                                     t_gen_median=med, t_gen_ci=[lo, hi],
                                     delay_median=dmed, delay_ci=[dlo, dhi],
                                     t_gen_all=sub.t_gen.tolist(), t_fit_all=sub.t_fit.tolist())
    base_tg = df[df.cond == "baseline"].t_gen.dropna()
    tests = {}
    for c in conds:
        if c == "baseline":
            continue
        x = df[df.cond == c].t_gen.dropna()
        if len(x) and len(base_tg):
            u, pv = mannwhitneyu(x, base_tg, alternative="two-sided")
            tests[COND_LABELS[c]] = dict(U=float(u), p=float(pv),
                                         speedup_vs_baseline=float(base_tg.median() / x.median())
                                         if x.median() else None)
    # Paired per-seed contrasts (same seed = same init & split): t_gen ratios vs baseline,
    # censored values already set to budget (conservative), Wilcoxon signed-rank.
    from scipy.stats import wilcoxon
    paired = {}
    base_by_seed = df[df.cond == "baseline"].set_index("seed").t_gen
    for c in sorted(set(df.cond) - {"baseline"}):
        sub = df[df.cond == c].set_index("seed")
        common = sub.index.intersection(base_by_seed.index)
        if len(common) < 3:
            continue
        x, b = sub.loc[common].t_gen.astype(float), base_by_seed.loc[common].astype(float)
        try:
            w, pv = wilcoxon(x, b)
        except ValueError:
            w, pv = None, None
        paired[c] = dict(n_pairs=int(len(common)),
                         ratios={int(s): round(float(x[s] / b[s]), 3) for s in common},
                         median_ratio=float((x / b).median()),
                         n_censored=int(sub.loc[common].censored.sum()),
                         wilcoxon_W=w if w is None else float(w),
                         wilcoxon_p=pv if pv is None else float(pv))
    json.dump(dict(per_condition=stats, mannwhitney_vs_baseline=tests, paired_vs_baseline=paired),
              open(os.path.join(OUT, "stats.json"), "w"), indent=2)
    print(json.dumps(tests, indent=2))
    print(json.dumps(paired, indent=2))

    # ---- Fig 1: test-accuracy curves (per-seed thin, median thick) ----
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for c in conds:
        lab = COND_LABELS[c]
        runs = df[df.cond == c].run
        curves = []
        for r in runs:
            L = logs[r]
            ax.plot(L.epoch, L.test_acc, color=COLORS[lab], alpha=0.18, lw=0.7)
            curves.append(L.set_index("epoch").test_acc)
        if curves:
            M = pd.concat(curves, axis=1).ffill()
            ax.plot(M.index, M.median(axis=1), color=COLORS[lab], lw=2.0, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_curves.{ext}"), dpi=300)
    plt.close(fig)

    # ---- Fig 2: t_gen and delay per condition ----
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for ax, col, title in [(axes[0], "t_gen", "Epochs to generalization"),
                           (axes[1], "delay", "Delay (t_gen − t_fit)")]:
        for i, c in enumerate(conds):
            lab = COND_LABELS[c]
            vals = df[df.cond == c][col].dropna()
            ax.scatter([i] * len(vals), vals, color=COLORS[lab], s=18, zorder=3)
            if len(vals):
                ax.hlines(vals.median(), i - 0.25, i + 0.25, color=COLORS[lab], lw=2.5)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([COND_LABELS[c] for c in conds], rotation=25, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_delay.{ext}"), dpi=300)
    plt.close(fig)

    # ---- Fig 3: representation timing ----
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for c in conds:
        lab = COND_LABELS[c]
        runs = df[df.cond == c].run
        f_curves, g_curves, cka_all = [], [], []
        for r in runs:
            L = logs[r]
            f_curves.append(L.set_index("epoch").fourier_top8)
            g_curves.append(L.set_index("epoch").cos_gap)
            try:
                ep, cka = cka_trajectory(os.path.join(GRID, r))
                cka_all.append(pd.Series(cka, index=ep))
            except Exception:
                pass
        if f_curves:
            M = pd.concat(f_curves, axis=1).ffill()
            axes[0].plot(M.index, M.median(axis=1), color=COLORS[lab], lw=1.8, label=lab)
            M = pd.concat(g_curves, axis=1).ffill()
            axes[1].plot(M.index, M.median(axis=1), color=COLORS[lab], lw=1.8)
        if cka_all:
            M = pd.concat(cka_all, axis=1).ffill()
            axes[2].plot(M.index, M.median(axis=1), color=COLORS[lab], lw=1.8)
    for ax, ylab in zip(axes, ["Embedding Fourier power in top-8 freqs",
                               "Class cosine gap (within − between)",
                               "CKA with final (post-grok) reps"]):
        ax.set_xscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig3_reps.{ext}"), dpi=300)
    plt.close(fig)

    # ---- Fig 4: mediation checks ----
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for c in conds:
        lab = COND_LABELS[c]
        runs = df[df.cond == c].run
        w_curves, ls_curves = [], []
        for r in runs:
            L = logs[r]
            w_curves.append(L.set_index("epoch").wnorm)
            ls_curves.append(L.set_index("epoch").logit_scale)
        if w_curves:
            M = pd.concat(w_curves, axis=1).ffill()
            axes[0].plot(M.index, M.median(axis=1), color=COLORS[lab], lw=1.8, label=lab)
            M = pd.concat(ls_curves, axis=1).ffill()
            axes[1].plot(M.index, M.median(axis=1), color=COLORS[lab], lw=1.8)
    axes[0].set_ylabel("Total weight norm")
    axes[1].set_ylabel("Logit scale (mean ||logits||)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Epoch")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig4_mediation.{ext}"), dpi=300)
    plt.close(fig)

    # ---- Fig 5: dose-response over lambda (true vs shuffled structure) ----
    dose = df[df.cond.str.startswith("supcon_")].copy()
    if len(dose):
        dose["family"] = np.where(dose.cond.str.startswith("supcon_true"), "True structure",
                                  "Shuffled structure")
        dose["lam"] = dose.cond.str.extract(r"lam([\d.]+)").astype(float)
        base_med = df[df.cond == "baseline"].t_gen.median()
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for fam, color in [("True structure", "#d62728"), ("Shuffled structure", "#1f77b4")]:
            sub = dose[dose.family == fam]
            meds = sub.groupby("lam").t_gen.median()
            ax.plot(meds.index, meds.values, "o-", color=color, label=fam)
            cens = sub[sub.censored]
            ax.scatter(cens.lam, cens.t_gen, marker="^", color=color, s=42, zorder=4)
            ax.scatter(sub.lam, sub.t_gen, color=color, s=10, alpha=0.4, zorder=3)
        if not np.isnan(base_med):
            ax.axhline(base_med, color="#444444", ls="--", lw=1.2, label="Baseline")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Contrastive strength $\lambda$")
        ax.set_ylabel("Epochs to generalization")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"fig5_dose.{ext}"), dpi=300)
        plt.close(fig)

    # ---- Fig 6: survival curves per lambda (fraction not yet grokked vs epoch) ----
    budget = 50000

    def survival_xy(sub):
        ts = np.sort(sub.t_gen.values)
        n = len(ts)
        xs, ys = [0], [1.0]
        for i, t in enumerate(ts):
            if t >= budget:
                break
            xs += [t, t]
            ys += [ys[-1], 1.0 - (i + 1) / n]
        xs.append(budget)
        ys.append(ys[-1])
        return xs, ys

    lams = ["0.1", "0.3", "1.0"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True)
    for ax, lam in zip(axes, lams):
        panel = [("baseline", "Baseline", "-"),
                 (f"supcon_true_lam{lam}", "SupCon-true", "-"),
                 (f"norm_matched_lam{lam}", "Norm-matched", "-"),
                 (f"supcon_shuffled_lam{lam}", "SupCon-shuffled", "--"),
                 ("grokfast", "Grokfast", "-")]
        for c, lab, ls in panel:
            sub = df[df.cond == c]
            if not len(sub):
                continue
            xs, ys = survival_xy(sub)
            ax.step(xs, ys, where="post", color=COLORS[lab], lw=1.8, ls=ls,
                    label=f"{lab} (n={len(sub)})")
        ax.set_title(rf"$\lambda={lam}$", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylim(-0.03, 1.06)
        ax.legend(frameon=False, fontsize=6.5, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Fraction not yet generalized")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig6_survival.{ext}"), dpi=300)
    plt.close(fig)

    # ---- Pooled grok fractions + Fisher exact tests ----
    from scipy.stats import fisher_exact
    fam = {"supcon_true": df.cond.str.startswith("supcon_true"),
           "supcon_shuffled": df.cond.str.startswith("supcon_shuffled"),
           "norm_matched": df.cond.str.startswith("norm_matched"),
           "baseline": df.cond == "baseline", "grokfast": df.cond == "grokfast"}
    pooled = {}
    for k, mask in fam.items():
        sub = df[mask]
        pooled[k] = dict(grokked=int((~sub.censored).sum()), total=int(len(sub)))
    ft = {}
    t = pooled["supcon_true"]
    for other in ("supcon_shuffled", "norm_matched"):
        o = pooled[other]
        _, pv = fisher_exact([[t["grokked"], t["total"] - t["grokked"]],
                              [o["grokked"], o["total"] - o["grokked"]]])
        ft[f"true_vs_{other}"] = float(pv)
    with open(os.path.join(OUT, "stats.json")) as f:
        S = json.load(f)
    S["pooled_grok_fractions"] = pooled
    S["fisher_exact"] = ft
    json.dump(S, open(os.path.join(OUT, "stats.json"), "w"), indent=2)
    print(json.dumps({"pooled": pooled, "fisher": ft}, indent=2))

    print("wrote figures + stats to", OUT)


if __name__ == "__main__":
    main()
