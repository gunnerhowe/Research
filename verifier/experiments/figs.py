"""Figures from committed results (no GPU). Grows as the ladder completes."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def short(j):
    return j.split("/")[-1].replace("-instruct", "").replace("-Instruct", "")


def fig_e0():
    d = json.load(open(ROOT / "results" / "e0_results.json"))
    judges = d["meta"]["judges"]
    labels = [short(j) for j in judges] + ["POOLED"]
    bS = [d["per_judge"][j]["beta_S"] for j in judges] + [d["pooled"]["beta_S"]]
    bG = [d["per_judge"][j]["beta_G"] for j in judges] + [d["pooled"]["beta_G"]]
    ciS = [d["per_judge"][j]["ci_S"] for j in judges] + [d["pooled"]["ci_S"]]
    ciG = [d["per_judge"][j]["ci_G"] for j in judges] + [d["pooled"]["ci_G"]]
    hack = [d["hackability"][j]["hackability_index"] for j in judges] + [d["pooled_hackability"]["hackability_index"]]
    hackci = [d["hackability"][j]["ci"] for j in judges] + [d["pooled_hackability"]["ci"]]

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    def err(ci, b):
        return [[b[i] - ci[i][0] for i in range(len(b))], [ci[i][1] - b[i] for i in range(len(b))]]

    w = 0.34
    ax1.bar(x - w / 2, bS, w, yerr=err(ciS, bS), capsize=3, color="#4C9F70", label="β_S  (substantive novelty)")
    ax1.bar(x + w / 2, bG, w, yerr=err(ciG, bG), capsize=3, color="#C1666B", label="β_G  (novelty signal)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("effect on judge novelty score (1–9)")
    ax1.set_title("Signal effect exceeds substance effect (E0)")
    ax1.legend(fontsize=8, loc="upper left"); ax1.axhline(0, color="k", lw=0.6)

    ax2.bar(x, hack, 0.55, yerr=err(hackci, hack), capsize=3, color="#6272A4")
    ax2.axhline(0.5, color="k", ls="--", lw=0.9, label="chance (signal = substance)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("P( signaled non-novel  >  plain novel )")
    ax2.set_title("Hackability index")
    ax2.set_ylim(0, 1); ax2.legend(fontsize=8)

    fig.suptitle("LLM research judges score novelty SIGNAL over SUBSTANCE  (N=154 stems, 616 items)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_e0_effects.{ext}", bbox_inches="tight", dpi=150)
    print(f"wrote {FIGS/'fig_e0_effects.png'}")


def fig_e5():
    d = json.load(open(ROOT / "results" / "e5_reconciliation.json"))
    judges = [j for j in d["per_judge"]]
    labels = [short(j) for j in judges]
    vG = [d["per_judge"][j]["var_share_G"] for j in judges]
    vS = [d["per_judge"][j]["var_share_S"] for j in judges]
    rem = [max(0, 1 - g - s) for g, s in zip(vG, vS)]
    over = [d["per_judge"][j]["over_rating_rate"] for j in judges]
    under = [d["per_judge"][j]["under_rating_rate"] for j in judges]
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1.bar(x, vG, 0.6, color="#C1666B", label="signal G")
    ax1.bar(x, vS, 0.6, bottom=vG, color="#4C9F70", label="substance S")
    ax1.bar(x, rem, 0.6, bottom=[g + s for g, s in zip(vG, vS)], color="#cccccc", label="interaction")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("share of judge-score variance"); ax1.set_ylim(0, 1)
    ax1.set_title("Score variance is driven by SIGNAL, not substance")
    ax1.legend(fontsize=8)
    w = 0.34
    ax2.bar(x - w / 2, over, w, color="#C1666B", label="over-rate: non-novel > avg novel")
    ax2.bar(x + w / 2, under, w, color="#6272A4", label="under-rate: novel < avg non-novel")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("rate"); ax2.set_title("Both tails from one mechanism")
    ax2.legend(fontsize=8)
    fig.suptitle("E5: one signal-tracking mechanism reconciles over-rating (RQ-Bench) "
                 "and compression (RINoBench)", fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_e5_reconciliation.{ext}", bbox_inches="tight", dpi=150)
    print(f"wrote {FIGS/'fig_e5_reconciliation.png'}")


def fig_e4():
    d = json.load(open(ROOT / "results" / "e4_multilayer.json"))
    judges = list(d["per_judge"])
    labels = [short(j) for j in judges]
    x = np.arange(len(labels))
    bS0 = [d["per_judge"][j]["sweep"]["0.0"]["beta_S"] for j in judges]
    bG0 = [d["per_judge"][j]["sweep"]["0.0"]["beta_G"] for j in judges]
    bS1 = [d["per_judge"][j]["sweep"]["1.0"]["beta_S"] for j in judges]
    bG1 = [d["per_judge"][j]["sweep"]["1.0"]["beta_G"] for j in judges]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    w = 0.2
    ax.bar(x - 1.5 * w, bS0, w, color="#4C9F70", label="β_S baseline")
    ax.bar(x - 0.5 * w, bS1, w, color="#4C9F70", alpha=0.5, hatch="//", label="β_S ablated")
    ax.bar(x + 0.5 * w, bG0, w, color="#C1666B", label="β_G baseline")
    ax.bar(x + 1.5 * w, bG1, w, color="#C1666B", alpha=0.5, hatch="//", label="β_G ablated")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("effect on novelty score")
    ax.set_title("E4: ablating the signal direction does NOT selectively remove β_G\n"
                 "(β_G survives or β_S dies with it — no recalibration)", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_e4_steering_fails.{ext}", bbox_inches="tight", dpi=150)
    print(f"wrote {FIGS/'fig_e4_steering_fails.png'}")


def fig_v1():
    d = json.load(open(ROOT / "results" / "v1_neutralized.json"))
    judges = list(d["per_judge"])
    labels = [short(j) for j in judges]
    x = np.arange(len(labels))
    g0 = [d["per_judge"][j]["e0_beta_G"] for j in judges]
    g1 = [d["per_judge"][j]["beta_G"] for j in judges]
    s0 = [d["per_judge"][j]["e0_beta_S"] for j in judges]
    s1 = [d["per_judge"][j]["beta_S"] for j in judges]
    h0 = [d["per_judge"][j]["e0_hack"] for j in judges]
    h1 = [d["per_judge"][j]["hackability"] for j in judges]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    w = 0.2
    ax1.bar(x - 1.5 * w, g0, w, color="#C1666B", label="β_G baseline")
    ax1.bar(x - 0.5 * w, g1, w, color="#C1666B", alpha=0.45, hatch="//", label="β_G after v1")
    ax1.bar(x + 0.5 * w, s0, w, color="#4C9F70", label="β_S baseline")
    ax1.bar(x + 1.5 * w, s1, w, color="#4C9F70", alpha=0.45, hatch="//", label="β_S after v1")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("effect on novelty score"); ax1.legend(fontsize=8, ncol=2)
    ax1.set_title("v1 closes the signal channel (β_G→0), keeps substance (β_S)")
    ax2.bar(x - w / 2, h0, w, color="#6272A4", label="baseline judge")
    ax2.bar(x + w / 2, h1, w, color="#6272A4", alpha=0.45, hatch="//", label="v1 (neutralize→judge)")
    ax2.axhline(0.5, color="k", ls="--", lw=0.9, label="chance")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("hackability index"); ax2.set_ylim(0, 1); ax2.legend(fontsize=8)
    ax2.set_title("Hackability 0.82 → 0.16 (below chance)")
    fig.suptitle("v1 verifier: input-space decontamination succeeds where activation-"
                 "steering (E4) failed", fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_v1_neutralize.{ext}", bbox_inches="tight", dpi=150)
    print(f"wrote {FIGS/'fig_v1_neutralize.png'}")


def fig_verifiers():
    e0 = json.load(open(ROOT / "results" / "e0_results.json"))
    v1 = json.load(open(ROOT / "results" / "v1_neutralized.json"))
    v2 = json.load(open(ROOT / "results" / "v2_grounded.json"))
    rt = json.load(open(ROOT / "results" / "v1_redteam.json"))
    js = list(e0["per_judge"])
    naive_hack = np.mean([e0["hackability"][j]["hackability_index"] for j in js])
    v1_hack = np.mean([v1["per_judge"][j]["hackability"] for j in js])
    v2_hack = v2["framed"]["hackability"]
    naive_rt = np.mean([rt["per_judge"][j]["attack_gain_naive"] for j in js])
    v1_rt = np.mean([rt["per_judge"][j]["v1_residual"] for j in js])
    v2_rt = v2["redteam_attack_gain_mean"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    labels = ["naive judge", "v1 neutralize", "v2 grounded"]
    cols = ["#C1666B", "#E0A458", "#4C9F70"]
    ax1.bar(labels, [naive_hack, v1_hack, v2_hack], color=cols)
    ax1.axhline(0.5, color="k", ls="--", lw=0.9, label="chance")
    ax1.set_ylabel("hackability (framed 2×2)"); ax1.set_ylim(0, 1)
    ax1.set_title("On the KNOWN attack: lower is better"); ax1.legend(fontsize=8)
    ax2.bar(labels, [naive_rt, v1_rt, v2_rt], color=cols)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_ylabel("score gain from held-out attack")
    ax2.set_title("On the RED-TEAM (novel rhetoric): lower is better")
    fig.suptitle("Building a better verifier: v1 beats the known attack, v2 generalizes to novel ones",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_verifiers_compare.{ext}", bbox_inches="tight", dpi=150)
    print(f"wrote {FIGS/'fig_verifiers_compare.png'}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "e0"
    if which in ("e0", "all"):
        fig_e0()
    if which in ("e5", "all"):
        fig_e5()
    if which in ("e4", "all"):
        fig_e4()
    if which in ("v1", "all"):
        fig_v1()
    if which in ("verifiers", "all"):
        fig_verifiers()
