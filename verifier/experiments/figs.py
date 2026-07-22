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


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "e0"
    if which in ("e0", "all"):
        fig_e0()
