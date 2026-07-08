"""Regenerate all paper figures from committed raw JSONL + results JSONs.

Usage: python experiments/make_figures.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from faithsel.analysis import augment, load_rows  # noqa: E402
from faithsel import figures as F  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIGS = os.path.join(ROOT, "paper", "figs")


def main():
    os.makedirs(FIGS, exist_ok=True)
    raw_main = os.path.join(ROOT, "results", "raw", "qwen7b_e0.jsonl")
    res_main_path = None
    for cand in ("qwen7b_e3.json", "qwen7b_e0.json"):
        p = os.path.join(ROOT, "results", cand)
        if os.path.exists(p):
            res_main_path = p
            break

    if os.path.exists(raw_main):
        df = augment(load_rows(raw_main))
        F.fig_confound(df, path=os.path.join(FIGS, "fig1_confound.pdf"))
        F.fig_lens(df, path=os.path.join(FIGS, "fig3_lens.pdf"))
        print("wrote fig1, fig3")
    if res_main_path:
        res = F.load_results(res_main_path)
        F.fig_correction(res, path=os.path.join(FIGS, "fig2_correction.pdf"))
        F.fig_sensitivity(res, path=os.path.join(FIGS, "fig4_sensitivity.pdf"))
        print("wrote fig2, fig4")

    model_results, labels = [], []
    for tag, lab in (("qwen7b_e3", "Qwen2.5-7B"),
                     ("mistral7b_e3", "Mistral-7B"),
                     ("phi35_e3", "Phi-3.5-mini")):
        p = os.path.join(ROOT, "results", f"{tag}.json")
        if os.path.exists(p):
            model_results.append(F.load_results(p))
            labels.append(lab)
    if model_results:
        F.fig_models(model_results, labels,
                     path=os.path.join(FIGS, "fig5_models.pdf"))
        print("wrote fig5")


if __name__ == "__main__":
    main()
