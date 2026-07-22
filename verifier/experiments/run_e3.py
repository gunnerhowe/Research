"""E3 — mechanism: probe the judge's residual stream for a linear novelty-SIGNAL
direction, and test that it generalizes OUT OF DOMAIN (train ML -> test econ).

For each judge: capture the scoring-position residual at a depth sweep, then per
layer report (a) in-distribution stem-grouped CV AUROC for G-high vs G-low, and
(b) OOD AUROC across domains. Saves the difference-of-means G-direction per layer
(consumed by the E4 steering-decontamination fix).

Usage: python experiments/run_e3.py [--judges ...] [--out results/e3_probe.json]
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from novjudge.dataio import load_dataset, scoring_rows  # noqa: E402
from novjudge.probe import (capture_dataset, probe_in_distribution,  # noqa: E402
                            probe_ood_domain, diff_of_means_direction)

DEFAULT_JUDGES = [
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset_v1.json"))
    ap.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES)
    ap.add_argument("--out", default=str(ROOT / "results" / "e3_probe.json"))
    ap.add_argument("--stamp", default="")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N stems")
    args = ap.parse_args()

    stems = load_dataset(args.data)["stems"]
    if args.limit:
        stems = stems[:args.limit]
    rows = scoring_rows(stems)
    print(f"{len(rows)} items")
    dirdir = ROOT / "results" / "directions"
    dirdir.mkdir(parents=True, exist_ok=True)

    import torch
    from novjudge.judge_local import load_judge

    out = {"meta": {"data": Path(args.data).name, "stamp": args.stamp, "judges": args.judges},
           "per_judge": {}}
    for model_id in args.judges:
        print(f"\n=== {model_id} ===")
        j = load_judge(model_id, quantize=True)
        n_layers = len(j.model.model.layers)
        layers = sorted({int(round(f * n_layers)) for f in
                         (0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85)})
        layers = [L for L in layers if 0 <= L < n_layers]
        print(f"n_layers={n_layers}; sweeping {layers}")
        X, lab = capture_dataset(j, rows, layers)
        del j.model; gc.collect(); torch.cuda.empty_cache()

        short = model_id.split("/")[-1]
        per_layer = {}
        for L in layers:
            ind = probe_in_distribution(X[L], lab["g"], lab["stem"])
            ood_me = probe_ood_domain(X[L], lab["g"], lab["domain"], "ml", "econ")
            ood_em = probe_ood_domain(X[L], lab["g"], lab["domain"], "econ", "ml")
            per_layer[L] = {"auroc_in": ind["auroc_mean"], "auroc_in_std": ind["auroc_std"],
                            "auroc_ood_ml2econ": ood_me["auroc"], "auroc_ood_econ2ml": ood_em["auroc"]}
            d = diff_of_means_direction(X[L], lab["g"])
            np.save(dirdir / f"{short}__L{L}__Gdir.npy", d)
            print(f"  L{L:>2}: in-dist AUROC {ind['auroc_mean']:.3f}  "
                  f"OOD ml->econ {ood_me['auroc']:.3f}  econ->ml {ood_em['auroc']:.3f}")
        best = max(per_layer, key=lambda L: per_layer[L]["auroc_in"])
        out["per_judge"][model_id] = {"n_layers": n_layers, "layers": layers,
                                      "per_layer": per_layer, "best_layer": best,
                                      "best_auroc_in": per_layer[best]["auroc_in"]}
        print(f"  best layer {best}: in-dist AUROC {per_layer[best]['auroc_in']:.3f}")

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}  (+ directions in results/directions/)")


if __name__ == "__main__":
    main()
