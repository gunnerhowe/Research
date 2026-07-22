"""E4 — steering-decontamination fix: project the novelty-SIGNAL direction (from
E3) out of the judge's residual stream and re-score, testing whether calibration
to SUBSTANTIVE novelty is recovered.

For each judge we re-score every S x G cell while ablating the G-direction at its
best E3 layer, sweeping alpha (0 = baseline, 1 = full projection, >1 = over-remove).
Endpoint (PLAN.md E4): beta_G shrinks monotonically in alpha AND calibration to S
improves (beta_S/|beta_G| rises, hackability -> 0.5). Off-target guard: beta_S must
not collapse and scores must not degenerate (variance preserved).

Usage: python experiments/run_e4.py [--judges ...] [--alphas 0 0.5 1 1.5 2]
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from novjudge.dataio import load_dataset, scoring_rows  # noqa: E402
from novjudge.rubric import pointwise_messages  # noqa: E402
from novjudge.estimate import fit_mixed, hackability_index  # noqa: E402

DEFAULT_JUDGES = [
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset_v1.json"))
    ap.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    ap.add_argument("--e3", default=str(ROOT / "results" / "e3_probe.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "e4_steering.json"))
    ap.add_argument("--stamp", default="")
    args = ap.parse_args()

    rows = scoring_rows(load_dataset(args.data)["stems"])
    e3 = json.loads(Path(args.e3).read_text())

    import torch
    from novjudge.judge_local import load_judge, expected_score, ablate

    out = {"meta": {"stamp": args.stamp, "alphas": args.alphas}, "per_judge": {}}
    for model_id in args.judges:
        best_layer = e3["per_judge"][model_id]["best_layer"]
        short = model_id.split("/")[-1]
        d = np.load(ROOT / "results" / "directions" / f"{short}__L{best_layer}__Gdir.npy")
        direction = torch.tensor(d, dtype=torch.float16)
        print(f"\n=== {model_id}  (ablate G-dir at layer {best_layer}) ===")

        j = load_judge(model_id, quantize=True)
        sweep = {}
        for alpha in args.alphas:
            long = []
            for r in rows:
                msgs = pointwise_messages(r["text"], r["prior_work"])
                if alpha == 0:
                    y = expected_score(j, msgs)
                else:
                    with ablate(j, best_layer, direction, alpha=alpha):
                        y = expected_score(j, msgs)
                long.append({**{k: r[k] for k in ("stem", "domain", "cell", "s", "g")},
                             "y": y, "judge": model_id})
            df = pd.DataFrame(long)
            fit = fit_mixed(df, judge=model_id, B=1000)
            hack = hackability_index(df, judge=model_id, B=1000)
            sweep[alpha] = {"beta_S": fit.beta_S, "beta_G": fit.beta_G,
                            "ratio_S_over_G": fit.ratio_S_over_G,
                            "hackability": hack["hackability_index"],
                            "y_std": float(df["y"].std())}
            print(f"  alpha={alpha}: beta_S={fit.beta_S:+.3f} beta_G={fit.beta_G:+.3f} "
                  f"ratio={fit.ratio_S_over_G:.2f} hack={hack['hackability_index']:.2f} "
                  f"y_std={df['y'].std():.2f}")
        out["per_judge"][model_id] = {"best_layer": best_layer, "sweep": sweep}
        del j.model; gc.collect(); torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
