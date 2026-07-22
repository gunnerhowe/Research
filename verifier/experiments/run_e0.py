"""E0 — the gate: does novelty SIGNAL beat SUBSTANCE?

Scores every S x G cell of the frozen dataset with each local judge (expected-digit
readout under the frozen pointwise rubric), fits Y ~ S + G + S:G per judge (+ a
pooled fit), and reports the signal effect beta_G, the substance effect beta_S,
the calibration ratio, and the hackability index P(Y[LH] > Y[HL]).

Usage:
  python experiments/run_e0.py [--data data/dataset_v1.json] \
      [--judges microsoft/Phi-3.5-mini-instruct Qwen/Qwen2.5-7B-Instruct nvidia/Llama-3.1-Nemotron-Nano-8B-v1] \
      [--out results/e0_results.json] [--stamp <iso>]
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from novjudge.dataio import load_dataset, scoring_rows  # noqa: E402
from novjudge.rubric import pointwise_messages, RUBRIC_IDS  # noqa: E402
from novjudge.estimate import fit_mixed, hackability_index  # noqa: E402

DEFAULT_JUDGES = [
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
]


def score_with_judge(model_id: str, rows: list[dict]) -> list[float]:
    import torch
    from novjudge.judge_local import load_judge, expected_score

    j = load_judge(model_id, quantize=True)
    ys = []
    for i, r in enumerate(rows):
        ys.append(expected_score(j, pointwise_messages(r["text"], r["prior_work"])))
        if (i + 1) % 100 == 0:
            print(f"  {model_id}: {i + 1}/{len(rows)}")
    del j.model
    gc.collect()
    torch.cuda.empty_cache()
    return ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset_v1.json"))
    ap.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES)
    ap.add_argument("--out", default=str(ROOT / "results" / "e0_results.json"))
    ap.add_argument("--stamp", default="")
    args = ap.parse_args()

    ds = load_dataset(args.data)
    stems = ds["stems"]
    rows = scoring_rows(stems)
    print(f"dataset: {len(stems)} stems, {len(rows)} items; judges: {args.judges}")

    long = []
    for model_id in args.judges:
        print(f"scoring with {model_id} ...")
        ys = score_with_judge(model_id, rows)
        for r, y in zip(rows, ys):
            long.append({**{k: r[k] for k in ("stem", "domain", "cell", "s", "g")},
                         "y": y, "judge": model_id})
    df = pd.DataFrame(long)

    fits, hacks = {}, {}
    for model_id in args.judges:
        fit = fit_mixed(df, judge=model_id)
        fits[model_id] = fit.as_dict()
        hacks[model_id] = hackability_index(df, judge=model_id)
        print(f"{model_id}: beta_S={fit.beta_S:+.3f}{fit.ci_S}  "
              f"beta_G={fit.beta_G:+.3f}{fit.ci_G}  ratio_S/G={fit.ratio_S_over_G:.2f}  "
              f"hack={hacks[model_id]['hackability_index']:.2f}")
    pooled = fit_mixed(df, judge="")
    pooled_hack = hackability_index(df, judge="")

    out = {
        "meta": {"data": Path(args.data).name, "n_stems": len(stems),
                 "n_items": len(rows), "judges": args.judges,
                 "rubric_ids": RUBRIC_IDS, "stamp": args.stamp},
        "per_judge": fits, "hackability": hacks,
        "pooled": pooled.as_dict(), "pooled_hackability": pooled_hack,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    # also persist the raw scores so figures/regen never re-run the GPU
    df.to_csv(Path(args.out).with_suffix(".scores.csv"), index=False)
    print(f"\npooled: beta_S={pooled.beta_S:+.3f}  beta_G={pooled.beta_G:+.3f}  "
          f"ratio={pooled.ratio_S_over_G:.2f}  hack={pooled_hack['hackability_index']:.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
