"""E1 robustness — pairwise: is E0 an artifact of the pointwise digit-readout?

Head-to-head under the frozen pairwise rubric, the reward-hack matchup:
  LH (low-substance, SIGNALED)  vs  HL (high-substance, PLAIN).
A/B order is randomized per stem (both orders scored) to cancel position bias.
Endpoint: P(judge picks the signaled non-novel idea) — > 0.5 replicates the
signal-beats-substance effect in a comparative setting the readout can't fake.

Usage: python experiments/run_e1_pairwise.py [--judges ...]
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from novjudge.dataio import load_dataset  # noqa: E402
from novjudge.rubric import pairwise_messages  # noqa: E402

DEFAULT_JUDGES = [
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset_v1.json"))
    ap.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES)
    ap.add_argument("--out", default=str(ROOT / "results" / "e1_pairwise.json"))
    ap.add_argument("--stamp", default="")
    args = ap.parse_args()

    stems = load_dataset(args.data)["stems"]

    import torch
    from novjudge.judge_local import load_judge, pairwise_prob_A

    out = {"meta": {"stamp": args.stamp, "n_stems": len(stems),
                    "matchup": "LH (low-substance signaled) vs HL (high-substance plain)"},
           "per_judge": {}}
    for model_id in args.judges:
        print(f"\n=== {model_id} ===")
        j = load_judge(model_id, quantize=True)
        pick_signaled = []
        for s in stems:
            lh, hl, pw = s["LH"], s["HL"], s["prior_work"]
            # order 1: A=LH(signaled), B=HL(plain) -> P(A)=P(pick signaled)
            p1 = pairwise_prob_A(j, pairwise_messages(lh, hl, pw))
            # order 2: A=HL(plain), B=LH(signaled) -> P(pick signaled)=1-P(A)
            p2 = pairwise_prob_A(j, pairwise_messages(hl, lh, pw))
            pick_signaled.append(0.5 * (p1 + (1.0 - p2)))
        arr = np.array(pick_signaled)
        # stem bootstrap CI on the mean
        rng = np.random.default_rng(0)
        boots = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(2000)]
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        out["per_judge"][model_id] = {
            "P_pick_signaled_mean": float(arr.mean()), "ci": ci,
            "frac_stems_signaled_wins": float(np.mean(arr > 0.5)),
        }
        print(f"  P(pick signaled non-novel over plain novel) = {arr.mean():.3f}  CI{tuple(round(x,3) for x in ci)}")
        print(f"  signaled wins in {np.mean(arr > 0.5):.0%} of stems")
        del j.model; gc.collect(); torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
