"""V0 — the ceiling test for a presentation-invariant verifier.

Score the DE-RHETORICIZED technical statements directly (the dataset's
s_low_content / s_high_content — the neutral content with no framing at all).
Question: with rhetoric absent, does the judge track SUBSTANCE?
  - substance margin = mean[ Y(s_high_content) - Y(s_low_content) ] per judge;
  - tracking rate    = P( Y(high) > Y(low) ) over stems.
If the margin is clean and positive, "strip rhetoric then judge" is a viable
verifier (V1); if not, the judge can't tell real novelty even without hype, and
we need a grounded verifier instead.
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from novjudge.dataio import load_dataset  # noqa: E402
from novjudge.rubric import pointwise_messages  # noqa: E402

JUDGES = ["microsoft/Phi-3.5-mini-instruct", "Qwen/Qwen2.5-7B-Instruct",
          "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"]


def main():
    stems = load_dataset(str(ROOT / "data" / "dataset_v1.json"))["stems"]
    import torch
    from novjudge.judge_local import load_judge, expected_score

    out = {"per_judge": {}}
    for jid in JUDGES:
        j = load_judge(jid, quantize=True)
        lo = np.array([expected_score(j, pointwise_messages(s["s_low_content"], s["prior_work"])) for s in stems])
        hi = np.array([expected_score(j, pointwise_messages(s["s_high_content"], s["prior_work"])) for s in stems])
        diff = hi - lo
        rng = np.random.default_rng(0)
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        out["per_judge"][jid] = {
            "mean_low": float(lo.mean()), "mean_high": float(hi.mean()),
            "substance_margin": float(diff.mean()), "ci": ci,
            "tracking_rate": float(np.mean(diff > 0)),
        }
        print(f"{jid.split('/')[-1]}: Y(low_content)={lo.mean():.2f} Y(high_content)={hi.mean():.2f}  "
              f"substance margin={diff.mean():+.2f} CI{tuple(round(x,2) for x in ci)}  "
              f"tracks S in {np.mean(diff>0):.0%} of stems")
        del j.model; gc.collect(); torch.cuda.empty_cache()

    (ROOT / "results" / "v0_content_ceiling.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/v0_content_ceiling.json")
    print("Compare to E0 framed-cell beta_S (Phi +0.40 / Qwen +1.14 / Nemo +0.47):"
          " a clean positive margin here = strip-then-judge is viable (V1).")


if __name__ == "__main__":
    main()
