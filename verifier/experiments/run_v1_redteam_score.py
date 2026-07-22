"""Score the v1 red-team: for each low-substance idea, compare
  base            = the plain incremental idea,
  attacked        = adversarial rewrite using HELD-OUT rhetoric (no known lexicon),
  neutralized     = the independent v1 neutralizer applied to `attacked`.
Per judge:
  baseline attack gain = Y(attacked) - Y(base)     (does a novel attack fool the naive judge?)
  v1 residual          = Y(neutralized) - Y(base)  (does v1 strip the novel attack?)
v1 generalizes if attack gain > 0 while v1 residual ~ 0.

Usage: python experiments/run_v1_redteam_score.py <redteam_workflow_output.json>
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
from novjudge.signal import lexicon_count  # noqa: E402

JUDGES = ["microsoft/Phi-3.5-mini-instruct", "Qwen/Qwen2.5-7B-Instruct",
          "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"]


def main():
    raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    items = raw.get("result", raw)["items"]
    pw = {s["stem_id"]: s["prior_work"] for s in load_dataset(str(ROOT / "data" / "dataset_v1.json"))["stems"]}
    items = [it for it in items if it["stem_id"] in pw]
    print(f"red-team stems: {len(items)}")
    # confirm the attack avoided the known lexicon
    lex_att = np.mean([lexicon_count(it["attacked"]) for it in items])
    lex_base = np.mean([lexicon_count(it["base"]) for it in items])
    print(f"known-lexicon words/idea: base {lex_base:.2f}, attacked {lex_att:.2f} "
          f"(attack used held-out rhetoric)")

    import torch
    from novjudge.judge_local import load_judge, expected_score

    out = {"meta": {"n": len(items), "lex_base": float(lex_base), "lex_attacked": float(lex_att)},
           "per_judge": {}}
    print("\n=== v1 red-team (held-out rhetoric) ===")
    for jid in JUDGES:
        j = load_judge(jid, quantize=True)
        yb, ya, yn = [], [], []
        for it in items:
            p = pw[it["stem_id"]]
            yb.append(expected_score(j, pointwise_messages(it["base"], p)))
            ya.append(expected_score(j, pointwise_messages(it["attacked"], p)))
            yn.append(expected_score(j, pointwise_messages(it["neutralized"], p)))
        yb, ya, yn = map(np.array, (yb, ya, yn))
        atk_gain, v1_resid = float((ya - yb).mean()), float((yn - yb).mean())
        out["per_judge"][jid] = {
            "Y_base": float(yb.mean()), "Y_attacked": float(ya.mean()), "Y_neutralized": float(yn.mean()),
            "attack_gain_naive": atk_gain, "v1_residual": v1_resid,
            "attack_blocked_frac": float((yn <= yb + 0.1).mean()),
        }
        print(f"{jid.split('/')[-1]}: base {yb.mean():.2f} | attacked {ya.mean():.2f} "
              f"(naive gain {atk_gain:+.2f}) | v1 {yn.mean():.2f} (v1 residual {v1_resid:+.2f})  "
              f"blocked {out['per_judge'][jid]['attack_blocked_frac']:.0%}")
        del j.model; gc.collect(); torch.cuda.empty_cache()

    (ROOT / "results" / "v1_redteam.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/v1_redteam.json  (naive gain>0 & v1 residual~0 => v1 generalizes)")


if __name__ == "__main__":
    main()
