"""E4b — multi-layer decontamination: project the novelty-signal direction out at
a BAND of early-mid layers simultaneously (each layer using its own E3 direction),
so a single downstream layer cannot re-derive the signal. The pre-registered
'did you try harder?' robustness variant for the E4 steering fix.

Endpoint unchanged: beta_G shrinks while beta_S is preserved (selective
recalibration to substance), not both collapsing together (damage).
"""

import argparse
import contextlib
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
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 1.0])
    ap.add_argument("--max_depth", type=float, default=0.55, help="ablate layers up to this depth fraction")
    ap.add_argument("--e3", default=str(ROOT / "results" / "e3_probe.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "e4_multilayer.json"))
    ap.add_argument("--stamp", default="")
    args = ap.parse_args()

    rows = scoring_rows(load_dataset(args.data)["stems"])
    e3 = json.loads(Path(args.e3).read_text())

    import torch
    from novjudge.judge_local import load_judge, expected_score, ablate

    out = {"meta": {"stamp": args.stamp, "alphas": args.alphas, "max_depth": args.max_depth},
           "per_judge": {}}
    for model_id in args.judges:
        short = model_id.split("/")[-1]
        n_layers = e3["per_judge"][model_id]["n_layers"]
        band = [L for L in e3["per_judge"][model_id]["layers"] if L <= args.max_depth * n_layers]
        dirs = {}
        for L in band:
            p = ROOT / "results" / "directions" / f"{short}__L{L}__Gdir.npy"
            if p.exists():
                dirs[L] = torch.tensor(np.load(p), dtype=torch.float16)
        print(f"\n=== {model_id}  (ablate band {list(dirs)} of {n_layers}) ===")

        j = load_judge(model_id, quantize=True)
        sweep = {}
        for alpha in args.alphas:
            long = []
            for r in rows:
                msgs = pointwise_messages(r["text"], r["prior_work"])
                if alpha == 0:
                    y = expected_score(j, msgs)
                else:
                    with contextlib.ExitStack() as st:
                        for L, d in dirs.items():
                            st.enter_context(ablate(j, L, d, alpha=alpha))
                        y = expected_score(j, msgs)
                long.append({**{k: r[k] for k in ("stem", "domain", "cell", "s", "g")},
                             "y": y, "judge": model_id})
            df = pd.DataFrame(long)
            fit = fit_mixed(df, judge=model_id, B=1000)
            hack = hackability_index(df, judge=model_id, B=1000)
            sweep[alpha] = {"beta_S": fit.beta_S, "beta_G": fit.beta_G,
                            "ratio_S_over_G": fit.ratio_S_over_G,
                            "hackability": hack["hackability_index"], "y_std": float(df["y"].std())}
            print(f"  alpha={alpha}: beta_S={fit.beta_S:+.3f} beta_G={fit.beta_G:+.3f} "
                  f"ratio={fit.ratio_S_over_G:.2f} hack={hack['hackability_index']:.2f} y_std={df['y'].std():.2f}")
        out["per_judge"][model_id] = {"band": list(dirs), "sweep": sweep}
        del j.model; gc.collect(); torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
