"""Fit the selection models from raw JSONL runs and write results JSONs.

Usage:
  python experiments/fit_all.py --raw results/raw/qwen7b_e0.jsonl \
      --tag qwen7b_e0 [--n-boot 1000] [--heckprob] [--quick]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from faithsel.analysis import (augment, e1_balance, fit_report,  # noqa: E402
                               heckprob_report, load_rows, placebo_report,
                               save_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True,
                    help="raw JSONL path(s), comma-separated (merged)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--heckprob", action="store_true",
                    help="also run the observation-only binary pipeline")
    ap.add_argument("--quick", action="store_true",
                    help="skip bootstrap/sensitivity (pilot look)")
    ap.add_argument("--vjudge", default=None,
                    help="path to a qid->V_judge json; use validated LLM-judge "
                         "V instead of the lexical detector")
    args = ap.parse_args()

    import json
    import pandas as pd
    parts = [load_rows(p) for p in args.raw.split(",")]
    raw = pd.concat(parts, ignore_index=True)
    v_override = None
    if args.vjudge:
        v_override = {k: int(v) for k, v in json.load(open(args.vjudge)).items()}
        print(f"[fit] using judge-V from {args.vjudge} "
              f"({len(v_override)} labels, V-rate "
              f"{sum(v_override.values())/max(len(v_override),1):.3f})", flush=True)
    df = augment(raw, v_override=v_override)
    os.makedirs(args.outdir, exist_ok=True)
    nb = 0 if args.quick else args.n_boot

    out = {"tag": args.tag, "raw": args.raw,
           "n_rows": int(len(df)),
           "outcomes": {}}
    for outcome in ("R_TE", "R_NDE", "R_pre"):
        if outcome not in df.columns:
            continue
        print(f"[fit] {args.tag} outcome={outcome}", flush=True)
        out["outcomes"][outcome] = fit_report(
            df, outcome=outcome, n_boot=nb,
            do_sensitivity=not args.quick,
            per_hint=(outcome in ("R_TE", "R_NDE")))
    out["e1_balance"] = e1_balance(df)
    # placebo carries no real cue, so its verbalization is uncontroversial;
    # score it with the lexical detector (judge-V does not cover placebo).
    placebo_df = augment(raw) if v_override is not None else df
    out["placebo"] = placebo_report(placebo_df)
    out["v_source"] = "judge" if v_override is not None else "detector"
    if args.heckprob:
        print(f"[fit] {args.tag} heckprob", flush=True)
        out["heckprob"] = heckprob_report(df, n_boot=0 if args.quick
                                          else min(nb, 300))

    path = os.path.join(args.outdir, f"{args.tag}.json")
    save_json(out, path)
    print(f"[fit] wrote {path}", flush=True)

    # console summary
    for oc, rep in out["outcomes"].items():
        g = rep["gate"]
        print(f"  {oc}: rho_mle={rep['mle']['rho']:+.3f} "
              f"(LR p={rep['rho_lr']['p']:.2g}) "
              f"V-rate={rep['verbalization_rate']:.2f} "
              f"naive_sel={rep['two_step']['estimands']['naive_selected']:+.3f} "
              f"corr_pop={rep['two_step']['estimands']['corrected_pop']:+.3f} "
              f"true_pop={rep['targets']['true_pop']:+.3f} "
              f"beats_naive={g['corrected_beats_naive_selected']} "
              f"beats_hidden={g['corrected_beats_naive_hidden']}", flush=True)


if __name__ == "__main__":
    main()
