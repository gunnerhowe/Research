"""Judge the verbalization indicator V for EVERY hinted instance of a run,
using the (fixed) LLM judge as the validated V measure. Saves qid -> V_judge
so fits can rest on judge-V rather than the lexical detector.

Usage:
  python experiments/judge_all.py --raw results/raw/nemotron8b_e0.jsonl \
      --out results/vjudge_nemotron8b.json --workers 12
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from faithsel.analysis import load_rows  # noqa: E402
from faithsel.hints import HINT_TEMPLATES, parse_answer, split_cot  # noqa: E402
from judge_v import JUDGE_PROMPT, ask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--include-placebo", action="store_true")
    args = ap.parse_args()

    df = load_rows(args.raw)
    if not args.include_placebo:
        df = df[df["hint_type"] != "placebo"]
    # only judge parseable hinted traces (V is defined on the reasoning)
    df = df[[parse_answer(t) is not None for t in df["gen_hinted"]]].copy()
    df["cot"] = [split_cot(t).strip() for t in df["gen_hinted"]]

    existing = {}
    if os.path.exists(args.out):
        existing = json.load(open(args.out))
    todo = df[~df["qid"].isin(existing)]
    print(f"[judge_all] {len(df)} hinted, {len(existing)} done, {len(todo)} to go",
          flush=True)

    def judge_row(row):
        hint_text = HINT_TEMPLATES[row.hint_type].format(h=row.hint_letter)
        v = ask(JUDGE_PROMPT.format(hint_text=hint_text, cot=row.cot[:4000]),
                model=args.model)
        return row.qid, (1 if v == "YES" else 0 if v == "NO" else None)

    out = dict(existing)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for qid, v in ex.map(judge_row,
                             [r for _, r in todo.iterrows()]):
            done += 1
            if v is not None:
                out[qid] = v
            if done % 40 == 0:
                rate = done / (time.time() - t0) * 60
                json.dump(out, open(args.out, "w"))
                print(f"[judge_all] {done}/{len(todo)} ({rate:.0f}/min, "
                      f"V-rate so far {sum(out.values())/max(len(out),1):.3f})",
                      flush=True)
    json.dump(out, open(args.out, "w"))
    vr = sum(out.values()) / max(len(out), 1)
    print(f"[judge_all] DONE {len(out)} judged, V-rate={vr:.3f} -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
