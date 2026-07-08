"""E2 observation-only arm: Claude via the claude CLI (print mode).

Generates hinted + matched unhinted CoT answers for N instances; records
TEXT ONLY (the observation-only regime the correction is for). Analysis then
runs the heckprob pipeline via fit_all.py --heckprob.

Usage:
  python experiments/exp2_claude.py --n 400 --out results/raw/claude_e2.jsonl \
      [--model claude-haiku-4-5] [--workers 6]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from faithsel.data import load_pool, make_specs  # noqa: E402
from faithsel.hints import SYSTEM_PROMPT, build_user_prompt  # noqa: E402


def ask(prompt: str, model: str, timeout: int = 180) -> str:
    cmd = ["claude", "-p", "--model", model,
           "--system-prompt", SYSTEM_PROMPT,
           "--setting-sources", ""]
    for attempt in range(3):
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True,
                               text=True, timeout=timeout, encoding="utf-8",
                               errors="replace", shell=True)
            out = (r.stdout or "").strip()
            if out:
                return out
        except subprocess.TimeoutExpired:
            pass
        time.sleep(2.0 * (attempt + 1))
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--hints", default="sycophancy,authority,metadata,consistency")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    pool = load_pool(seed=args.seed, max_per_dataset=1200)
    specs = make_specs(pool, args.hints.split(","), args.n, seed=args.seed)

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["qid"])
    todo = [s for s in specs if s.qid not in done]
    print(f"[claude-e2] {len(specs)} specs, {len(done)} done, {len(todo)} to go",
          flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def run_one(s):
        gh = ask(build_user_prompt(s, hinted=True), args.model)
        gu = ask(build_user_prompt(s, hinted=False), args.model)
        if not gh or not gu:
            return None
        return {"qid": s.qid, "dataset": s.dataset, "hint_type": s.hint_type,
                "hint_letter": s.hint_letter, "correct": s.correct, "z": s.z,
                "question_len": len(s.question),
                "gen_hinted": gh, "gen_unhinted": gu,
                "model": f"claude:{args.model}", "seed": args.seed}

    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for row in ex.map(run_one, todo):
            n_done += 1
            if row is not None:
                with open(args.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            if n_done % 20 == 0:
                rate = n_done / (time.time() - t0) * 60
                print(f"[claude-e2] {n_done}/{len(todo)} ({rate:.1f}/min)",
                      flush=True)


if __name__ == "__main__":
    main()
