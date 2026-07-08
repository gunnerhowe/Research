"""GPU run driver: generate + measure all conditions for one model.

Writes one JSONL row per instance, appending chunk-by-chunk with resume
support (already-present qids are skipped on restart), so long runs survive
interruptions. Analysis is done separately by fit_all.py from the raw JSONL.

Usage:
  python experiments/run_model.py --model qwen7b --n 600 \
      --hints sycophancy,authority,metadata,consistency --seed 0 \
      --out results/raw/qwen7b_e0.jsonl
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from faithsel.data import load_pool, make_specs  # noqa: E402
from faithsel.gen import load_model, process_chunk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--hints", default="sycophancy,authority,metadata,consistency")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--gen-bs", type=int, default=12)
    ap.add_argument("--read-bs", type=int, default=8)
    ap.add_argument("--lens-bs", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=448)
    ap.add_argument("--max-per-dataset", type=int, default=1200)
    ap.add_argument("--no-lens", action="store_true")
    ap.add_argument("--system-thinking", default=None,
                    help="override system prompt (nemotron reasoning toggle)")
    args = ap.parse_args()

    hint_types = args.hints.split(",")
    pool = load_pool(seed=args.seed, max_per_dataset=args.max_per_dataset)
    specs = make_specs(pool, hint_types, args.n, seed=args.seed)

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["qid"])
    todo = [s for s in specs if s.qid not in done]
    print(f"[run] {args.model}: {len(specs)} specs, {len(done)} done, "
          f"{len(todo)} to go", flush=True)
    if not todo:
        return

    if args.system_thinking is not None:
        import faithsel.gen as G
        import faithsel.hints as H
        H.SYSTEM_PROMPT = args.system_thinking
        G.SYSTEM_PROMPT = args.system_thinking

    model, tok = load_model(args.model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    t0 = time.time()
    for i in range(0, len(todo), args.chunk):
        chunk = todo[i:i + args.chunk]
        rows = process_chunk(model, tok, chunk,
                             max_new_tokens=args.max_new_tokens,
                             gen_bs=args.gen_bs, read_bs=args.read_bs,
                             lens_bs=args.lens_bs, do_lens=not args.no_lens)
        with open(args.out, "a", encoding="utf-8") as f:
            for r in rows:
                r["model"] = args.model
                r["seed"] = args.seed
                f.write(json.dumps(r) + "\n")
        n_done = i + len(chunk)
        rate = n_done / (time.time() - t0)
        eta_min = (len(todo) - n_done) / max(rate, 1e-9) / 60
        print(f"[run] {n_done}/{len(todo)} ({rate*60:.1f}/min, "
              f"ETA {eta_min:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
