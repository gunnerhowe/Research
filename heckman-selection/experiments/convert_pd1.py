"""One-time converter: PD1 (matched, phase 0+1) -> data/pd1_cache.npz

Per selected task: matrix of validation ACCURACY (1 - valid/error_rate),
shape (n_trials, T_modal); trials with missing/short/non-finite curves are
dropped (counts printed). Eval index is the time axis (evals are evenly
spaced in steps within a task).

Run: python experiments/convert_pd1.py
"""

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = [ROOT / "data" / "pd1" / "pd1_matched_phase0_results.jsonl.gz",
       ROOT / "data" / "pd1" / "pd1_matched_phase1_results.jsonl.gz"]
OUT = ROOT / "data" / "pd1_cache.npz"

TASKS = ["imagenet__resnet", "lm1b__transformer", "uniref50__transformer",
         "svhn_no_extra__wide_resnet"]


def main():
    curves = defaultdict(list)
    for src in SRC:
        with gzip.open(src, "rt") as f:
            for line in f:
                d = json.loads(line)
                key = d["dataset"] + "__" + d["model"]
                if key not in TASKS:
                    continue
                ve = d.get("valid/error_rate")
                if isinstance(ve, list) and len(ve) > 3:
                    curves[key].append(np.asarray(ve, dtype=np.float32))

    store = {}
    for key in TASKS:
        cs = curves[key]
        lens = Counter(len(c) for c in cs)
        T = lens.most_common(1)[0][0]
        kept = [1.0 - c[:T] for c in cs
                if len(c) >= T and np.all(np.isfinite(c[:T]))]
        mat = np.stack(kept)
        # clip pathological values (diverged runs report error > 1)
        mat = np.clip(mat, 0.0, 1.0)
        store[key] = mat.astype(np.float32)
        print(f"{key:35s} kept {len(kept):4d}/{len(cs):4d} T={T}")
    np.savez_compressed(OUT, **store)
    print("saved", OUT)


if __name__ == "__main__":
    main()
