"""Freeze the S x G dataset: take the construction-workflow output, apply the
judge-independent G-separation gate (local half of K4), write data/dataset_v1.json.

Usage:
  python experiments/build_dataset.py <workflow_output.json> [--out data/dataset_v1.json]

<workflow_output.json> is the build-sxg-dataset task output; result.valid holds
the S/plausibility/leak-validated stems.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from novjudge.dataio import filter_g_separation, save_dataset  # noqa: E402
from novjudge.rubric import RUBRIC_IDS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow_output")
    ap.add_argument("--out", default=str(ROOT / "data" / "dataset_v1.json"))
    ap.add_argument("--stamp", default="", help="ISO timestamp (passed in; scripts can't read the clock)")
    args = ap.parse_args()

    raw = json.loads(Path(args.workflow_output).read_text(encoding="utf-8"))
    result = raw.get("result", raw)
    valid = result.get("valid", [])
    print(f"loaded {len(valid)} S/plausibility/leak-validated stems "
          f"(built {result.get('n_built')} / specs {result.get('n_specs')}; "
          f"drops {result.get('drops')})")

    kept, reports = filter_g_separation(valid)
    n_fail = sum(1 for r in reports if not r.passes)
    print(f"G-separation gate: {len(kept)} kept, {n_fail} dropped (signaled did "
          f"not out-signal plain on MiniLM+lexicon)")

    meta = {
        "name": "sxg_dataset_v1",
        "n_stems": len(kept),
        "domains": sorted({s["domain"] for s in kept}),
        "rubric_ids": RUBRIC_IDS,
        "gate": "S blind-adjudicated + plausibility>=3 & balanced + no content-leak "
                "(workflow) THEN MiniLM+lexicon G-separation (local)",
        "stamp": args.stamp,
        "source_workflow_output": str(Path(args.workflow_output).name),
    }
    save_dataset(kept, args.out, meta)
    print(f"wrote {args.out}  ({len(kept)} stems, {len(kept) * 4} items)")
    if len(kept) < 120:
        print(f"WARNING: {len(kept)} < 120 target — top up specs before E0 to hit the prereg N.")


if __name__ == "__main__":
    main()
