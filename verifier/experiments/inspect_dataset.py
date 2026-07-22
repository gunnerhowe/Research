"""Construct-validity QA for the frozen S x G dataset (no judges). Confirms the
G manipulation landed on the KEPT set: signaled cells out-signal plain cells on
the judge-independent measures, and prints per-cell signal/lexicon means."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from novjudge.dataio import load_dataset, CELLS  # noqa: E402
from novjudge.signal import lexicon_count, SignalEmbedder  # noqa: E402


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "dataset_v1.json")
    ds = load_dataset(path)
    stems = ds["stems"]
    print(f"dataset: {len(stems)} stems, {len(stems) * 4} items  (meta n={ds['meta'].get('n_stems')})")
    dom = {}
    for s in stems:
        dom[s["domain"]] = dom.get(s["domain"], 0) + 1
    print("domains:", dom)

    emb = SignalEmbedder()
    lex = {c: [] for c in CELLS}
    sig = {c: [] for c in CELLS}
    for s in stems:
        scores = dict(zip(CELLS, emb.score_many([s[c] for c in CELLS])))
        for c in CELLS:
            lex[c].append(lexicon_count(s[c]))
            sig[c].append(scores[c])
    print("\nper-cell mean lexicon count (signal words):")
    for c in CELLS:
        print(f"  {c}: {np.mean(lex[c]):.2f}")
    print("per-cell mean MiniLM signal score:")
    for c in CELLS:
        print(f"  {c}: {np.mean(sig[c]):+.3f}")

    d_low = np.array(sig["LH"]) - np.array(sig["LL"])
    d_high = np.array(sig["HH"]) - np.array(sig["HL"])
    print(f"\nG-separation (signaled - plain), should be > 0 on all kept stems:")
    print(f"  S=low  (LH-LL): mean {d_low.mean():+.3f}, min {d_low.min():+.3f}, frac>0 {np.mean(d_low > 0):.2f}")
    print(f"  S=high (HH-HL): mean {d_high.mean():+.3f}, min {d_high.min():+.3f}, frac>0 {np.mean(d_high > 0):.2f}")

    ex = stems[0]
    print(f"\n--- example {ex['stem_id']} ({ex['domain']} / {ex.get('subfield','')}) ---")
    print("PRIOR WORK:", ex["prior_work"][:300])
    for c in CELLS:
        print(f"\n[{c}] {ex[c]}")


if __name__ == "__main__":
    main()
