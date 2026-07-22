"""Score the v2 red-team: does TOPICAL PADDING (cross-field jargon that inflates
distance-from-prior-work without adding novelty) fool the grounded verifier v2?
v2 score = 1 - cos(idea, prior_work). Reports the padding gain on the v2 score
(base -> padded). Large gain = v2's hole; small = v2 robust even to this.

Usage: python experiments/run_v2_redteam_score.py <v2_redteam_workflow_output.json>
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from novjudge.dataio import load_dataset  # noqa: E402


def main():
    raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    items = raw.get("result", raw)["items"]
    pw = {s["stem_id"]: s["prior_work"] for s in load_dataset(str(ROOT / "data" / "dataset_v1.json"))["stems"]}
    items = [it for it in items if it["stem_id"] in pw]

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def dist(text, prior):
        v = m.encode([text, prior], normalize_embeddings=True)
        return float(1.0 - np.dot(v[0], v[1]))

    gains = []
    for it in items:
        p = pw[it["stem_id"]]
        gains.append(dist(it["padded"], p) - dist(it["base"], p))
    gains = np.array(gains)
    # scale-reference: the full S=low->S=high genuine distance gap (a "real novelty" unit)
    ds = load_dataset(str(ROOT / "data" / "dataset_v1.json"))["stems"]
    real_gap = np.mean([dist(s["s_high_content"], s["prior_work"]) - dist(s["s_low_content"], s["prior_work"]) for s in ds])

    out = {"n": len(items), "v2_padding_gain_mean": float(gains.mean()),
           "v2_padding_gain_std": float(gains.std()),
           "real_S_distance_gap": float(real_gap),
           "padding_gain_in_real_novelty_units": float(gains.mean() / real_gap) if real_gap else None}
    (ROOT / "results" / "v2_redteam.json").write_text(json.dumps(out, indent=2))
    print(f"v2 red-team (topical padding), n={len(items)}:")
    print(f"  padding gain on v2 grounded distance: {gains.mean():+.4f} (std {gains.std():.4f})")
    print(f"  real S=low->S=high distance gap (genuine novelty unit): {real_gap:+.4f}")
    print(f"  => padding buys {gains.mean()/real_gap:.2f} 'real-novelty units' of v2 score "
          f"({'v2 IS foolable by topical padding' if gains.mean()/real_gap > 0.5 else 'v2 largely holds'})")
    print("wrote results/v2_redteam.json")


if __name__ == "__main__":
    main()
