"""v3 substance-tracking test. We have, per stem: the extracted claim of the
INCREMENTAL idea (base_claim, from v3_claims_extracted.json) and of the
GENUINELY-NOVEL idea (high_claim, from this extraction). v3 is a real verifier
only if genuine novelty stays DISTINCT after extraction:
  v3 substance gap = mean[ dist(high_claim) - dist(base_claim) ]  (want > 0, ~ raw gap)
  cos(base_claim, high_claim)  << the attack-collapse cos (0.99)   (distinct ideas)
Contrast: the attacks collapsed to cos ~0.99 and gain ~0; genuine novelty must NOT.

Usage: python experiments/run_v3_substance_score.py <high_claim_workflow_output.json>
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    high = {it["stem_id"]: it["high_claim"] for it in raw.get("result", raw)["items"]}
    base = {it["stem_id"]: it["base_claim"] for it in json.loads((ROOT / "data" / "v3_claims_extracted.json").read_text())["items"]}
    stems = {s["stem_id"]: s for s in json.loads((ROOT / "data" / "dataset_v1.json").read_text())["stems"]}
    ids = [s for s in high if s in base and s in stems]

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def enc(t):
        return m.encode(t, normalize_embeddings=True)

    def dist(v, vp):
        return float(1.0 - np.dot(v, vp))

    v3_gap, raw_gap, sim_bh = [], [], []
    for sid in ids:
        s = stems[sid]
        vp = enc([s["prior_work"]])[0]
        vb, vh = enc([base[sid], high[sid]])
        v3_gap.append(dist(vh, vp) - dist(vb, vp))
        sim_bh.append(float(np.dot(vb, vh)))
        vlo, vhi = enc([s["s_low_content"], s["s_high_content"]])
        raw_gap.append(dist(vhi, vp) - dist(vlo, vp))

    v3c = json.loads((ROOT / "results" / "v3_claims.json").read_text())
    out = {"n": len(ids),
           "v3_substance_gap": float(np.mean(v3_gap)),
           "raw_content_gap": float(np.mean(raw_gap)),
           "cos_base_high_claim": float(np.mean(sim_bh)),
           "cos_attack_collapse_ref": (v3c["claim_collapse_base_sig"] + v3c["claim_collapse_base_pad"]) / 2,
           "pad_gain_after_extract_ref": v3c["pad_gain_after_extract"]}
    (ROOT / "results" / "v3_substance.json").write_text(json.dumps(out, indent=2))
    print(f"v3 substance test, n={len(ids)}:")
    print(f"  v3 substance gap (high-claim vs base-claim distance) = {out['v3_substance_gap']:+.4f}  "
          f"(raw content gap {out['raw_content_gap']:+.4f})")
    print(f"  cos(base_claim, high_claim) = {out['cos_base_high_claim']:.3f}  "
          f"vs attack-collapse cos {out['cos_attack_collapse_ref']:.3f}")
    keep = out["v3_substance_gap"] > 0.5 * out["raw_content_gap"] and out["cos_base_high_claim"] < 0.95
    print(f"  attacks collapse (cos~0.99, gain~0) but genuine novelty stays distinct: "
          f"{'YES — v3 is robust AND discriminating' if keep else 'NO — v3 over-collapses (useless)'}")
    print("wrote results/v3_substance.json")


if __name__ == "__main__":
    main()
