"""Score the v3 EXTRACTOR red-team: does fabricated substance (a hollow but
concrete-sounding mechanism bolted onto an incremental idea) survive claim
extraction and inflate v3's grounded score? Unlike framing attacks, fabricated
CONTENT changes the extracted claim — and no framing-invariant verifier can tell
invented substance from real substance without an oracle.

  fabrication gain = dist(fabricated_claim) - dist(base_claim)   (want ~0; expect > 0)
Compare to the GENUINE substance gap (+0.030): if fabrication gain ~ genuine gap,
v3 cannot distinguish fabricated from real novelty (the irreducible no-oracle limit).

Usage: python experiments/run_v3_extractor_redteam_score.py <output.json>
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    items = raw.get("result", raw)["items"]
    base = {it["stem_id"]: it["base_claim"] for it in json.loads((ROOT / "data" / "v3_claims_extracted.json").read_text())["items"]}
    pw = {s["stem_id"]: s["prior_work"] for s in json.loads((ROOT / "data" / "dataset_v1.json").read_text())["stems"]}
    items = [it for it in items if it["stem_id"] in base and it["stem_id"] in pw]

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def enc(t):
        return m.encode(t, normalize_embeddings=True)

    def dist(v, vp):
        return float(1.0 - np.dot(v, vp))

    gains = []
    for it in items:
        vp = enc([pw[it["stem_id"]]])[0]
        vb, vf = enc([base[it["stem_id"]], it["fabricated_claim"]])
        gains.append(dist(vf, vp) - dist(vb, vp))
    gains = np.array(gains)
    genuine = json.loads((ROOT / "results" / "v3_substance.json").read_text())["v3_substance_gap"]

    out = {"n": len(items), "fabrication_gain": float(gains.mean()),
           "genuine_substance_gap": genuine,
           "fabrication_in_genuine_units": float(gains.mean() / genuine) if genuine else None}
    (ROOT / "results" / "v3_extractor_redteam.json").write_text(json.dumps(out, indent=2))
    print(f"v3 EXTRACTOR red-team (fabricated substance), n={len(items)}:")
    print(f"  fabrication gain on v3 score = {gains.mean():+.4f}  "
          f"(genuine substance gap {genuine:+.4f})")
    print(f"  => fabrication buys {gains.mean()/genuine:.2f}x a genuine-novelty unit of v3 score")
    print(f"  {'v3 IS fooled by fabricated substance = the no-oracle limit (frame-invariance cannot verify content)' if gains.mean() > 0.5*genuine else 'v3 resists fabrication'}")
    print("wrote results/v3_extractor_redteam.json")


if __name__ == "__main__":
    main()
