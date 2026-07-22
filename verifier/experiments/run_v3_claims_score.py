"""Score v3 (claim-extraction then grounded distance). For each stem we have the
extracted atomic claim of the base idea, the significance-attack, and the
topical-padding attack. v3 defeats an attack if, after extraction, the attack's
claim (a) reads like the base claim and (b) scores like it.

Metrics vs prior_work distance:
  sig_gain / pad_gain = dist(attack_claim) - dist(base_claim)   (want ~0)
  claim collapse       = cos(base_claim, attack_claim)          (want ~1)
Compare to the pre-extraction attack strengths (v2 topical-pad raw gain +0.099).

Usage: python experiments/run_v3_claims_score.py <v3_extract_workflow_output.json>
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    claims = {it["stem_id"]: it for it in raw.get("result", raw)["items"]}
    inp = {it["stem_id"]: it for it in json.loads((ROOT / "data" / "v3_inputs.json").read_text())["items"]}
    ids = [s for s in claims if s in inp]

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def enc(texts):
        return m.encode(texts, normalize_embeddings=True)

    def dist(v, vp):
        return float(1.0 - np.dot(v, vp))

    sig_gain, pad_gain, sim_bs, sim_bp = [], [], [], []
    # also the RAW (pre-extraction) topical-pad gain, for contrast
    raw_pad_gain = []
    for sid in ids:
        c, it = claims[sid], inp[sid]
        vp = enc([it["prior_work"]])[0]
        vb, vs, vd = enc([c["base_claim"], c["sig_claim"], c["pad_claim"]])
        sig_gain.append(dist(vs, vp) - dist(vb, vp))
        pad_gain.append(dist(vd, vp) - dist(vb, vp))
        sim_bs.append(float(np.dot(vb, vs)))
        sim_bp.append(float(np.dot(vb, vd)))
        vraw_b, vraw_p = enc([it["base"], it["pad"]])
        raw_pad_gain.append(dist(vraw_p, vp) - dist(vraw_b, vp))

    out = {
        "n": len(ids),
        "sig_gain_after_extract": float(np.mean(sig_gain)),
        "pad_gain_after_extract": float(np.mean(pad_gain)),
        "raw_pad_gain_before_extract": float(np.mean(raw_pad_gain)),
        "claim_collapse_base_sig": float(np.mean(sim_bs)),
        "claim_collapse_base_pad": float(np.mean(sim_bp)),
    }
    (ROOT / "results" / "v3_claims.json").write_text(json.dumps(out, indent=2))
    print(f"v3 claim-extraction, n={len(ids)}:")
    print(f"  claim collapse cos(base, sig) = {out['claim_collapse_base_sig']:.3f}   "
          f"cos(base, pad) = {out['claim_collapse_base_pad']:.3f}   (want ~1)")
    print(f"  topical-pad gain: raw {out['raw_pad_gain_before_extract']:+.4f}  ->  "
          f"after claim-extract {out['pad_gain_after_extract']:+.4f}   (want ~0)")
    print(f"  significance-attack gain after claim-extract {out['sig_gain_after_extract']:+.4f}   (want ~0)")
    ok = abs(out['pad_gain_after_extract']) < 0.3 * abs(out['raw_pad_gain_before_extract'] or 1)
    print(f"  => v3 {'DEFEATS both red-teams (extraction collapses the attacks)' if ok else 'only partially closes the hole'}")
    print("wrote results/v3_claims.json")


if __name__ == "__main__":
    main()
