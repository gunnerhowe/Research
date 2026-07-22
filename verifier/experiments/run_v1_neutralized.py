"""v1 verifier = neutralize-then-judge. Given the neutralized cell texts (from the
v1-neutralize workflow), (a) validate the neutralizer stripped framing but kept
content, then (b) re-score with each judge and re-fit S/G effects. If v1 works:
beta_G collapses toward 0 (rhetoric gone -> hack channel closed) while beta_S is
preserved (substance still tracked). Compare to the E0 baseline.

Usage: python experiments/run_v1_neutralized.py <neutralize_workflow_output.json>
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from novjudge.dataio import load_dataset, CELL_SG, CELLS  # noqa: E402
from novjudge.rubric import pointwise_messages  # noqa: E402
from novjudge.signal import lexicon_count, SignalEmbedder  # noqa: E402
from novjudge.estimate import fit_mixed, hackability_index  # noqa: E402

JUDGES = ["microsoft/Phi-3.5-mini-instruct", "Qwen/Qwen2.5-7B-Instruct",
          "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"]


def main():
    raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    neut = {it["stem_id"]: it for it in raw.get("result", raw)["items"]}
    stems = {s["stem_id"]: s for s in load_dataset(str(ROOT / "data" / "dataset_v1.json"))["stems"]}
    ids = [sid for sid in stems if sid in neut]
    print(f"neutralized {len(ids)}/{len(stems)} stems")

    # --- validate the neutralizer: framing collapsed, content preserved ---
    emb = SignalEmbedder()
    emb._ensure()
    lex_before, lex_after = [], []
    sim_same, sim_diff = [], []
    for sid in ids:
        o, nz = stems[sid], neut[sid]
        for c in CELLS:
            lex_before.append(lexicon_count(o[c]))
            lex_after.append(lexicon_count(nz[c]))
        vecs = emb._model.encode([nz["LL"], nz["LH"], nz["HL"], nz["HH"]], normalize_embeddings=True)
        sim_same.append(float(np.dot(vecs[0], vecs[1])))   # LL vs LH (same content)
        sim_same.append(float(np.dot(vecs[2], vecs[3])))   # HL vs HH (same content)
        sim_diff.append(float(np.dot(vecs[0], vecs[2])))   # LL vs HL (diff content)
    print(f"validation: signal-lexicon words/cell {np.mean(lex_before):.2f} -> {np.mean(lex_after):.2f}; "
          f"same-content sim {np.mean(sim_same):.3f}  vs  diff-content sim {np.mean(sim_diff):.3f}")

    rows = []
    for sid in ids:
        for c in CELLS:
            s, g = CELL_SG[c]
            rows.append({"stem": sid, "domain": stems[sid]["domain"], "cell": c,
                         "s": s, "g": g, "text": neut[sid][c],
                         "prior_work": stems[sid]["prior_work"]})

    import torch
    from novjudge.judge_local import load_judge, expected_score

    long = []
    for jid in JUDGES:
        j = load_judge(jid, quantize=True)
        for r in rows:
            long.append({**{k: r[k] for k in ("stem", "domain", "cell", "s", "g")},
                         "y": expected_score(j, pointwise_messages(r["text"], r["prior_work"])),
                         "judge": jid})
        del j.model; gc.collect(); torch.cuda.empty_cache()
    df = pd.DataFrame(long)

    e0 = json.load(open(ROOT / "results" / "e0_results.json"))
    out = {"meta": {"n_stems": len(ids),
                    "lex_before": float(np.mean(lex_before)), "lex_after": float(np.mean(lex_after)),
                    "sim_same_content": float(np.mean(sim_same)), "sim_diff_content": float(np.mean(sim_diff))},
           "per_judge": {}}
    print("\n=== v1 (neutralize-then-judge) vs E0 baseline ===")
    for jid in JUDGES:
        f = fit_mixed(df, judge=jid)
        h = hackability_index(df, judge=jid)
        b = e0["per_judge"][jid]; bh = e0["hackability"][jid]["hackability_index"]
        out["per_judge"][jid] = {"beta_S": f.beta_S, "beta_G": f.beta_G,
                                 "hackability": h["hackability_index"],
                                 "e0_beta_S": b["beta_S"], "e0_beta_G": b["beta_G"], "e0_hack": bh}
        print(f"{jid.split('/')[-1]}: beta_G {b['beta_G']:+.2f} -> {f.beta_G:+.2f}   "
              f"beta_S {b['beta_S']:+.2f} -> {f.beta_S:+.2f}   hack {bh:.2f} -> {h['hackability_index']:.2f}")
    (ROOT / "results" / "v1_neutralized.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/v1_neutralized.json  (beta_G->0 = hack channel closed; beta_S kept = substance retained)")


if __name__ == "__main__":
    main()
