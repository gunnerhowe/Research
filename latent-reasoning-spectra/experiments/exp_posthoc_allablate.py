"""Post-hoc probe (NOT pre-registered; labeled exploratory in the paper):

E0/E1 established that replacing any SINGLE latent thought is causally near-inert.
Is the content of the recycled thoughts collectively inert too?  Replace ALL six fed
vectors simultaneously with (a) the dataset-mean thought for each slot, (b) zero
vectors, (c) thoughts transplanted from a random donor problem, and measure accuracy
+ margins on the full test set.  If accuracy stays at baseline, the recycled CONTENT
is fully inert and only the segment structure matters (dynamical confirmation of the
curriculum-is-the-mechanism thesis).  Writes results/exp_posthoc_allablate.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrspec.harness import Harness, N_LATENT  # noqa: E402
from lrspec.paths import DATA, RESULTS, ROOT  # noqa: E402
from lrspec.prosqa import load_problems  # noqa: E402

import os as _os
RUNS = Path(_os.environ.get("LRSPEC_RUNS", ROOT / "runs"))


@torch.no_grad()
def run_with_forced_thoughts(h, p, forced):
    """Run the latent phase feeding forced[t] at slot t (t=0..5), then read out."""
    ids, slot_pos, start_pos, end_pos = h.encode(p)
    prefix = torch.tensor([ids[: start_pos + 1]], device=h.device)
    out = h.model(
        input_ids=prefix,
        position_ids=torch.arange(prefix.shape[1], device=h.device).view(1, -1),
        output_hidden_states=True, use_cache=True,
    )
    cache = out.past_key_values
    for t in range(N_LATENT):
        _, _, cache = h._fwd_one(forced[t], slot_pos[t], cache)
    _, logits_end, cache = h._fwd_one(h.emb.weight[h.end_id], end_pos, cache)
    kv = h._kv_of(cache)
    n_ctx = end_pos + 1
    tgt = h.answer_ids(p, p.target)
    neg = h.answer_ids(p, p.neg_target)
    lp_t = h._teacher_forced_logprob(kv, logits_end, n_ctx, tgt)
    lp_n = h._teacher_forced_logprob(kv, logits_end, n_ctx, neg)
    # greedy
    toks = []
    logits = logits_end
    cache2 = h._cache_from(kv, n_ctx)
    for i in range(16):
        nxt = int(torch.argmax(logits).item())
        if nxt == h.eos_id:
            break
        toks.append(nxt)
        _, logits, cache2 = h._fwd_one(h.emb.weight[nxt], n_ctx + i, cache2)
    text = h.tok.decode(toks)
    ans = text.split("#")[-1].replace(",", "").strip()
    return {"margin": lp_t - lp_n, "correct": ans == p.answer}


def main():
    problems = load_problems(DATA / "prosqa_test.json")
    cap = np.load(RUNS / "exp0_capture_M2.npz")
    n = int(cap["n_done"][0])
    mean_thoughts = torch.tensor(
        np.nanmean(cap["hs"][:n, :N_LATENT], axis=0), device="cuda")
    hs_all = cap["hs"][:n]

    h = Harness("M2")
    rng = np.random.default_rng(0)
    conds = {"baseline": [], "all_mean": [], "all_zero": [], "all_donor": []}
    margins = {k: [] for k in conds}
    t0 = time.time()
    for i, p in enumerate(problems[:n]):
        run = h.run_latent(p)
        ro = h.readout(run, p)
        conds["baseline"].append(ro["greedy_text"].split("#")[-1]
                                 .replace(",", "").strip() == p.answer)
        margins["baseline"].append(ro["margin"])

        r = run_with_forced_thoughts(h, p, mean_thoughts)
        conds["all_mean"].append(r["correct"]); margins["all_mean"].append(r["margin"])

        zeros = torch.zeros(N_LATENT, 768, device="cuda")
        r = run_with_forced_thoughts(h, p, zeros)
        conds["all_zero"].append(r["correct"]); margins["all_zero"].append(r["margin"])

        j = int(rng.integers(0, n))
        while j == i:
            j = int(rng.integers(0, n))
        donor = torch.tensor(hs_all[j, :N_LATENT], device="cuda")
        r = run_with_forced_thoughts(h, p, donor)
        conds["all_donor"].append(r["correct"]); margins["all_donor"].append(r["margin"])

        if (i + 1) % 50 == 0:
            print(f"{i+1}/{n} " + " ".join(
                f"{k}={np.mean(v):.3f}" for k, v in conds.items()), flush=True)

    out = {}
    for k in conds:
        out[k] = {"accuracy": float(np.mean(conds[k])),
                  "mean_margin": float(np.mean(margins[k])), "n": len(conds[k])}
    # McNemar-ish: flips relative to baseline
    base = np.array(conds["baseline"])
    for k in ["all_mean", "all_zero", "all_donor"]:
        arr = np.array(conds[k])
        out[k]["n_changed_vs_baseline"] = int((arr != base).sum())
    with open(RESULTS / "exp_posthoc_allablate.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
