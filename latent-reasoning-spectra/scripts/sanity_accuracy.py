"""Sanity gate: reproduce the published ProsQA test accuracies of the checkpoints.

Published (bmarti44/coconut-curriculum-checkpoints): M1 83.0, M2 97.0, M3 96.6, M4 94.8.
Answer extraction mirrors facebookresearch/coconut run.py:
  answer_output = text.split("#")[-1].replace(",", "").strip()
"""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrspec.harness import Harness, MAX_NEW_TOKENS  # noqa: E402
from lrspec.paths import DATA, RESULTS  # noqa: E402
from lrspec.prosqa import load_problems  # noqa: E402


def eval_latent_model(key: str, problems) -> dict:
    h = Harness(key)
    n_cor = 0
    margins = []
    t0 = time.time()
    for p in problems:
        run = h.run_latent(p)
        ro = h.readout(run, p)
        ans = ro["greedy_text"].split("#")[-1].replace(",", "").strip()
        n_cor += ans == p.answer
        margins.append(ro["margin"])
    del h
    torch.cuda.empty_cache()
    return {
        "accuracy": n_cor / len(problems),
        "n": len(problems),
        "mean_margin": sum(margins) / len(margins),
        "runtime_s": round(time.time() - t0, 1),
    }


@torch.no_grad()
def eval_cot_model(problems) -> dict:
    h = Harness("M1")
    n_cor = 0
    t0 = time.time()
    for p in problems:
        ids = h.tok.encode(p.question + "\n", add_special_tokens=True)
        x = torch.tensor([ids], device=h.device)
        out = h.model.generate(
            x, max_new_tokens=64, do_sample=False,
            pad_token_id=h.eos_id, eos_token_id=h.eos_id,
        )
        text = h.tok.decode(out[0][len(ids):], skip_special_tokens=True)
        ans = text.split("#")[-1].replace(",", "").strip()
        n_cor += ans == p.answer
    del h
    torch.cuda.empty_cache()
    return {"accuracy": n_cor / len(problems), "n": len(problems),
            "runtime_s": round(time.time() - t0, 1)}


def main():
    """One model per process (VRAM discipline): merge into the JSON incrementally.

    Usage: python scripts/sanity_accuracy.py M2   (or M1/M3/M4, or 'all' to spawn
    one subprocess per model)."""
    import subprocess

    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "sanity_accuracy.json"

    if which == "all":
        for key in ["M2", "M3", "M4", "M1"]:
            r = subprocess.run([sys.executable, "-X", "utf8", "-W", "ignore",
                                __file__, key])
            if r.returncode != 0:
                sys.exit(r.returncode)
        return

    problems = load_problems(DATA / "prosqa_test.json")
    out = json.load(open(path)) if path.exists() else {}
    out.setdefault("published", {"M1": 0.830, "M2": 0.970, "M3": 0.966, "M4": 0.948})
    if which == "M1":
        out["M1"] = eval_cot_model(problems)
    else:
        out[which] = eval_latent_model(which, problems)
    print(which, out[which], flush=True)
    json.dump(out, open(path, "w"), indent=2)


if __name__ == "__main__":
    main()
