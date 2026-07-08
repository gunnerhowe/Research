"""Dataset pooling: MMLU, ARC-Easy, ARC-Challenge, OpenbookQA.

All items are normalized to 4-option MCQ with a known correct letter and a
deterministic per-item id. Sampling is seeded and stratified by dataset.
"""

from __future__ import annotations

import hashlib
import random

from datasets import load_dataset

from .hints import LETTERS, PromptSpec, HINT_TEMPLATES

MMLU_SUBJECTS = [
    "high_school_chemistry", "high_school_physics", "college_biology",
    "high_school_world_history", "logical_fallacies", "econometrics",
    "high_school_psychology", "professional_medicine",
]


def _norm_arc(row, ds_name):
    ch = row["choices"]
    if len(ch["text"]) != 4:
        return None
    labels = ch["label"]
    if row["answerKey"] not in labels:
        return None
    idx = labels.index(row["answerKey"])
    return {"question": row["question"], "options": list(ch["text"]),
            "correct": LETTERS[idx], "dataset": ds_name,
            "qid": f"{ds_name}:{row['id']}"}


def load_pool(seed: int = 0, max_per_dataset: int = 1200) -> list[dict]:
    """Deterministic pooled item list across the four datasets."""
    items = []

    for subj in MMLU_SUBJECTS:
        ds = load_dataset("cais/mmlu", subj, split="test")
        for i, row in enumerate(ds):
            if len(row["choices"]) != 4:
                continue
            items.append({"question": row["question"],
                          "options": list(row["choices"]),
                          "correct": LETTERS[int(row["answer"])],
                          "dataset": "mmlu",
                          "qid": f"mmlu:{subj}:{i}"})

    for cfg, name in [("ARC-Easy", "arc_easy"), ("ARC-Challenge", "arc_chal")]:
        ds = load_dataset("allenai/ai2_arc", cfg, split="test")
        for row in ds:
            it = _norm_arc(row, name)
            if it:
                items.append(it)

    ds = load_dataset("allenai/openbookqa", "main", split="test")
    for row in ds:
        ch = row["choices"]
        if len(ch["text"]) != 4 or row["answerKey"] not in ch["label"]:
            continue
        idx = ch["label"].index(row["answerKey"])
        items.append({"question": row["question_stem"],
                      "options": list(ch["text"]),
                      "correct": LETTERS[idx], "dataset": "obqa",
                      "qid": f"obqa:{row['id']}"})

    # deterministic cap per dataset
    rng = random.Random(seed)
    by_ds: dict[str, list] = {}
    for it in items:
        by_ds.setdefault(it["dataset"], []).append(it)
    pool = []
    for name in sorted(by_ds):
        rows = sorted(by_ds[name], key=lambda r: r["qid"])
        rng.shuffle(rows)
        pool.extend(rows[:max_per_dataset])
    return pool


def make_specs(pool: list[dict], hint_types: list[str], n_total: int,
               seed: int = 0) -> list[PromptSpec]:
    """Assign items -> (hint type, hinted letter, instrument arm).

    Hinted letter is a uniformly random WRONG option (standard Turpin
    protocol), so reliance == being steered off the model's own evidence.
    The instrument arm Z is an independent fair coin per instance.
    Assignment is a pure function of (qid, seed) so reruns are reproducible.
    """
    rng = random.Random(seed + 1)
    items = sorted(pool, key=lambda r: r["qid"])
    rng.shuffle(items)
    items = items[:n_total]
    specs = []
    for k, it in enumerate(items):
        # per-item deterministic sub-rng: robust to changes in pool order
        h = hashlib.sha256(f"{seed}:{it['qid']}".encode()).digest()
        sub = random.Random(h)
        hint_type = hint_types[k % len(hint_types)]
        # A random WRONG option (standard Turpin protocol). For the placebo
        # this is a hidden pseudo-target: it is never shown in the prompt but
        # anchors the (expected-null) reliance measurement.
        wrong = [L for L in LETTERS if L != it["correct"]]
        hint_letter = sub.choice(wrong)
        z = sub.randint(0, 1)
        assert hint_type in HINT_TEMPLATES
        specs.append(PromptSpec(
            qid=it["qid"], dataset=it["dataset"], question=it["question"],
            options=it["options"], correct=it["correct"],
            hint_type=hint_type, hint_letter=hint_letter, z=z))
    return specs
