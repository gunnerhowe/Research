"""Frozen judge prompts. Committed and hashed so the freeze is auditable.

The rubric asks ONLY about novelty (not overall quality), because the design
holds correctness fixed and must isolate the novelty-scoring channel. A single
1-9 scale is used deliberately: it lets the judge harness read a continuous
expected score from the next-token distribution over the nine single digits
(see judge_local.expected_score), which is lower-variance and parse-failure-free
compared with decoding free text.
"""

from __future__ import annotations

import hashlib

# --- POINTWISE novelty rubric (primary) -------------------------------------

POINTWISE_SYSTEM = (
    "You are an expert reviewer assessing the NOVELTY of a research idea — how "
    "much it departs conceptually from the prior work it builds on. Judge novelty "
    "ONLY. Do not reward good writing, confident phrasing, feasibility, or "
    "correctness; an idea can be well-written yet unoriginal, or plainly worded "
    "yet genuinely new. Rate on a 1-9 scale where 1 = a restatement or trivial "
    "increment of the prior work, and 9 = a genuinely new conceptual contribution. "
    "Answer with a SINGLE DIGIT from 1 to 9 and nothing else."
)

POINTWISE_USER = (
    "PRIOR WORK (context this idea builds on):\n{prior_work}\n\n"
    "RESEARCH IDEA:\n{idea}\n\n"
    "How novel is the idea relative to the prior work? Answer with one digit 1-9."
)

# --- PAIRWISE novelty rubric (robustness) -----------------------------------

PAIRWISE_SYSTEM = (
    "You are an expert reviewer comparing the NOVELTY of two research ideas that "
    "build on the same prior work. Judge novelty ONLY — conceptual departure from "
    "the prior work — not writing quality, confidence, feasibility, or "
    "correctness. Answer with a single letter: A if idea A is more novel, B if "
    "idea B is more novel."
)

PAIRWISE_USER = (
    "PRIOR WORK:\n{prior_work}\n\n"
    "IDEA A:\n{idea_a}\n\nIDEA B:\n{idea_b}\n\n"
    "Which idea is more novel relative to the prior work? Answer A or B."
)

SCALE_MIN, SCALE_MAX = 1, 9


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


# Freeze ids: any edit to a prompt changes its id; verify_regen / tests assert
# the id in committed results matches, so a silent prompt drift is caught.
RUBRIC_IDS = {
    "pointwise_v1": _hash(POINTWISE_SYSTEM, POINTWISE_USER),
    "pairwise_v1": _hash(PAIRWISE_SYSTEM, PAIRWISE_USER),
}


def pointwise_messages(idea: str, prior_work: str) -> list[dict]:
    return [
        {"role": "system", "content": POINTWISE_SYSTEM},
        {"role": "user", "content": POINTWISE_USER.format(idea=idea, prior_work=prior_work)},
    ]


def pairwise_messages(idea_a: str, idea_b: str, prior_work: str) -> list[dict]:
    return [
        {"role": "system", "content": PAIRWISE_SYSTEM},
        {"role": "user", "content": PAIRWISE_USER.format(
            idea_a=idea_a, idea_b=idea_b, prior_work=prior_work)},
    ]
