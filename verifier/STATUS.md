# STATUS — verifier / P1

**Signal vs. substance in LLM research judges: a crossed-factor test, a
reward-hackability demonstration, and a steering fix.**
Author: Gunner Levi Howe. Last updated: 2026-07-22.

## One-line

We test whether LLM research judges score *surface novelty-signaling* rather than
*substantive novelty* (correctness held fixed via a crossed 2×2), show the
resulting verifier is reward-hackable, localize the response to a linear
direction, and test a steering fix that recovers calibration — reconciling two
2026 papers that found opposite-signed novelty bias.

## Outcome: OPEN — infra built & engine validated; awaiting gap verdict before freeze

`PLAN.md` is a DRAFT pre-registration. **No experiment has been run**; no judge
has scored any real item; nothing is contaminated. What exists now:

- Full measurement infra written and gated (`src/novjudge/`): frozen `rubric.py`
  (1-9 novelty, hashed freeze ids), judge-independent `signal.py` (lexicon +
  MiniLM difference-of-centroids), `estimate.py` (stem-clustered mixed fit +
  hackability index + cluster bootstrap), `judge_local.py` (4-bit load,
  expected-digit score readout, residual capture + steering hooks). 8/8 tests
  pass (schema, estimator recovery, signal).
- Engine validated on GPU (Phi-3.5-mini 4-bit, peak 3.72 GB of ~6 GB free;
  Qwen2.5-7B / Nemotron-8B expected to fit in 4-bit). Hooks fire.
- **Smoke signal (N=1 dummy pair, NOT a result):** identical attention-gating
  idea scored 2.37/9 plain vs 9.00/9 signaled; steering the plain item along the
  signal direction moved 2.37 -> 8.52. Confirms the instrument detects the effect
  and steering moves it — de-risks E0/E3/E4; proves nothing on its own.

Gates before any real scoring: (1) `verify-p1-gap` workflow verdict — DONE,
**ADJUST** (E clear/lead; A–D cite-and-distinguish; six forced changes folded
into `PLAN.md`; RINoBench corrected to regression-to-mean/both-tails); (2) freeze
(commit) `PLAN.md` — DONE this session; (3) build + validate the S x G dataset
(S/G/correctness pass, K4) — NEXT.

Near-scoop to beat: **Breaking the Mirror** (2509.03647, self-preference
probe+steer) — C/D must use a trained probe, OOD transfer, and dissociate the
novelty-signal direction from self-preference/uncertainty. Full must-cite map in
`lit_notes.md`.

## Provenance

Repo seeded from `compass_artifact_*.md` (report: "The Verification Bottleneck in
Automated AI Research"). Of the report's five candidate projects (P1–P5), the
user selected **P1** (novelty-vs-consensus LLM judge). The initial literature
sweep (see `lit_notes.md`) found two 2026 preprints on the same turf with
contradictory findings; that contradiction motivated the sharpened
signal-vs-substance design in `PLAN.md`, which is an evolution of the report's
original P1 (it replaces the survivorship-prone "historically-vindicated ideas"
framing with a correctness-controlled crossed design).

## Experiment ladder (planned; see PLAN.md)

- **E0 gate** — crossed 2×2 (S×G), mixed-model β_G test. [not run]
- **E1** — identification: β_S, β_G, hackability index, correctness arm, placebo. [not run]
- **E2** — reward-hackability: optimize Y under S-freeze; measure gamed gain. [not run]
- **E3** — mechanism: linear G-direction probe (in-dist + OOD). [not run]
- **E4** — steering fix: ablate/steer G-direction; recover calibration. [not run]
- **E5** — reconciliation of RQ-Bench (over) / RINoBench (under). [not run]

## Kill conditions (pre-registered)

K1 judges track substance (honest negative) · K2 behavioral-only (no direction /
steering null) · K3 reconciliation fails · K4 construction confound (fix before
scoring).

## Next actions

1. User sign-off on the reframing → freeze `PLAN.md` (first commit).
2. Build item-construction pipeline (stems → S×G items → S/G/correctness
   validation) with the generator/adjudicator/judge separation.
3. Implement the frozen rubric + pointwise/pairwise judge harness (local 4-bit +
   API) and the mixed-model estimator, gated by tests before any scoring.

## Deliverables (planned)

`PLAN.md` (prereg) · `lit_notes.md` · this `STATUS.md` · `paper/` (+ machine
`numbers.tex`, `verify_regen.py`) · `results/*.json` (committed) · `laymen.md`.
