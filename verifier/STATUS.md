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

## Outcome: POSITIVE core + a sharp dissociation (detect-but-can't-fix)

Emerging story (honest, strong): LLM research judges score novelty SIGNAL over
SUBSTANCE (E0, all 3 judges, robust to length, hackability 0.82); the signal
response is linearly DETECTABLE and OOD-robust (E3, AUROC ~1.0); BUT it is NOT
removable by activation steering (E4, single- and multi-layer both fail) — the
bias is entangled with novelty computation, not a separable direction. So
detectability != steerability, contra Breaking-the-Mirror. This makes the verifier
warning stronger, not weaker: you cannot cheaply patch it. The lead is E0 + the
detect/can't-fix dissociation.

**BUILD PHASE — v1 verifier WORKS (first brick).** V0: judge tracks substance on
neutral content (margin = E0 beta_S). v1 = neutralize-then-judge (subagent strips
rhetoric, keeps content; validated lexicon 1.50->0.01, same-content sim 0.966 vs
diff 0.682). Re-scored: **beta_G +0.75/1.49/0.84 -> ~0** (hack channel closed) while
**beta_S preserved** (0.40->0.39, 1.14->1.15, 0.47->0.46); hackability 0.71-0.89 ->
0.06-0.22 (below chance = now favors substance). Input-space decontamination
SUCCEEDS where E4 activation-steering FAILED (on our lexicon). `results/v1_neutralized.json`.

**v1 RED-TEAM (honest tempering): v1 is a PARTIAL mitigation.** Held-out attacks
(significance/sophistication framing, known novelty lexicon banned) still fool the
NAIVE judge (naive gain +0.61/+1.24/+0.38 Phi/Qwen/Nemo) — so the vulnerability is
deeper than a lexicon. v1's neutralizer only partially strips it: residual
+0.14/+0.80/-0.17 (blocks ~77/35/100% of the gain; fully-blocked stems 42/18/62%).
Qwen leaks badly (+0.80). `results/v1_redteam.json`. => input-neutralization closes
the easy channel but sophisticated significance-framing survives, unevenly across
judges. NEXT: v2 = framing-invariant BY CONSTRUCTION (decompose idea -> atomic
technical claims, score novelty as claim distance from prior_work), which discards
framing rather than trying to strip it.

**v2 = grounded novelty (MiniLM distance idea<->prior_work).** On framed 2x2 it
INVERTS naive: beta_S +0.79 > beta_G +0.23, hack 0.34; on the v1 red-team
(significance-framing) attack gain -0.04 (immune where v1 leaked). BUT v2 RED-TEAM
(topical padding: same idea recast in distant-field jargon) FOOLS it — padding gain
+0.099 = 2.15x the genuine S=low->S=high distance gap (+0.046). `results/v2_grounded.json`,
`results/v2_redteam.json`. So: naive fooled by all rhetoric; v1 resists known
rhetoric/leaks on novel significance-framing; v2 resists significance-framing/fooled
by topical padding. Each has a DISTINCT hole. NEXT v3 = claim-extraction-then-score
(reduce to atomic technical claim, discarding framing AND cross-field jargon) to
defeat BOTH red-teams; test extracted-claim score(base) ~ score(sig-attack) ~
score(topical-pad).

**v3 = claim-extraction-then-grounded-distance: ROBUST AND DISCRIMINATING (best so
far).** Extraction collapses BOTH attacks onto the base claim: cos(base,sig)=0.991,
cos(base,pad)=0.985; topical-pad gain +0.0999 -> +0.0042, significance gain +0.0047
(both ~0). AND it preserves genuine novelty: high-substance claims stay distinct
(cos(base,high)=0.635 << 0.99 attack-collapse; v3 substance gap +0.030 = 74% of raw
+0.041). So v3 defeats the attacks that beat v1 (significance) and v2 (padding)
while still tracking substance. `results/v3_claims.json`, `results/v3_substance.json`.
Honest limits: substance signal modest (+0.030, crude proxy); NOT yet red-teamed at
the EXTRACTOR level (next: inject pseudo-substance to fool the claim extractor);
50-stem tests, synthetic, MiniLM. Verifier scorecard: naive fails all; v1 known-only;
v2 significance-robust/padding-weak; v3 passes both red-teams + substance.

**LADDER COMPLETE (E0-E5).** Paper drafted + compiled: `paper/main.tex` +
machine-generated `paper/numbers.tex` (80 macros, verify_regen byte-identical) +
`paper/main.pdf` (5pp) + figs (E0/E4/E5); `laymen.md`. Every number sourced from
committed `results/*.json`. Extensions: E3 dissociation (novelty-signal vs
self-preference/uncertainty directions), external E5 (re-analyze RQ-Bench/RINoBench
corpora), frontier judges, natural (non-synthetic) ideas.

## History

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

## Experiment ladder (see PLAN.md)

- **E0 gate — PASSED (signal beats substance, all 3 judges).** N=154 stems /
  616 items. beta_G > beta_S for every judge, all CIs exclude 0. Pooled
  beta_S=+0.67, beta_G=+1.03 (ratio 0.65); hackability index 0.82 (pooled) —
  ~82% of matched stems, a low-substance signaled idea outscores a high-substance
  plain idea. Per judge: Phi beta_G +0.75 / hack 0.89; Qwen +1.49 / 0.85;
  Nemotron +0.84 / 0.71. K1 does NOT fire. `results/e0_results.json`.
- **E1 length control — PASSED (not a verbosity artifact).** Signaled cells only
  ~6% longer (91.7 vs 97.0 words); within-stem beta_G does NOT shrink under a
  length covariate (shrink -3% to -6%; beta_len ~0/negative); CIs exclude 0.
  `results/e1_length_control.json`. Remaining E1: pairwise robustness, correctness
  arm (discriminant validity), placebo, rationalization.
- **E2** — reward-hackability: optimize Y under S-freeze; measure gamed gain. [next]
- **E3 — DONE.** Linear, OOD-robust novelty-signal direction, all 3 judges:
  in-dist AUROC 0.999-1.000, OOD 0.997-1.000. `results/e3_probe.json`. Caveat:
  near-perfect AUROC partly expected (signal is surface-lexical) — supporting, not
  headline. Dissociation from self-preference/uncertainty directions still TODO.
- **E4 — DONE: FIX FAILS (K2 on the fix).** Neither single-layer (e4_steering) nor
  multi-layer band ablation (e4_multilayer) of the diff-of-means G-direction gives
  selective recalibration (beta_G down, beta_S preserved). Instead: Phi inflates or
  loses substance-tracking, Qwen negligible, Nemotron collapses BOTH effects. The
  bias is linearly DETECTABLE but not steerable away = detectability != removability;
  contrasts Breaking-the-Mirror (self-preference WAS steerable). A sharper warning
  than a working patch.
- **E2 — DONE: reward-hackable.** Trivial rhetoric-injection (no substance) farms
  the score +0.99/+2.10/+1.92 (Phi/Qwen/Nemo); low-substance ideas farmed past the
  high-substance mean rise 47/7/20% -> 100/83/100%. `results/e2_rewardhack.json`.
- **E1 pairwise — DONE** (above): survives comparative readout 2/3 judges.
- **E5** — reconciliation of RQ-Bench (over) / RINoBench (both tails). [pending; internal
  from E0 data + optional external anchor]

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
