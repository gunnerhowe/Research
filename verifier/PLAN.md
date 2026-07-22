# PLAN.md — Pre-registration (freeze candidate v1)

**Project:** Signal vs. substance in LLM research judges: a correctness-controlled
crossed-factor test, a novelty-reward-hackability demonstration, a probed &
dissociated *novelty-signal direction*, and a steering-decontamination fix — with
a unifying surface-signal account that reconciles the over- and under-rating of
LLM novelty judges.
**Author:** Gunner Levi Howe. **Committed before any experimental results.**
**Provenance of adjustments:** the design below folds in the six forced changes
from the adversarial gap review (`lit_notes.md`, workflow verify-p1-gap,
2026-07-22), made BEFORE any judge scored any real item.

## Motivation

The report seeding this repo argues the binding constraint on autonomous AI
research is *verification*, not generation, and names LLM-as-judge for research
ideas as a load-bearing but under-audited verifier. Two 2026 benchmarks already
measure its novelty scoring: **RQ-Bench** (arXiv:2606.12071) finds judges
*over*-rate the novelty of model-generated research questions (the "novelty
mirage"); **RINoBench** (arXiv:2603.10303) finds judge novelty verdicts diverge
sharply from expert gold, with a **central-tendency / regression-to-the-mean**
profile — models essentially never say "not novel" and rarely assign the top
grade, so both extremes are compressed. Neither localizes a mechanism, runs a
causal intervention, or separates novelty from correctness. We hypothesize one
mechanism that produces all of these symptoms and, if true, is fatal to naive
use of these judges in an RL / autonomous-research loop.

## Claim under test

**LLM research judges score *surface novelty-signaling* (framing/rhetoric that
asserts novelty) rather than *substantive novelty* (actual conceptual distance
from prior work), holding correctness fixed.** If so:

- it reconciles both benchmarks: signal-rich low-substance ideas are over-rated;
  plainly-framed genuinely-novel ideas are under-rated; across a corpus this
  yields RQ-Bench's mirage AND RINoBench's central compression + decoupling of
  score from true novelty (one mechanism, both tails);
- the judge is **reward-hackable**: an optimizer farms high novelty score by
  signaling without increasing substance — so the judge cannot be the verifier
  the autonomous-research thesis needs;
- the signaling response is carried by a **linear, dissociable direction** in the
  judge's residual stream (distinct from self-preference and uncertainty
  directions), and **projecting it out (decontamination) recalibrates the judge
  toward substantive novelty** — a cheap, deployable fix.

We do NOT claim to have discovered LLM-judge novelty miscalibration (RQ-Bench,
RINoBench, NovBench) nor the probe+steer machinery (Breaking the Mirror), nor
reward-hacking-by-rhetoric in general (persuasion/correctness precedents). The
contributions are the **novelty-signal-vs-substance decomposition under a
correctness-controlled crossed design**, the **novelty-specific reward-hack
demonstration**, a **probed, OOD-generalizing, dissociated novelty-signal
direction**, a **steering-as-decontamination recalibration to substance**, and
the **two-tails reconciliation (E)** — the load-bearing original result.

## Constructs (frozen definitions)

- **S — substantive novelty** ∈ {low, high}: conceptual distance of an idea's
  *technical content* from a fixed prior-work context. Within a stem we author
  two distinct contents: S=low (incremental extension of the context) and S=high
  (a genuine departure), matched on plausibility/soundness.
- **G — novelty signal** ∈ {low, high}: rhetoric asserting novelty ("challenges
  the prevailing view", "surprising", "a fundamentally new", "contrary to
  standard practice"), applied as a *framing rewrite that holds technical content
  fixed*. Quantified judge-independently (frozen lexicon + MiniLM
  difference-of-centroids, `src/novjudge/signal.py`) so the manipulation is
  checked, not assumed.
- **Correctness** is held at "plausible and sound" in every cell by construction
  and audited; a separate correctness manipulation (below) proves the judge *can*
  move on real quality.
- **Y — judge novelty score**: expected digit over the next-token distribution
  restricted to 1-9 under the frozen rubric (`src/novjudge/rubric.py`,
  `judge_local.expected_score`); pairwise as robustness.

## Design — the crossed 2×2 (the clean core)

Per **stem** (fixed prior-work context) we build four matched items crossing
S × G, correctness fixed. The S dimension varies *content*; the G dimension
varies *framing only*:

| | G=low (plain) | G=high (signaled) |
|---|---|---|
| **S=low** (incremental) | LL | **LH ← reward-hack cell** |
| **S=high** (departure)  | HL | HH |

- **Discriminant-validity anchor (correctness arm):** sound vs. subtly-broken
  variants at fixed S,G, to confirm the judge moves on real quality (so a null G
  effect can't be blamed on an insensitive judge).
- **Placebo:** two paraphrases at identical S and G — Y must not differ.
- **Rationalization probe (ported from Faithful-or-Fabricated / The Silent
  Judge):** after scoring the LH cell, elicit the judge's free-text justification
  and have an independent judge classify whether it attributes the (signal-driven)
  score to *substance*. Endpoint: **unfaithful-rationalization rate** — the judge
  is not just biased but misattributes the cause.

## Ground truth & item construction

- Stems generated across two domains (ML + economics) grounded in real subfields,
  via a **generator agent**; each stem yields the 2 contents × 2 framings.
- **G check (K4):** frozen lexicon count + MiniLM signal score must rank
  G-high > G-low within each content, or the pair is dropped.
- **S check (K4):** an **independent adjudicator agent** (role-separated from the
  generator and from the judges), blind to G, given the prior-work, must agree
  S-high departs and S-low is incremental; disagreements are dropped. Human
  spot-check on a stratified subsample (κ reported).
- **Correctness / leak audit (K4):** adjudicator confirms all four cells are
  equally sound and that G-high added no real content (manipulation-leak check);
  flagged items dropped.
- **Real-idea anchor (E5 only):** published ideas with human novelty/soundness
  sub-scores (OpenReview/RINoBench-style) for external validity; contamination
  (judges may know outcomes) flagged; never a primary endpoint.
- Generator ≠ adjudicator ≠ judges wherever possible; overlaps declared.

## Judges (roster)

- **Local (activation access + steering):** Qwen2.5-7B-Instruct,
  Llama-3.1-Nemotron-Nano-8B, Phi-3.5-mini-instruct — 4-bit (bitsandbytes), HF
  forward hooks (validated: Phi-3.5-mini loads at 3.72 GB peak; hooks fire).
- **API/frontier (behavioral external validity, no activations):** a Claude
  judge via subagent (harness), to show the effect is not a small-open-model
  artifact. Pointwise primary; pairwise robustness.

## Experiments

- **E0 (gate) — does signal beat substance?** Score the 2×2 across the local
  roster + the frontier judge on ≥ ~120 validated stems. Fit
  `Y ~ S + G + S:G + (1|stem) + (1|judge)`. **Gate:** significant β_G > 0. If
  β_G ≈ 0 and β_S > 0 → K1.
- **E1 (identification / confounds).** Report β_S, β_G, hackability index, the
  correctness-arm effect, the placebo null, and the unfaithful-rationalization
  rate. Establishes the G effect is not correctness, paraphrase noise, or a
  faithfully-reasoned judgment.
- **E2 (novelty reward-hackability).** A search agent with query-only access to a
  judge rewrites a fixed base idea to maximize Y under an **S-freeze** audited by
  the independent adjudicator + the judge-independent G score. **Endpoint:** Y
  rises while an independent substance metric does not (gamed gain). Framed as the
  *novelty analogue* of persuasion/correctness reward-hacking (2508.07805,
  2607.05904, 2606.04923), which we cite; the novelty-signal-specific hack is what
  we demonstrate.
- **E3 (mechanism — probed & dissociated direction).** Train a **supervised
  linear probe** (not only difference-of-means) on the scoring-position residual
  to separate G-high vs G-low; report AUROC **in-distribution and on a held-out
  domain (OOD)**. **Dissociation (vs Breaking the Mirror):** build a
  self-preference direction and an uncertainty/calibration direction; show the
  novelty-signal direction is distinct (low cosine) and that steering it moves
  novelty score but not self-preference behavior. Mediation: Y loads on the
  G-projection with S held fixed.
- **E4 (steering-decontamination fix).** **Project out / ablate** the G-direction
  during scoring (robustness with a probe-orthogonal variant); re-fit E0.
  **Endpoint:** β_G shrinks monotonically in the ablation strength AND calibration
  to S improves (β_S/β_G rises; hackability index falls) — recalibration to
  *substance*, not merely a verdict flip (the axis separating us from
  remove-self-preference and add-to-amplify-creativity). Off-target check:
  correctness-arm effect and general judging quality on a neutral benchmark must
  not collapse.
- **E5 (reconciliation).** On the real-idea anchor, test whether controlling /
  ablating G expands the compressed score range and restores substance-tracking —
  i.e., RQ-Bench's over-rating and RINoBench's central compression are two faces
  of one surface-signal bias.

## Primary endpoints (decided before results)

1. **Signal effect** β_G (mixed model), per judge, cluster-bootstrap CI.
2. **Substance effect** β_S and calibration ratio β_S/β_G.
3. **Hackability index** = P(Y[LH] > Y[HL]) over matched stems (>0.5 = signal
   beats substance) + the E2 gamed-gain.
4. **Unfaithful-rationalization rate** in LH.
5. **Mechanism:** probe AUROC (in-dist + OOD) + cosine dissociation from
   self-preference/uncertainty directions + mediation fraction.
6. **Decontamination:** Δβ_G and Δ(β_S/β_G) vs. ablation strength; off-target null.

## Pre-registered kill conditions

- **K1 — judges track substance.** β_G ≈ 0 while β_S > 0 → the worry is wrong for
  these judges; honest negative (the reconciling test fails; report the residual
  cause of the two-benchmark discrepancy).
- **K2 — behavioral only.** A G effect exists but there is no separable/dissociable
  linear G-direction (probe near chance or indistinct from self-preference) or
  ablation does not move β_G → demote to a behavioral benchmark + reward-hack
  result; drop the mechanism/fix claims.
- **K3 — reconciliation fails.** The surface-signal account does not span both
  tails (controlling G neither expands RINoBench compression nor removes the
  RQ-Bench mirage) → the mechanism is not (only) surface signal; report the
  residual.
- **K4 — construction confound.** S/G/correctness validation fails or adjudicator
  κ is low → fix construction (pre-result amendment, logged) before any scoring.

## Differentiation from prior work (cite-and-distinguish)

- **A** vs *Faithful or Fabricated?* (2605.23970), *The Silent Judge*
  (2509.26072): they hold content fixed and vary source/verbosity/confidence/
  recency cues; we cross a **novelty-signal × substantive-novelty** factorial with
  **correctness fixed by construction** and add their rationalization measure.
- **B** vs *Can You Trick the Grader?* (2508.07805), *More Convincing Not More
  Correct* (2607.05904), rubric-RL hacking (2606.04923): persuasion/correctness
  precedents; we demonstrate the **novelty-signal** hack with substance held fixed.
- **C** vs *Breaking the Mirror* (2509.03647), *Calibrating LLM Judges*
  (2512.22245), creativity steering (2412.06060): same probe/direction machinery
  on **other** attributes; ours is a **novelty-signal** direction, **dissociated**
  from self-preference/uncertainty, with **OOD** transfer.
- **D** vs Breaking the Mirror, honest-steering (2505.17760): steer-a-judge is
  established; ours is **decontamination that recalibrates to substantive
  novelty**, not a verdict flip or amplification.
- **E** — unclaimed: no work reconciles RQ-Bench (2606.12071) and RINoBench
  (2603.10303); typicality/familiarity bias (2510.01171) supplies an under-tail
  mechanism but no reconciliation. **Lead with E.**

## House discipline

Every paper number a machine-generated macro (`paper/numbers.tex` +
`verify_regen.py`) from committed `results/*.json`; bootstrap/cluster CIs; frozen
rubric/lexicon/exemplars (hashed); generator/adjudicator/judge separation; a
recency re-sweep before submission; **this file commits BEFORE any results
commit**; no experiment scored before S/G/correctness validation (K4). Must-cite
set recorded in `lit_notes.md`.

## Known constraints (declared up front)

- One RTX 3080, ~6 GB free: judges run 4-bit; steering via HF hooks (validated);
  no judge fine-tuning.
- Item generation + the frontier-judge arm run through harness subagents (no
  `ANTHROPIC_API_KEY` in-env); judges-under-test held separate from generator and
  adjudicator to avoid self-preference leakage.
