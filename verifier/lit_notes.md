# lit_notes.md — competitive landscape, the gap, and differentiation

Sweeps: initial 2026-07-22; adversarial gap review 2026-07-22 (workflow
`verify-p1-gap`, 6 agents, 3 deep reads + scoop-hunt + adjacency sweep +
conservative verdict). Preprint numbers provisional until read against source
PDFs. Verdict: **ADJUST** — integrated A–E program unclaimed; E clear; A–D ship
as cite-and-distinguish.

## The two benchmarks we reconcile (claim E — the crown jewel)

- **RQ-Bench** — *On the Limits of LLM-as-Judge for Scientific Novelty
  Assessment* (arXiv:2606.12071, DeCLaRe/NTU). 1,434 author-anchored RQs from 746
  arXiv CS papers. **Judges OVER-rate model-generated RQs ("novelty mirage"),
  stronger under pairwise; experts prefer author RQs** (expert win 78%/56% for GT
  vs judges favoring model outputs 82%/52%; human–LLM agreement 22% vs 60%
  expert–expert). Prompt-side mitigation only (scoring source-boundedness
  re-aligns judges). No crossed design, no correctness control, no mechanism, no
  steering, never mentions RINoBench. **must_cite.**
- **RINoBench** — *Is this Idea Novel?* (arXiv:2603.10303, Schopf & Färber).
  1,381 ICLR-review-derived ideas, gold = averaged reviewer novelty. **Judge
  verdicts diverge sharply from gold with a CENTRAL-TENDENCY profile: all models
  scored 0.0 F1 on category 1 ("not novel"), predictions concentrate mid-range,
  rarely top grade** → both extremes compressed. **Correction to our original
  framing:** this is NOT clean one-directional under-rating; it already contains
  both tails, which our single signal-tracking mechanism predicts. Behavioral
  only; no mechanism/steering. **must_cite.**

Our surface-signal account predicts exactly this joint pattern: signal-rich
low-substance → over-rated (RQ-Bench); plain high-substance → under-rated;
corpus-wide → central compression + score/substance decoupling (RINoBench). **No
paper reconciles the two — E is the least-threatened, load-bearing contribution.**

## Near-scoops — the method moves we must out-run (cite-and-distinguish)

- **Breaking the Mirror** (arXiv:2509.03647) — THE near-scoop for C+D. Localizes
  **self-preference** bias to a linear residual direction via diff-of-means/probes
  and **causally steers it out**, with cross-setting (partly OOD) generalization
  (~97% of biased samples flipped by 3/4 vectors). Same probe+steer recipe, other
  construct. → We must use a **trained** probe, require **OOD** transfer, and
  **dissociate** the novelty-signal direction from self-preference & uncertainty.
- **Faithful or Fabricated?** (arXiv:2605.23970) — crossed cue-invariance design
  (content fixed; vary source/verbosity/confidence; Blind/Truth/Flip/Placebo) +
  "rationalization bias" measure. Design template for A, different cues. → Port to
  novelty; adopt their rationalization bar.
- **The Silent Judge** (arXiv:2509.26072) — injects recency/provenance cues;
  judges follow them but rationalize as content quality. Empirical backbone for
  "signal drives score, reported as substance." → Different cue family; A/B support.
- **Can You Trick the Grader?** (arXiv:2508.07805) — persuasive language inflates
  judge scores with no correctness gain. B precedent (persuasion, not novelty).
- **More Convincing, Not More Correct** (arXiv:2607.05904) — self-play hacks a
  reference-free judge (pass 0.72→0.94, accuracy flat 0.20). B precedent
  (correctness domain).
- **Calibrating LLM Judges: Linear Probes…** (arXiv:2512.22245) — linear probe on
  a judge's residual stream with OOD generalization; steering follows. C precedent
  (uncertainty, not novelty-signal).
- **But what is your honest answer?** (arXiv:2505.17760) — steering vectors on
  judges to change verdicts. D precedent (honesty).
- **Steering LLMs to Evaluate and Amplify Creativity** (arXiv:2412.06060) —
  diff-of-means creativity direction, layer 8, add-to-amplify; ID-only, generator
  not judge. C/D precedent (creativity, add not ablate).

## Motivating / framing must-cites (not threats)

- **NovBench** (arXiv:2604.11543) — LLMs have only surface-level understanding of
  novelty; cleanest prior evidence for the surface-vs-substantive split.
- **Reliability without Validity** (arXiv:2606.19544) — the exact critique our
  A–E program operationalizes for novelty.
- **Typicality Bias / Verbalized Sampling** (arXiv:2510.01171) — annotators favor
  familiar text independent of quality; mechanism for the under-tail of E.
- **HindSight** (arXiv:2603.15164) — judges reward framing over impact; warns
  optimizing generators against judges → "impressive-sounding but vacuous."
  Motivates A/B qualitatively.
- Idea-evaluation context: Si et al. ideation study; Ideation-Execution Gap
  (2506.20803); AI Idea Bench 2025 (2504.14191); Artificial Hivemind (2510.22954).

## The gap, in one sentence

No prior work runs the correctness-controlled **novelty-signal × substance**
factorial (A), demonstrates the **novelty-specific** reward hack (B), probes a
**dissociated, OOD-generalizing novelty-signal direction** (C), uses **steering as
decontamination to recalibrate to substance** (D), or **reconciles RQ-Bench and
RINoBench** as one surface-signal bias (E). Every method move is precedented on an
*adjacent* bias — so the paper leads with E and ships A–D as explicit
cite-and-distinguish against Breaking the Mirror, Faithful-or-Fabricated, and The
Silent Judge.

## Six forced design changes (folded into PLAN.md before freeze)

1. **E:** correct RINoBench to regression-to-mean/both-tails; reframe E as a
   surface-signal account spanning both tails.
2. **A:** port cue-invariance to novelty; correctness fixed by construction; add
   the unfaithful-rationalization measure.
3. **B:** actually run the optimizer-in-the-loop; show substance doesn't rise.
4. **C:** trained supervised probe + OOD transfer + dissociation from
   self-preference/uncertainty directions.
5. **D:** frame steering as decontamination → restored calibration to substance.
6. **Global:** add reliability-without-validity framing; cite NovBench.
