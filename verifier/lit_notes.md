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

- **Breaking the Mirror** (arXiv:2509.03647) — CORRECTED after independent
  verification (2026-07-22, workflow verify-foundations). It reduces UNJUSTIFIED
  self-preference via activation steering (up to 97%) BUT its headline caveat is
  that steering is **UNSTABLE** and self-preference **"spans multiple or nonlinear
  directions"** — it does NOT cleanly localize to a single linear direction, and it
  does NOT show cross-setting generalization (the opposite: instability across
  legitimate self-preference / unbiased agreement). My earlier notes/commits/paper
  said "localizes to a linear direction + steers it out + cross-setting" = WRONG
  (overstated). CONSEQUENCE: our E4 "detectability != steerability" is CONVERGENT
  with BtM (both find judge-bias steering doesn't cleanly work), NOT a contrast.
  main.tex reworded (do not ship the "contra BtM" framing). We still use a trained
  probe + OOD + dissociation, but frame E4 as extending BtM's instability finding to
  novelty-signal, not refuting a clean BtM success.
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

- **NovBench** (arXiv:2604.11543) — LLMs have only "limited understanding" of
  novelty (its wording; it does NOT itself draw a surface-vs-substantive split — see
  verification note below; do not attribute that dichotomy to it).
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

## Citation verification + v4 prior-art (2026-07-22, workflow verify-foundations)

Prompted by the user asking "how do you know the literature is even true?" —
independently fetched + checked each load-bearing citation:

- **RQ-Bench (2606.12071): CONFIRMED, verbatim.** "novelty mirage" is the paper's
  own abstract term; experts prefer author-anchored RQs. **E5 anchor #1 SOUND.**
- **RINoBench (2603.10303): CONFIRMED, verbatim.** Central-tendency confirmed in the
  body (0.0 F1 on lowest category, clusters on 3-4, avoids extremes). **E5 anchor #2
  SOUND.** => the lead contribution (E5 reconciliation) holds.
- **Breaking the Mirror (2509.03647): PARTIAL — I OVERSTATED IT.** See corrected
  entry above. Reworded in main.tex.
- **NovBench (2604.11543): PARTIAL.** Says "limited understanding" not "surface-
  level"; makes no surface/substance split. main.tex reworded to the paper's wording.

**v4 (retrieval/literature-grounded novelty verification) = SCOOPED (severe).** It is
a crowded 2025-26 subfield; must NOT be positioned as introducing the idea, and its
prose-only baseline is already beaten:
- **OpenNovelty** (2601.01576) — near-exact: extract claims -> retrieve real papers
  -> contribution-level comparison -> verifiable report; deployed on 500+ ICLR 2026.
- **ScholarEval** (2510.16234) — RAG idea eval on novelty AND SOUNDNESS (empirical
  validity grounded in literature) — its soundness axis overlaps the fabricated-
  mechanism-detection angle I proposed for v4.
- **Idea Novelty Checker** (2506.22026), **NoveltyRank** (2512.14738),
  **Sakana AI-Scientist v2** (Semantic-Scholar novelty gate, deployed),
  **MemoNoveltyAgent** (2603.20884), **2503.01508**.
- If v4 proceeds at all: must benchmark a measured DELTA over OpenNovelty/ScholarEval
  on their public data (e.g. ScholarIdeas, 117 ideas), least-saturated slice =
  isolated fabricated-MECHANISM detection (still adjacent to ScholarEval soundness).
  Must cite all the above in related work.
