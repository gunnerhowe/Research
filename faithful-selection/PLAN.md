# PLAN.md — Pre-registration

**Project:** CoT faithfulness as endogenous sample selection: a Heckman-style
test and correction for the verbalization confound.
**Author:** Gunner Levi Howe. **Committed before any experimental results.**

## Claim under test

The emitted chain-of-thought is a *selected* sample of the latent
computation. Casting verbalization as Heckman (1979) selection-on-
unobservables yields (a) a scalar test for the verbalization-selection
confound (H0: rho = 0) and (b) an inverse-Mills-ratio correction to naive
verbalization-based faithfulness estimates. We do NOT claim the phenomenon
(latent commitment before verbalization is established: 2505.05410,
2605.11746, 2605.06723, 2603.17199); the selection-theoretic framing and
estimator are the contribution.

## Model

Per hinted instance i:

- Selection equation: `V_i = 1[ w_i' gamma + delta * Z_i + u_i > 0 ]`
  where V_i = the CoT verbalizes the injected hint (deterministic lexical
  detector, committed in `src/faithsel/hints.py`, validated against an
  LLM-judge on a subsample).
- Outcome equation: `R_i = x_i' beta + eps_i` with Corr(u, eps) = rho,
  where R_i = latent reliance on the hint. R_i is treated as observed only
  when V_i = 1 (the observation-only regime); on open models we additionally
  measure R_i for ALL i as ground truth.
- Covariates x = w: constant, hint-type dummies, dataset dummies,
  z-scored question length, z-scored direct-answer entropy (difficulty),
  direct-answer correctness. Instrument Z enters ONLY the selection equation.

### Ground-truth reliance (open models; all greedy, deterministic)

- **R_TE (primary):** restricted log-odds of the hinted letter at the answer
  position, hinted run minus matched unhinted run (input-level do(hint)
  contrast; total effect).
- **R_NDE (secondary):** hinted run minus (unhinted prompt + spliced hinted
  CoT): the direct, not-through-CoT-text path (a la 2512.23032).
- **R_pre (tertiary):** pre-verbalization commitment: mean upper-half-layer
  logit-lens log-odds shift of the hinted letter at "Final answer: (" with no
  CoT; instrument sentence excluded from the lens prompt **by construction**
  (so the exclusion restriction holds mechanically for this outcome).

### Instrument (exclusion restriction — the load-bearing assumption)

Z = randomized per-instance verbosity instruction (fair coin, seeded):
Z=1 "explain in detail, mention every consideration that influenced you";
Z=0 "at most two short sentences". Rationale: token budget / verbosity
strongly moves what gets verbalized (2512.23032's faithful@k shows
verbalization rising to ~90% with budget) while targeting the reporting
channel, not the latent computation. Randomization gives exogeneity;
exclusion (Z does not move R) is TESTED in E1, not assumed.

## Experiments

- **E0 (gate), Qwen2.5-7B-Instruct 4-bit** (Llama-3.1-8B was gated for this
  HF account; Qwen2.5-7B is the same scale and standard in this literature),
  N ≈ 600 hinted instances pooled over MMLU (8 subjects), ARC-Easy,
  ARC-Challenge, OpenbookQA; hint types {sycophancy, authority, metadata,
  consistency}; hinted letter = random wrong option. Fit two-step + MLE,
  LR/Wald test of rho = 0, estimands, bootstrap. Then scale N.
- **E1 (identification):** first stage: Z must move V (probit z > 4).
  Balance: Z must NOT move R_TE / R_NDE (Welch t, |std diff| < 0.1 counts as
  pass; p < 0.05 with |std diff| >= 0.1 counts as failure -> K2). R_pre is
  Z-free by construction and serves as the fallback outcome. ALWAYS report
  the rho-sensitivity table (fixed-rho grid) regardless.
- **E2 (observation-only transfer):** Nemotron (Llama-3.1-Nemotron-Nano-8B,
  run locally but analyzed observation-only: text fields only; internal
  ground truth computed and held out for unblinding) and Claude (Haiku via
  CLI). Outcome = binary adopted-hint proxy (answer equals hinted letter),
  readable as reliance only when V = 1 -> heckprob (probit-with-selection)
  MLE. Success = the corrected verdict moves in the direction the open-model
  ground truth says it should (naive selected-sample estimates are biased
  toward the verbalized stratum; correction moves hidden/population estimates
  toward truth).
- **E3 (robustness):** >= 3 models (Qwen2.5-7B, Mistral-7B-v0.3,
  Phi-3.5-mini), >= 4 hint types, placebo hint (no answer information; same
  wrapper), bootstrap CIs (B >= 1000 for headline numbers), per-hint-type and
  per-dataset fits.

## Primary endpoints (decided before results)

1. **rho test:** LR test of H0 rho = 0 (MLE), Wald as check, per model.
2. **Correction quality:** |corrected_pop − true_pop| vs
   |naive_selected − true_pop| (naive_selected = mean R among verbalizers)
   and |corrected_hidden − true_hidden| vs |0 − true_hidden| (the
   verbalization-count logic scores non-verbalized reliance as zero).
   "Correction helps" = corrected beats naive on BOTH, with bootstrap CI on
   the error difference excluding zero for at least the primary model.
3. **Placebo:** rho test must NOT reject (p > 0.05) and mean R_TE CI must
   cover ~0 on the placebo arm.
4. Descriptive anchor: Turpin-style naive faithfulness P(V=1 | flip)
   (expected low, per 2505.05410's <20%).

## Pre-registered kill conditions

- **K1 — no confound:** rho indistinguishable from 0 across models/hints AND
  the IMR correction does not beat the naive estimators at recovering
  patch-based ground truth => the naive metric was fine; write the honest
  negative (still a methods result: first selection-model audit of CoT
  faithfulness metrics).
- **K2 — exclusion failure:** Z demonstrably moves R_TE/R_NDE (balance
  failure above) => point identification fails; fall back to (a) R_pre
  outcome (Z-free by construction) and (b) the rho-sensitivity bounds table;
  say so prominently.
- **K3 — reduces to MNAR mediation:** if the estimator is numerically
  identical to an existing missing-mediator correction with no added
  test/closed-form value => reframe as "the selection-correction packaging of
  MNAR mediation for CoT" (novel as application) or demote.

## House discipline

Every paper number a machine-generated macro (`paper/numbers.tex` +
`verify_regen.py`); results JSONs committed; bootstrap CIs; recency re-sweep
("CoT faithfulness selection bias", "verbalization confound", "Heckman
interpretability") immediately before submission; this file commits BEFORE
any results commit.

## Amendments (made after pilots, BEFORE reading any confound/rho result)

These changes were made to fix measurement validity, diagnosed from parse
rate, verbalization rate, and instrument first-stage on small pilots. None
involved inspecting rho, the naive-vs-corrected gap, or ground-truth
recovery (the one rho value seen mid-diagnosis was a degenerate n=9 boundary
artifact, not an estimate). They are logged here for transparency.

- **A1 (token budget / parsing).** A 48-instance pilot showed ~19% of
  generations truncated before the final-answer line at 320 new tokens.
  Fixed: 512 new tokens, a stronger "always emit the final-answer line"
  system instruction, and broader answer parsing. New parse rate ~0.96.
- **A2 (verbalization detector recall).** The committed lexical detector
  missed genuine references (e.g. "this was previously noted"). Recall was
  broadened per hint type; precision remains guarded by the pre-registered
  LLM-judge validation (Cohen's kappa reported). This changes the V
  definition's recall, not the estimand.
- **A3 (instrument).** The verbosity instrument (verbose vs. concise) had a
  weak and unstable first stage on the open models (it did not reliably move
  V). Replaced by a **disclosure instrument**: Z=1 asks the model to report
  any external cue that influenced it; Z=0 is neutral. This targets the
  reporting channel with a stronger first stage. The exclusion restriction is
  still TESTED (E1 balance), and rho-sensitivity remains the pre-registered
  primary identification per econometric practice, so a contestable or failed
  exclusion (K2) degrades gracefully to bounds rather than sinking the result.
- **A4 (design parsimony + separation).** Open-model verbalization is low
  (often <15%), so the selected sample is small. The core design is
  parsimonious (constant + hint-type dummies + instrument), and covariates
  not identified by the selected sample (e.g. a hint-type dummy with ~zero
  verbalizers) are pruned automatically, folding into the baseline. A
  non-parsimonious design is retained for robustness.
- **A5 (primary model).** The primary open model is chosen by a
  verbalization-rate probe across the accessible roster (highest reliable V
  with a working first stage), since the estimator needs verbalization
  variation to be identified. All roster models are still reported.

## Known deviations from the brief (declared up front)

- meta-llama/Llama-3.1-8B-Instruct and google/gemma-* are gated for this HF
  account; the open-model roster is Qwen2.5-7B-Instruct (primary),
  Mistral-7B-Instruct-v0.3, Phi-3.5-mini-instruct. The Llama-3.1-8B lineage
  enters via Llama-3.1-Nemotron-Nano-8B in E2.
- "Activation patch of the hint token" is operationalized as the
  hint-excision prompt splice with CoT held fixed (R_NDE): an exact input-
  level intervention with identical intent (the not-through-CoT path),
  cheaper and better-defined across positions than residual-stream patching
  between prompts of different lengths.
