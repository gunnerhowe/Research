# PLAN — Selection Models for Machine Learning (two-paper program)

Working plan per info.txt. Every deviation from the brief is recorded in the
"Deviations & decisions" section below, dated.

## Status board

### Shared core
- [x] Package skeleton (src/heckesel), pyproject, PLAN
- [x] selection.py: probit, classic two-step Heckman, joint MLE (BFGS + exact-Hessian Newton polish)
- [x] synth.py: controlled selection generators (MAR / unobservables / mixtures, instrument on/off)
- [x] A-E0 faithfulness gate — **GATE PASS** (11 tests): two-step AND MLE match
      Cameron & Trivedi's seven-digit Stata output on RandHIE (loglik -10170.11,
      rho .7356, sigma 1.5701 reproduced); probit == statsmodels to 1e-8;
      statsmodels-composed two-step on Mroz87 matched to 1e-8; large-n synthetic
      recovery (rho, sigma, beta) with correct rho=0 MAR control.

### Paper A — Heckman-Corrected Epistemic Uncertainty (paper/) — COMPLETE (9pp)
- [x] deep.py: NN-feature-map Heckman (two-step + joint MLE w/ warm-up), corrected predictive variance
- [x] uq.py: deep ensembles, MC dropout, GP, oracle-IW, selection-blind two-head ablation
- [x] A-E1 (8 seeds) — **KILL CHECK CLEARED**: at rho=0.9 w/ instrument, oracle-IW
      cov 43.1% (WORSE than blind ensemble 64.4%), Heckman restores 88.9%
      (near oracle 89.9%), rho_hat=0.90; kill condition does NOT fire
      (paired Wilcoxon p=0.008). No-instrument honesty curve reported (MLE
      coverage inflated by weak-identification variance; bias stays large).
- [x] A-E2 semi-real (California, wine; induced MNAR; 8 seeds headline) —
      **honest, nuanced**: by region-ECE (the un-gameable metric) two-step
      Heckman is best non-oracle on BOTH (ECE 0.051/0.138; p<0.001 vs
      ensemble/IW/dropout); oracle-IW worst; coverage alone muddy because
      dropout/GP over-widen everywhere (GP 98% well-region coverage).
- [x] A-E3 full baseline suite + region-split ECE/coverage + paired tests
- [x] A-E4 PWC reporting-panel vignette: no-instrument boundary pathology,
      8/13 fits at |rho_hat|=1 (discussion only)
- [x] Paper A figures, numbers.tex (193 macros, auto), compiled clean, **regen-diff PASS**

### Paper B — Survivor Bias in Learning-Curve Surrogates (paper2/) — COMPLETE (8pp)
- [x] lc.py: pow3 family, SH censoring harness, naive/Tobit/EB/Heckman surrogates
- [x] B-E0 gate (8 seeds) — **GATE PASS**: survivor bias dose-responsive in
      noise (0.001->0.054) and eta; NEGLIGIBLE at LCBench-median noise
      (informative null about SH robustness, reported)
- [x] B-E1 corrected surrogate: multi-bracket Heckman (rung-assignment
      instrument) halves survivor-population bias but OVERSHOOTS; MAR
      all-prefix estimator near-unbiased and recommended; Tobit ablation;
      Arellano-Bond variant tried and dropped (worse than naive)
- [x] B-E2 LCBench+PD1 replays (160 cells): naive pow3 hurts SH, corrected
      ties last-value on LCBench, WORSE on PD1 (prediction-decision gap)
- [x] B-E3 baselines: last-value, pow3/DPL, variance-penalty; FT-PFN cite-compared
- [x] B-E4 null control: one-param variance penalty captures most of the
      decision benefit; nothing beats plain last-value SH
- [x] Paper B figures, numbers.tex (101 macros, auto), compiled clean, **regen-diff PASS**

### Program
- [x] Recency sweeps re-run pre-submission (logged above, 2026-07-05) — gaps hold
- [x] README + laymen.md + verify_regen scripts (both papers)
- [x] Atlas link-back note (below; cards to stamp on Zenodo ship)
- [x] 20 pytest green; both regen-and-diff PASS

**PROGRAM COMPLETE 2026-07-05.** Both papers build clean and reproduce
byte-identical. DOI slots (`10.5281/zenodo.TBD`) to fill at deposit.

## Atlas link-back (E:/GitHub/ai-atlas/ui/public/greens_ledger.json)

Two in-progress cards feed this repo:
- "Heckman selection correction (inverse Mills ratio) -> Epistemic
  Uncertainty" (Paper A). Honest outcome to stamp on ship:
  **positive-with-caveat** -- selection on unobservables defeats oracle
  importance weighting; the Heckman-corrected predictive distribution
  restores selected-against calibration WHEN an instrument exists, and
  degrades measurably without one (honesty curve reported). Methods finding:
  deep two-step > joint MLE for stability. Real-world vignette:
  benchmark-panel corrections are unidentified without a reporting
  instrument (boundary pathology observed).
- "Heckman + Arellano-Bond dynamic-panel -> HPO" (Paper B). Honest outcome:
  **null-with-mechanism** -- survivor bias is real and dose-responsive in
  the controlled family, but negligible at LCBench-median noise (informative
  null about SH robustness); the corrected surrogate fixes prediction and
  rescues naive extrapolators from a winner's curse but does not beat plain
  last-value SH at decision level on clean benchmarks (prediction-decision
  gap reported). Arellano-Bond variant tried and dropped (worse than naive).
  When each paper ships to Zenodo, tell the atlas assistant to ingest it as
  an own-lab node and stamp the card with these outcomes.

## Pre-registered gates and kill conditions (from info.txt, verbatim commitments)

- A-E0: our Heckman implementations reproduce classic econometric reference
  results (tests). No experiments before this gate passes.
- A kill: if IW-with-oracle-propensities matches Heckman under
  selection-on-unobservables in the controlled synthetic, the premise is wrong
  -> report and stop.
- A methods-contingency: if the inverse-Mills term is unstable with deep
  feature maps, characterize (two-step vs joint MLE, regularization) — a
  methods finding either way.
- B-E0 gate: survivor bias exists and grows with noise/selection pressure at
  noise levels calibrated from LCBench residuals. If negligible -> informative
  null about SH robustness; report it.
- B kill: corrected surrogate helps predictions but not decisions at matched
  budget -> report the prediction-decision gap honestly (itself a finding).

## House discipline checklist (enforced)

1. Every number machine-generated (numbers.tex via gen_paper_numbers.py);
   every prose ratio a macro; regenerate-and-diff at submission.
2. Tests green via plain pytest.
3. >=3 seeds (8 for headline comparisons), mean±sd, paired tests, Wilcoxon
   floors stated at small n.
4. Identification caveat (exclusion restriction) stated up front in BOTH papers;
   every grid has instrument-present AND instrument-absent conditions.
5. Honest abstract; "What we do not claim"; Limitations; Reproducibility with
   DOI slot FILLED at submission.
6. Recency sweep before each submission, logged here with date.
7. Cite econometrics primaries (Heckman 1979; Tobin 1958; Arellano-Bond 1991 if
   used) and the ML-adjacent lines named in info.txt.

## Kill-check log

- 2026-07-04: Atlas kill-check clean for Paper A card (3M-abstract index + web
  + adversarial pass): no deep-learning instantiation of selection-corrected
  epistemic UQ. Nearest lines to cite and distinguish: covariate-shift/IW UQ,
  selective labels (Lakkaraju et al.), PU learning, Cortes et al. 2008.
- 2026-07-05: Atlas kill-check clean for Paper B card: LC-surrogate line
  (freeze-thaw BO, DyHPO, DPL, FT-PFN) active; nobody models the censoring
  mechanism of SH/Hyperband/ASHA.
- 2026-07-05 (pre-submission re-sweep, DONE): newest-first web sweeps.
  Paper A queries: "Heckman selection correction epistemic uncertainty deep
  learning", "sample selection bias uncertainty quantification inverse Mills
  ratio neural network unobservables". Findings: 2025-26 epistemic-UQ work is
  evidential/credal/deep-ensemble calibration under covariate shift (Juergens
  et al. 2025; deep-ensembles frequentist arXiv:2510.22063) -- all selection-
  on-observables. NO deep-learning instantiation of Heckman-corrected
  epistemic UQ. Gap holds; novelty statement stands.
  Paper B queries: "learning curve extrapolation survivor bias successive
  halving Hyperband censoring", "...regression to the mean 2025". Findings:
  the closest new work, LKGP-for-SH (arXiv:2508.14818, 2025), fits a better
  LC predictor (latent Kronecker GP) to cut premature pruning but trains on
  survivor curves WITHOUT any selection/censoring correction (verified by
  fetching the paper: zero mentions of selection/survivor bias/Heckman/Tobit)
  -- cited as recent corroboration; the gap holds. LCDB 1.1 (arXiv:2505.15657)
  documents ill-behaved curves but not the selection link; cited. No scoop.

## Deviations & decisions

- 2026-07-05: statsmodels has NO mainline Heckman estimator (it never left the
  sandbox PR). A-E0 faithfulness is therefore tested against (i) statsmodels
  Probit (selection head must match to machine precision), (ii) a
  statsmodels-composed classic two-step pipeline (Probit + OLS with IMR
  regressor) on Mroz87 — our two-step must match to machine precision,
  (iii) published sampleSelection (R) two-step and ML estimates on Mroz87
  (Toomet & Henningsen, JSS 2008) as external reference values, and
  (iv) large-n parameter recovery on synthetic data with known truth.
