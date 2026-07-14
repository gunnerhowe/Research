# Project status: PAPER 3 COMPLETE (all five rungs) + Paper 1 v4 ready + P4 amendment PRE-REGISTERED
Updated: 2026-07-14

## P4 PRE-REGISTERED AMENDMENT (E2/E3/E4) — committed 2026-07-14 BEFORE any amended run starts

Context (from the completed free-norm cells of runs/grid4, 4 seeds each): supcon_aug is SLOWER at
free norm (4/4 seeds, t_gen 41.6-46.0k vs baseline 29.8-39.4k) BUT the logs show why: the aux
term's known norm side-effect (Channel 2) reaches equilibrium with weight decay and FREEZES the
weight norm at ~92 for the whole run, while baseline's norm decays 94->63 and it only generalizes
once norm reaches ~66. So supcon_aug generalized AT norm ~91-94 — a norm where baseline shows 21%
test acc — while carrying a frozen-clock handicap. supcon_label groks at norm ~101-104 (2.2x
faster). shufpair = ceiling 0.76-0.79 + norm explosion to ~136 (whether-block transfers). augce
abolishes the delay outright (t_gen < t_fit; regularizer, different phenomenon). The free-norm
race therefore confounds Channel 1 (structure) with Channel 2 (norm drag) — the exact confound
Papers 1-3 exist to separate. Audit found no bug (loss impl, pairing, ceilings, t_fit all clean);
the free-norm negative is REAL and will be reported regardless (see D2).

Amended arms (scripts/run_paper4b.py -> runs/grid4b, seeds {0,3} first, {1,4} via --extend):
- E2 matched-norm sweep: clamp {35,50,65,80,92} x {baseline, supcon_aug lam0.3}; c23 cells reused
  from grid4. Decisive cell: base_c92 — does baseline EVER generalize at the norm where the prior did?
- E3 content test: supcon_nn — label-free CROSS-EXAMPLE couples (greedy globally-nearest pixel
  matching; same batching/label machinery as shufpair so couples differ ONLY by construction;
  couple label purity measured & reported, never trained on). Free norm, lam 0.3.
- E4 dose: supcon_aug lam {0.03, 0.1} (less Channel-2 drag at lower dose; Paper 1 says dose matters).

Predictions (locked now):
- P1 baseline delay grows steeply with pinned norm; expected censoring at c80-c92 (100k budget).
- P2 aug_c92 groks in finite time ~40k (see D1: this value is post-hoc known from free-norm runs).
- P3 log(delay)-vs-norm slope baseline >> supcon_aug; slope ratio >= 2 = natural-data exponent flattening.
- P4 supcon_nn beats same-seed baseline at free norm (conditional on measured purity >= 0.8); nn >= aug.
- P5 cos_gap (class clustering) rises well before test acc in supcon_aug (probes lead behavior).

Kill criteria (any fires -> reported as stated, no reframing):
- K1 base_c92 groks within 1.5x of aug_c92 on both seeds -> NO matched-norm benefit; the free-norm
  negative stands in full and the paper is the sharp negative.
- K2 slope ratio < 2 -> no meaningful flattening on natural data.
- K3 supcon_nn fails to beat baseline on both seeds despite purity >= 0.8 -> cross-example label-free
  priors do not transfer at free norm on MNIST.

Disclosures:
- D1 P2 is not a blind prediction: free-norm supcon_aug sat at ~92 naturally and grokked 41.6-46k;
  the genuinely blind cells are every base_cXX and every pin below 92.
- D2 The original free-norm result (supcon_aug slower, 4/4) is reported in the paper regardless of
  E2-E4 outcomes, alongside whatever these amendments show, labeled as a pre-registered amendment.
- D3 supcon_nn purity is a measured property of MNIST, reported with the result.
- D4 One dead end already tried and kept: nothing — no amended run, pilot, or partial run of any
  E2/E3/E4 cell exists at commit time (grid4b/ does not exist yet).

## Paper 3 (paper2/main.pdf, 7pp) — "What Makes a Representational Prior Work?"
178 new runs (runs/grid3). All numbers macro-generated (paper2/gen_numbers.py), verify_regen PASSES.
- R1 band control: 1/15 grok ≈ random (p=0.43), ≠ true (p=2.3e-5) → FEATURE-FAMILY account CONFIRMED.
- R2 commutativity (label-free): 15/15 grok, 2.7x median; MORE reliable than supervised (p=0.038);
  comm+clamp45 = best method overall (5/5, median 17x, sub-1000 on some seeds).
- R3 windows: [0,2k] 10/10 @2.7x beats always-on (8/10 @1.25x) and anneal; brief beats sustained;
  early advantage real. Wall-clock objection dissolved (aux loss runs 4% of training).
- R4: mul replicates everything (shuffled 0/5, clamp 5/5 @9.8x); sub whether-rescues; 2-layer + LN
  traps replicate and clamp repairs; MLP mildest. EXPONENT: CE slope 0.344/unit (x31 per +10, lower
  bound, c65 censored) vs SupCon 0.020 (x1.22) → ~17x flattening. Low-norm floor + inversion at c35.
- R5 boundary: agreement grammars + small parity generalize-then-degrade (no memorize-first phase);
  prior lifts levels not trends.
- Corrections of companion stated in-text: speed at optimal clamp belongs to norm control (omnigrok);
  prior buys exponent-flattening robustness + whether-generalization.
TODO before submission: author affiliation check; decide arXiv timing vs Paper 1; consider sending
Khanh both PDFs (he's acknowledged in both).

## External review response (2026-07-14): 13/14 items fixed, both papers recompiled, both
verify_regen PASS. Cross-cites added both ways (companion2026a/b bib entries); Paper 2 abstract
"strongest = reliably fast without knowing critical norm"; pool enumerated (2 mitigations x 2
strengths = 40); bib authors fixed (NeuralGrok, task-intrinsic); 56-vs-57 trajectory caveat;
comm peak norm MEASURED (103 vs true 130 — corrected my "never builds up" overclaim); n=3-15;
17x collision de-fused; grokks typo; censored markers on exponent fig; non-vacuity 51/95 macro.
Item 3 answered WITH DATA: new matched-duration window [2k,4k] arm (10 seeds): 10/10 @2.1x —
early [0,2k] still wins at matched duration (2.7x vs 2.1x) AND matched-late beats 4x-longer late
(2.1x vs 1.9x), so brief>sustained and early>late are now separately evidenced. Item 14 (Paper 1
abstract length) left to user. grid3 now 188 runs.

## Paper 1 v4 (paper/) — unchanged since 2026-07-09, still unsubmitted as far as known

## v4 changes (response to user's review of v3)
1. POOLED-SIGNIFICANCE REFRAME (the one that mattered): the stall-removal p-value is a POOLED effect;
   per method it's 10/10 vs 8/10, Fisher p=0.47 (not significant — only 2 stalls at n=10). Abstract,
   Section 5, table caption, and contributions now LEAD with the robust monotone speedup (needs no
   p-value) and explicitly state the stall p is pooled/per-method-underpowered. The accelerator's case
   rests on the speedup, stall counts corroborating not load-bearing.
2. REGEN DISCIPLINE (house rule 1, was missing here vs other papers): paper/gen_numbers.py computes
   every cited number from run artifacts -> paper/numbers.tex (\ensuremath macros) + numbers.json;
   main.tex \input{numbers} and cites ONLY macros; paper/verify_regen.py regenerates + git-diffs
   (PASSES). This CAUGHT 2 stale hand-typed numbers: pooled stall was 0/36 p=1.2e-3 (pre-final-runs),
   correct is 0/40 p=7.7e-4; anneal speedup 1.65x -> 1.7x. Run `python paper/verify_regen.py` before push.
3. MEDIAN/MAX FIX: "up to 8.6x" was the MEDIAN; now "median 8.6x (up to 22x on fastest seeds, 850 epochs)".
4. Accelerator-novelty: abstract keeps "confirm the mechanism by prediction" dominant; speedup framed as
   "the weight-norm delay law run as a control knob" (not a novel accelerator).
- Paper now 16pp. arxiv_upload.zip MUST include numbers.tex AND references.bib (added) — clean-room
  tectonic compile verified 16pp, References intact, 0 unresolved. arxiv_abstract.txt updated (still <1920).
- Table 1 per-lambda rows hand-typed but verified vs analysis/out/results.csv (8/10 28725, 6/10 33325,
  8/10 24400). All headline + abstract + Section 5 + Paper 2 numbers are macro-backed.

## v3 changes (Paper 2 fold-in)

## Deliverable
paper/main.pdf — 15-page arXiv-ready draft, compiles clean via tools\tectonic.exe (uses bm pkg).
Now includes Section 5 "Controlling the norm side-effect yields a reliable, fast accelerator".
arxiv_upload.zip = main.tex + main.bbl + figs (8 figures). arxiv_abstract.txt = 1821-char
submission abstract (updated with accelerator result). zenodo_code_bundle.zip = full repo snapshot.

## What v3 added (the Paper 2 fold-in; user's + Khanh's ideas)
Mechanism prediction (Sec 4.4 race account) CONFIRMED: removing the norm-inflation side-effect
(Channel 2) removes the bimodal stalls while keeping/amplifying speed. Final numbers (runs/grid2,
analysis/out2 via analyze_paper2.py):

- Method comparison (SupCon-true lam=1.0, vs baseline, 10 seeds each):
  * norm-clamp45 (||W||=45, STANDALONE): 10/10 grok, median 8.6x, fastest 850 epochs (sub-1000). WINNER.
  * norm-replay (baseline traj): 10/10, 6.6x, norm 56 (needs baseline run).
  * adaptive-lambda(35/15) (closed-loop controller, user's idea, STANDALONE): 10/10, 2.8x, norm 57.
  * anneal (lambda->0 by 10k): 10/10, 1.65x, norm 120.
  * adaptive(50/40) loose: 8/9 (under-tuned).
  * SPEEDUP TRACKS NORM: lower held norm = faster grok (delay law run in reverse, constructively).
- Stall removal (pooled lam 0.3+1.0): mitigations 0/36 vs unmitigated 6/20, Fisher p=1.2e-3.
- Composition NEGATIVE: SupCon+Grokfast at default gf_lamb=2.0 stalls 0/10 (over-amplifies the
  SupCon-dominated gradient; hyperparameter artifact, user caught this). De-tuning (0.2) helps
  monotonically but stays seed-unreliable (4/10) and adds no speed. They do NOT beneficially compose.
- Timing invariant still holds across all runs.

## Code changes (all ADDITIVE, off by default — Paper 1 reproduces unchanged)
src/train.py flags: --lambda_anneal_epochs, --norm_traj (now works on supcon too), --norm_clamp,
--use_grokfast, --adaptive_lambda/--norm_target/--norm_band. Runners: scripts/run_paper2.py (wave 1),
run_paper2b.py (wave 2: adaptive50, gfgentle, lam0.3 mitigations), run_paper2c.py (wave 3:
normclamp45, adaptive35). Analysis: analysis/analyze_paper2.py.

## HONEST framing locked in the paper (do NOT oversell)
- The accelerator is presented as CONFIRMATION OF THE MECHANISM, not a production recipe.
- It REQUIRES known task structure (SupCon positives) -> not usable in frontier pretraining as-is;
  self-supervised positives = future work. Stated in Limitations.
- Grokfast is a RESEARCH baseline, NOT a big-lab production tool. "beats Grokfast" is only a raw
  in-setting speed number; methods have different requirements (norm-clamp needs known structure).
- Wall-clock: contrastive term ~1.5x/epoch, so only ratios < ~0.67 are net wall-clock wins (clamp's
  8.6x clears it; weaker mitigations may not).

## Before arXiv submission (user actions)
1. Confirm submission status: is Paper 1 already on arXiv (v1)? -> post v2. Not yet? -> submit v3 as v1.
2. Author block still "Gunner Levi Howe" placeholder — confirm.
3. Optional: reply to Khanh with the folded result (he suggested the wrong-structure control; the
   composition-artifact catch was the user's).
4. Submission abstract = paper/arxiv_abstract.txt (1821 chars). Upload arxiv_upload.zip. cs.LG primary,
   stat.ML cross-list (needs stat group registration, or add post-hoc), arXiv non-exclusive license.
