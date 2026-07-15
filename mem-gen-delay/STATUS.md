# Project status: THREE PAPERS SHIPPED (P1 on arXiv v3, P4 submitted) + P5 PRE-REGISTERED
Updated: 2026-07-14

## P5 "Does Interpretability Buy Warning Time?" — PRE-REGISTERED (see plan_p5.md)
Emergence-timing forecasting benchmark on the existing 466-run corpus + prospective rung.
R0 lit kill-test DONE (lit_notes_p5.md): Notsawo 2306.13253 = whether-from-loss-oscillations
(becomes our baseline feature); 2603.29805 = spectral early-warning indicators WITHOUT lead
time/calibration/FA/baselines; 2511.16893 = IH timing from pre-training config (kills our IH
headline -> stretch rung on seed variance); devinterp LLC = retrospective stage detection.
Surviving lane N1-N6: scored forecasts (lead time @ <=5% FA), probe-vs-scalar horse race,
interventional-shift robustness (P3: norm forecasters break on norm-decoupled supcon arms,
probes survive), negative-control FA accounting, prospective prereg, corpus release.
Kill criteria K1-K7 per rung in plan_p5.md, committed BEFORE R1 code exists.

### P5 R1/R2 FROZEN-SPEC VERDICTS (2026-07-14; frozen at analysis/freeze_eval_p5.py commit,
### test block executed once after)
- R1 MNIST (frozen SRq multivariate, train=even/TEST=odd): median lead 10,400 steps = 59.8%
  of the delay, miss 0/35, FA 1/3 — and the 1 is base_c92_s1, a CENSORING-NOISE negative
  (still climbing at budget, crossed 0.80 at 43k); both STRUCTURAL negatives (shufpairs,
  maxP~0.49 vs tau 0.861) correctly silent. K1 NO-FIRE (best scalar single: wnorm 1,000 ->
  SRq +940%); K2 NO-FIRE. Negative taxonomy formalized: budget-censored vs structurally
  blocked.
- R2 MNIST interventional shift: the FA-controlling multivariate COLLAPSES (lead 0, miss
  60%) while the bare content probe survives (cos_gap 4,400 lead / 4.7% miss; d.cos_gap
  16,400 / 0 miss) and the scalar clock collapses (wnorm 0 lead / 51% miss). P3 CONFIRMED
  for content probes. NEW FINDING (the paper's core): FA-control requires context features
  (norm) that are exactly what breaks under intervention -> robustness-vs-false-alarm
  tradeoff in emergence forecasting.
- ALGORITHMIC domain: NO feasible operating point at FA<=5% across single-signal, linear,
  and quadratic multivariate (trap negatives bind: supcon@c35 inversion runs carry COMPLETE
  structure, fourier 0.996 > positives' median-at-t_gen 0.69; band arms fire 0.81-0.91).
  K1/K2 FIRE for this domain as "infeasible" -> forecastability boundary result.

### P5 R2b AMENDMENT (pre-registered NOW, before running): mechanism-factored two-gate
Alarm iff content-probe gate AND norm-viability gate. Viability windows are MECHANISM
CONSTANTS fixed a priori (not fit): alg ||W|| in [40, inf) (c35 inversion below 40, Papers
2-3); mnist ||W|| in (0, 110] (wrong-structure explosion above ~110, Papers 4). Only the
content threshold is fit (train side, FA cap, veto active). Predictions: (i) alg becomes
feasible with positive lead (c35 traps vetoed; bands must fall to the content threshold);
(ii) mnist two-gate retains >=50% of cos_gap's shift lead at lower FA. K8: alg still
infeasible -> the boundary stands even with mechanism factoring; reported as such.

### P5 R3 PROSPECTIVE (pre-registered NOW; runs execute AFTER this commit)
grid5r3, 20 NEW runs: seeds 5-7 x {baseline, supcon_aug lam0.3, supcon_label lam0.3,
supcon_shufpair (structural negatives), base_clamp60 (UNSEEN pin), supcon_aug lam0.05
(UNSEEN dose)} + base_c92 s5,s6 (censoring-noise negatives). Frozen artifacts under test:
(a) R1 SRq forecaster (freeze_eval_p5.py: code+hyperparams+tau protocol, deterministic),
(b) first-pass single-signal thresholds (out5/r1r2_stats.json). Alarm decisions are causal
functions of log prefixes computed by frozen code — post-hoc application is equivalent to
online. Predictions: P-R3a SRq median lead on baseline-family >= 50% of 10,400 (K4 fires
below); P-R3b cos_gap probe on prior arms >= 50% of 4,400; P-R3c SRq misses >= 40% of
prior arms (the tradeoff, prospectively); P-R3d structural negatives: zero alarms from
both frozen artifacts; c92 runs exempt (censoring-noise class).

## PAPER 4 DELIVERABLE (paper3/ dir) — "Structure Transfers, Exponents Do Not" — COMPLETE
9pp, 62 macro-backed numbers (paper3/gen_numbers.py -> numbers.tex; verify_regen.py PASSES),
5 figures from analysis/out4 (analyze_paper4.py over 83 runs: grid4 35 + grid4b 48).
arxiv_upload_paper3.zip clean-room verified (9pp, refs intact, tectonic). arxiv_abstract.txt
1,896 chars (<=1920, single paragraph). Structure: free-norm negative led with -> confound
diagnosis -> prereg protocol verbatim (commits 96f9c0a/45a1d08/c9b856e in Sec 7 Repro) ->
E2 matched-norm win + K2 fired + pin-92 post-hoc robustness notes (0.80-bar + budget-attained
accuracy, labeled post-hoc) -> E3 null + E4 dose -> probes -> boundary-map discussion incl.
wall-clock honesty (2.5-3x/step aux cost) + augce-is-a-regularizer distinction.
House-rule catches during writing: clamp floor is 400--600 not uniform 400 (aug_clamp23_s4);
pin ceiling range 0.91--0.93 not 0.91--0.92. Both hand-typed guesses were wrong; macros right.
User actions before submission: author block confirm; arXiv timing/order vs Papers 1+3;
cs.LG primary + stat.ML cross-list as before; wire real arXiv IDs into companion cross-cites.

### User review response (2026-07-14, all 6 items addressed)
1. UNCENSORED SLOPE REFIT (the one that mattered): 0.80-bar refit, same pins/seeds, zero
   censoring -> slopes 0.058 vs 0.057, RATIO 1.03 (vs 1.17 censored LB). K2's negative is now
   an assertion, not a failure-to-confirm. Added to abstract + Sec 5 + Sec 8; also required
   tempering "norm-growing offset" -> the criterion-level ratio growth traces to the
   budget-attained-accuracy gap at high pins, stated in Sec 5.
2. "four orders of magnitude" -> "more than two orders (400-600 to beyond 100k)" — arithmetic
   slip in un-macro'd prose, exactly the failure mode the reviewer named.
3. paper3/-vs-paper4 codename note added to Reproducibility.
4. Sec 5 heading -> "the prior wins wherever there is delay to win".
5. GitHub push for third-party timestamps: pending repo-scope decision (git root spans multiple
   research projects) — surfaced to user.
6. "dose response is monotone" -> "no dose wins at free norm" (abstract, Sec 6, fig 3 caption,
   fig title). Seed-0 lam0.03 dips trivially below baseline; monotone was wrong as stated.
arxiv_abstract.txt now 1,910 chars; zip rebuilt + clean-room re-verified (9pp).

### Review round 2 (2026-07-14): refit-consistency propagation
The 1.03 parallel-slopes result contradicted two leftover "grows with norm" claims (intro
bullet 3; fig 2 caption asserted growth AND log-parallel constancy in adjacent sentences).
Unified story now everywhere: roughly constant multiplicative offset (slope ratio 1.03
uncensored) + budget-attained accuracy advantage at high pin that inflates strict-criterion
ratios. Abstract "from X to Y" -> "criterion-level ratios X at 50 and Y at 92". Slopes now
printed at 4 decimals (0.0571/0.0489 -> 1.17; 0.0584/0.0568 -> 1.03) so quoted ratios
reproduce from printed values — the 3-decimal prints computed to 1.02/1.16, a reproducibility
gap the user caught. arxiv_abstract.txt 1,902 chars; zip + clean-room re-verified (9pp).

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

Ops note (2026-07-14): two workers briefly raced on 3 seed-2 dirs (worker A owned seeds 0-2, a
redundant seed-2 worker was launched in parallel). No corruption: runs are deterministic, both
writers produced byte-identical records (independently computed identical t_gen), all seed-2
metrics.jsonl parse clean (monotone steps, zero bad lines, correct eval counts). No rerun.
grid4 free-norm grid now COMPLETE at 35/35.

### 2-seed verdicts vs prereg (2026-07-14, recorded BEFORE the seed extension launches)
- K1 NO-FIRE — MATCHED-NORM BENEFIT CONFIRMED. base/aug delay ratio at pinned norm: c35 ~1.0
  (floor), c50 1.44-1.77x, c65 1.53-1.67x, c80 1.86-2.09x, c92 2.08->=2.25x (base_c92_s0 CENSORED
  at 100k budget with max acc 0.841; aug_c92 44.4k/40.4k). P1 confirmed (steep growth + censoring
  at high pin), P2 confirmed (disclosed non-blind).
- K2 FIRES. Ascending-branch (c50->c92) log-delay slopes: baseline ~0.058/unit (lower bound, one
  censored cell) vs supcon_aug ~0.050/unit -> ratio ~1.15-1.25, far below the pre-registered 2.
  NO algorithmic-magnitude exponent flattening on MNIST (was 17x on modular arithmetic). The
  matched-norm benefit is a mild norm-growing multiplicative offset (1.4->2.3x), not a slope
  change. Will be reported exactly as such.
- E3 SPLIT 1-1 at n=2: nn_s0 38.0k vs baseline 33.4k (loses); nn_s3 28.4k vs 29.8k (wins).
  Purity 0.844/0.830 (>=0.8 bar met). Neither P4 nor K3 resolves. Note: nn final norm 114-119 —
  a BIGGER Channel-2 handicap than the view prior's ~92 — yet ~baseline speed at free norm.
- E4 monotone dose at free norm: lam 0.03 break-even (33.0k/31.0k vs 33.4k/29.8k), 0.1 slower
  (36.8k/35.6k), 0.3 slowest (43.6k/41.6k). Norm-freeze at ~90 persists even at lam 0.03 —
  Channel 2 saturates at tiny dose; there is NO free-norm dose at which the view prior wins.
- P5 directional-yes in prior arms (cos_gap 50%-of-max at 0.6-1.0k vs behavior at 15.6-17.6k;
  gap_max 0.15-0.17 vs baseline 0.118) with a stated caveat: 50%-of-max degenerates on baseline's
  low-amplitude curve (crosses at step 0); final analysis will use an absolute threshold.

EXTENSION RULE (stated before launching seeds {1,4}, which the prereg design named): all four
seeds reported regardless of outcome; E3 verdict = per-seed paired tally over 4 seeds (win =
strictly smaller t_gen than same-seed baseline); NO further seeds after {1,4} whatever happens;
E2 extension exists to put n=4 behind the slope fits, not to re-litigate K2.

### FINAL amendment verdicts, n=4 (2026-07-14, grid4b complete: 46 runs)
- E2 CONFIRMED at n=4, every seed, every pinned norm >= 50: base/aug delay ratio c50 1.44-1.96x,
  c65 1.53-2.03x, c80 1.86-2.57x, c92 2.08-2.42x. At c92 baseline CENSORS 3/4 seeds (only s3
  finishes, 84k) while the prior groks 4/4 (40.4-44.4k) -> at high pinned norm the prior is the
  difference between generalizing and not, on natural data. K1 definitively no-fire; c92 ratios
  are conservative lower bounds (censored scored at budget).
- K2 STANDS (fired): n=4 ascending-branch slope ratio ~1.15-1.2 (baseline side a lower bound via
  censoring). No algorithmic-magnitude exponent flattening on MNIST; benefit = multiplicative
  offset growing with norm, NOT a slope change. Scope limit on Paper 3's 17x, stated as such.
- E3 FINAL: 2-2 (per pre-stated tally rule). Wins marginal (s3 1.05x, s4 1.04x), losses real
  (s0 0.88x, s1 0.52x — 72.4k vs 37.6k). Purity 0.812-0.844 (bar met on all seeds). VERDICT:
  no evidence of free-norm transfer for the cross-example label-free prior; heterogeneous harm
  on one seed echoes the Paper-1 norm-channel bimodality. nn final norms 114-119 (worst
  Channel-2 handicap of any prior arm). Rule honored: no further seeds.
- Amendment scorecard: P1 Y, P2 Y (disclosed non-blind), P3 N (K2 fired), P4 N (E3 2-2), P5 Y
  (directional). Free-norm negative (view prior slower 5/5) reported in full per D2.
- Paper 4 story locked by the data: (1) free-norm race = channel confound that makes good priors
  look bad; (2) structure channel TRANSFERS (matched-norm 1.4-2.4x growing with norm; 3/4
  baseline censoring at c92 vs 4/4 prior groks); (3) exponent-flattening magnitude does NOT
  transfer (~1.2x vs 17x); (4) no free-norm dose wins, Channel-2 saturates by lam 0.03; (5)
  label prior 2.1x 5/5, wrong-structure ceiling-blocks 5/5, augce/clamp abolish the delay.

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
