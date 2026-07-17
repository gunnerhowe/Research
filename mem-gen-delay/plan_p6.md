# P6 PLAN + PRE-REGISTRATION: Capability-Emergence Forecasting on Language Models
## From grokking model-systems (P5) to the benchmark a frontier lab would actually use

Committed BEFORE any P6 work executes. Release policy (user directive 2026-07-16): NOTHING
ships until the combined result clears the community bar below. P5 is documented in
P5_RESULTS.md and stays internal until then.

## The community bar (what "done" means)
A releasable P6 = the FIRST forecasting benchmark for capability emergence on real language
models with ALL of: (1) actual LMs (public checkpoint suites + our own seed fleets), (2)
manufactured capability-blocked NEGATIVES enabling honest false-alarm accounting (the P5
lesson: negative-set quality is the binding resource; no public suite has negatives), (3)
per-seed timing forecasts that beat both loss-extrapolation AND the config-equation
baseline (arXiv:2511.16893 predicts a single time per config; seed variance is its blind
spot), (4) trajectory/conformal forecasts with coverage guarantees instead of fragile
single thresholds (the R5b lesson), (5) a blind prospective rung on a never-seen config,
pre-registered, commit-stamped. If any leg fails its kill, we keep working; we do not ship
partial.

## Workstreams -> rungs

R0 FEASIBILITY GATE (public checkpoints, ~no cost): probe EleutherAI Pythia checkpoint
   suites (70m, 160m first; 410m/1b if disk allows) on the 3080. Per checkpoint compute:
   behavioral copy advantage (2nd-half vs 1st-half CE on repeated random sequences â€” the
   induction bump), internal prefix-matching head score (attention to prev-occurrence+1),
   previous-token head score per layer (the known mechanistic PRECURSOR), and LM loss on a
   fixed local text batch. GATE: the IH phase change and its precursor must be visible in
   our trajectories (they are in the literature; this validates OUR instruments). Kill
   P6-K0: phase change not resolvable at Pythia's checkpoint granularity -> pivot entirely
   to own-fleet training (R2 absorbs R1).
R1 PYTHIA RETRODICTIVE CASE STUDY (after R0 curves; analysis spec pre-registered before
   any forecast is computed): across available suite runs (sizes x dedup variants, n~6-10),
   event = copy-advantage crossing 50% of its own asymptote (per-run relative criterion);
   question = do internal precursors (prev-token-head strength, prefix-score growth)
   anticipate the crossing earlier than loss-curvature extrapolation? Honest framing:
   case study (tiny n, no negatives in public suites), lead measured in tokens at
   checkpoint granularity. Kill P6-K1: precursors no earlier than loss on >= half the runs.
R2 SEED FLEET (the benchmark core; ~1-2 GPU-days): train n=30 small transformers (2-layer,
   d~256, ctx 256, bigram-rich data) with DENSE checkpointing/logging through the IH phase
   change; event = per-seed IH formation time (behavioral crossing; internal score as
   secondary). Baselines: config equation (constant per config), loss extrapolation.
   Forecast target: per-seed timing RESIDUALS from early internal precursors. Kill P6-K2:
   Spearman(precursor@t_early, formation time) CI includes 0 at n=30 for all probe times.
R3 MANUFACTURED NEGATIVES (the methodological headline; ~1 GPU-day): capability-blocked
   runs with identical logging: (a) data ablation â€” within-context repeated-bigram
   stripping (removes the IH training signal); (b) architectural â€” 1-layer models (IH
   structurally impossible); target n >= 10 negatives. Institutionalized P5 lesson: FA caps
   are only valid with >= 5 negatives per fit/eval side (hard rule, in the spec). Then the
   full FA-disciplined benchmark on fleet+negatives: lead @ FA <= 5%, train/test by seed.
   Kill P6-K3: FA uncontrollable even with manufactured negatives.
R4 TRAJECTORY/CONFORMAL FORECASTS (CPU, on fleet data): split-conformal time-to-event
   intervals over seeds from trajectory features; score coverage, width, lead. Replaces
   single thresholds. Kill P6-K4: intervals vacuous (width ~ budget) at nominal coverage.
R5 BLIND PROSPECTIVE (the ship-gate): freeze everything from R2-R4; pre-register point +
   interval forecasts for a NEVER-SEEN config (different width/data mix, 10 fresh seeds +
   3 fresh negatives); run; score. Kill P6-K5 = K4-style prospective collapse.

## R0 VERDICT (2026-07-16): K0 PASSES [runs/p6r0/*.jsonl, analysis/out6/fig_r0_pythia.*]
Phase change crisply resolvable in BOTH sizes: copy_adv 0.02-0.03 at 1.07B tokens -> 9.7-9.9
at 2.1B; induction score 0.02-0.03 -> 0.93-0.96 in the same window. THE PRECURSOR LEADS:
early-layer prev-token score is 4x baseline at 0.54B tokens (0.13-0.16) and 10x at 1.07B
(0.37) while induction + behavior are still at noise â€” the mechanistic antecedent is
measurable one full public-checkpoint stage before the capability, in models we did not
train. Bonus: pythia-70m partially LOSES induction late (0.97 -> 0.36 at 67B, partial
recovery); 160m keeps + grows deeper-layer IHs â€” capability regression exists and the
fleet (R2) should quantify it. Caveat: the public grid has NO checkpoint between step512
(1.07B) and step1000 (2.1B); the cliff sits inside that gap -> public-suite lead is
bounded at one checkpoint stage; dense granularity is the fleet's job.

## R1 SPEC (frozen NOW, before any forecast is computed; constants a priori)
Runs: pythia {70m, 160m, 410m} x {standard, deduped} as available (n~5-6). Event = first
checkpoint with copy_adv >= 50% of the run's max. PRECURSOR RULE (no tuning): alarm at
first checkpoint with early-layer prevtok >= 0.10 (~3x the step-0 baseline of 0.037).
LOSS RULES (best-case sweep, granted to the baseline): alarm at first checkpoint with
text_loss <= theta, theta swept over all values; also loss-slope variants. Scoring per
run: lead (in checkpoints and tokens) = event - alarm; a rule is VALID on a run if it
alarms strictly before the event. P6-P1: the precursor rule alarms exactly one stage
before the event on >= 4/5 runs with zero post-event alarms; no loss threshold achieves
uniform pre-event alarming across runs without firing at/before 0.27B tokens (vacuously
early, >= 4 stages ahead of any event) â€” i.e., loss carries no event-specific timing.
K1 as in plan: precursor no earlier than best loss rule on >= half the runs.

## R1 VERDICT (2026-07-16) [analysis/out6/r1_scored.json]
AS-REGISTERED: P6-P1 FAILS on both clauses, and the failures are informative. (1) The
precursor's lead is 2 stages, not the predicted "exactly one," on 3/5 runs (alarm at step
256 vs cliff at 1000). (2) The loss baseline is STRONGER than predicted at public
granularity: theta=6.264 alarms uniformly pre-event (1-stage lead on 4/4 clean runs) â€”
loss does carry one stage of timing here. K1 NO-FIRE: the precursor alarmed pre-event on
5/5 runs with ZERO post-event alarms and led the best-case loss rule strictly on 4/5
(tied 1/5). Net: precursor buys one EXTRA stage over best-case loss; my registered
prediction underestimated the baseline and overestimated precision.
EVENT-DEFINITION BUG (disclosed): 50%-of-run-max is fragile to late outliers â€” the
160m-deduped "event at 16.8B tokens" was an artifact of an anomalous final-checkpoint
copy_adv spike (24.3 vs ~12 plateau); its actual cliff is at step 1000 like every other
run. Same fragility class P5 flagged for 50%-of-max thresholds; lesson institutionalized:
R2+ event = ABSOLUTE copy_adv crossing (>= 2 nats: far from noise ~0 and plateau ~10-12),
frozen now.
KEY STRUCTURAL FINDING for the program: all 5 public runs (3 sizes x 2 data variants,
shared batch/context) cliff at the SAME ~2.1B tokens with the precursor up at 0.5-1.1B â€”
fully consistent with config-determined timing (arXiv:2511.16893). Public suites therefore
CANNOT distinguish precursor-forecasting from config-lookup (n=1 per config). The seed
fleet (R2) is where that question is answerable â€” per-seed variance at fixed config â€” and
is now unambiguously the critical rung.

## R2/R3 PRE-REGISTRATION (frozen BEFORE fleet launch; grid6r2 empty at commit)
Trainer src/train_lm.py: 2-layer TinyLM d256/h4/ctx256, vocab 2048, fixed bigram language
(LANG_SEED 777), variable-offset repeats p_rep=0.75 (positives); lr 1e-3 (warmup 100),
batch 64, budget 16,000 steps, eval every 25. Smoke (seed 0, 12k): event 6,175; precursor
lead ~1.5-2k steps; graded in-distribution ramp from ~2.4k. SEED 0 IS THE SMOKE AND IS
EXCLUDED from all confirmatory sets.
Fleet: rep seeds 1-30 (30 positives) + norep seeds 1-10 (data-ablation negatives) +
onelayer seeds 1-5 (architectural negatives). Event (frozen from R1 lesson): copy_adv >=
2.0 nats on 2 consecutive evals. Precursor time t_pv = first eval with layer-0 prevtok >=
0.10 (constant carried from R1). Graded-behavioral time t_ind = first eval with
indist_adv >= 0.10 (a priori).
Predictions:
- P2a: per-seed event spread >= 1.3x (max/min) across the 30 positives (else timing is
  config-determined even per-seed â€” reported as such).
- P2b: Spearman(t_pv, t_event) >= 0.5, bootstrap 95% CI excluding 0. K2 fires if CI
  includes 0 for ALL probe-derived times (t_pv, t_ind).
- P2c: best probe-time Spearman > best-case loss-threshold Spearman (theta swept, granted).
- P2d: prevtok FORMS in norep negatives (independently useful for Markov prediction) ->
  bare precursor rule false-alarms there; the a-priori CONJUNCTION (prevtok >= 0.10 AND
  indist_adv >= 0.10) achieves FA <= 1/10 on norep while retaining pre-event alarms on
  >= 25/30 positives.
- P2e: onelayer: zero events, prefix < 0.05 throughout; conjunctive rule alarms <= 1/5.
FA discipline: min-5-negatives rule satisfied per class (10 + 5).

## R2/R3 VERDICT (2026-07-17) [analysis/out6/r2_scored.json; one-shot sealed scoring]
- P2a PASS (barely): per-seed event spread 1.3097x (5,650-7,400) vs bar 1.3. Timing
  variance at fixed config is real but modest (~+-13%).
- P2b PASS, SPECTACULARLY: Spearman(t_pv, t_event) = 0.977, bootstrap CI [0.911, 0.995],
  n=30. The precursor's formation time forecasts each seed's emergence time almost
  perfectly, with median lead 975 steps (825-1,125) ~ 15% of training. THE CORE CLAIM OF
  THE PROGRAM, on LMs, at seed level, where the config equation predicts a constant.
- P2c FAIL AS REGISTERED + spec flaw disclosed: a best-case loss threshold (theta=2.077)
  ties the precursor's Spearman EXACTLY (0.9769) â€” but post-hoc lead analysis (labeled)
  shows why: its median lead is 50 steps (min -25) â€” it detects the cliff DURING descent.
  Equal ranking, 20x less warning. The registered metric was one-dimensional (rank corr
  without lead); rule for all future specs: forecasting comparisons are (correlation,
  lead) pairs. Also: t_ind (graded behavioral ramp) Spearman ~ 0 â€” behavioral early
  signals do NOT time the transition; the circuit precursor does. Clean dissociation.
- P2d PASS on its criteria (conjunction FA 0/10, pre-event 30/30) BUT the interior
  mechanistic prediction was WRONG and is reported so: prevtok does NOT form in norep
  negatives (max 0.036-0.061 < 0.10; bare-precursor FA 0/10). My reasoning failed because
  a pure bigram language is solvable by the embedding pathway alone â€” prev-token attention
  buys nothing without repetition. The manufactured negatives thus CERTIFIED the precursor
  rather than trapping it (certification is also what negative sets are for). Harder
  negative class for the trap (a language where prev-token context is independently
  needed, e.g. trigram/skip structure) = R5-adjacent follow-up.
- P2e PASS: onelayer 0/5 events, prefix <= 0.014 (structurally blocked confirmed),
  conjunction 0/5; descriptive: prevtok also stays ~0.04 in 1L â€” at this scale the
  precursor forms as part of 2L circuit assembly, not independently.
- K2 NO-FIRE (decisively). Fleet: 30/30 positives evented, 15/15 negatives silent.
HEADLINE: per-seed emergence forecasting from a mechanistic precursor â€” rho 0.98, ~1,000
steps (~15%) of lead, 0/15 false alarms on manufactured negatives, vs a config-equation
baseline that is constant-per-config and a loss baseline with 50 steps of lead.

## R4 SPEC (frozen before scoring; fleet outcomes already seen via R2 â€” disclosed; the
## blind validation of everything R4 produces is R5)
Split-conformal time-to-event intervals, anchored at the precursor alarm (t_pv, the
frozen prevtok >= 0.10 rule). Calibration = rep seeds 1-15; test = rep seeds 16-30
(a-priori split by seed number). Methods: (A) OFFSET â€” point forecast t_pv + median
calibration lead, nonconformity = absolute residual; (B) LINEAR â€” t_event ~ a + b*t_pv
fit on calibration, same conformal wrapper. Nominal coverage 90% (finite-sample
corrected quantile). Report per method: test coverage, median interval width, anchor
lead. Loss-anchored variant (theta = 2.077) reported for contrast â€” its anchor arrives
~50 steps pre-event, so whatever its width, it forecasts nothing. K4 (from plan):
intervals vacuous (width on the order of the 16,000-step budget) at nominal coverage.

## R5 PRE-REGISTRATION â€” THE BLIND SHIP-GATE (committed BEFORE grid6r5 exists)
FROZEN FORECASTER (verbatim, calibrated on grid6r2 seeds 1-15, untouched hereafter):
alarm at first eval with layer-0 prevtok >= 0.10; forecast the emergence interval
[t_alarm + 825, t_alarm + 1125] (offset 975 +- conformal q 150, nominal 90%).
Negative rule: conjunction (prevtok >= 0.10 AND indist_adv >= 0.10). Event rule:
copy_adv >= 2.0 on 2 consecutive evals.
NEVER-SEEN CONFIG (double shift, chosen a priori): d_model 256 -> 320 (architecture) and
p_rep 0.75 -> 0.6 (data mix); everything else unchanged (same language LANG_SEED 777,
lr 1e-3, ctx 256, vocab 2048, batch 64). Budget 20,000 steps (instrumentation headroom,
not calibration). Runs: rep seeds 101-110 (10 positives) + norep seeds 101-103 (3
negatives), grid6r5.
Predictions:
- P5a: >= 9/10 positives event within budget; 0/3 negatives event.
- P5b: precursor alarms strictly pre-event on >= 9/10 positives; conjunction FA 0/3.
- P5c (THE GATE): the frozen interval covers the true event time on >= 7/10 positives.
- Secondary (scoped-transfer reading if P5c fails): Spearman(t_pv, t_event) >= 0.5 on the
  10 new seeds would mean the ANCHOR transfers and only the offset is config-local.
- K5-gate: coverage <= 5/10 OR precursor pre-event on <= 7/10 -> frozen calibration does
  not survive the config shift (the P5-K5 precedent realized on LMs); iterate
  (config-conditional calibration) before any release. Honest expectation stated now:
  the offset delta is the component most at risk under shift.

## R5 VERDICT (2026-07-16) â€” THE BLIND SHIP-GATE PASSES, 10/10 ON EVERY CLAUSE
[analysis/out6/r5_scored.json; forecaster frozen at 377511b before grid6r5 existed]
- P5a PASS: 10/10 never-seen-config positives evented; 0/3 negatives.
- P5b PASS: precursor alarmed strictly pre-event 10/10; conjunction FA 0/3.
- P5c PASS AT CEILING: the frozen interval [t_alarm+825, t_alarm+1125] â€” calibrated on a
  DIFFERENT architecture (d256) and data mix (p_rep 0.75) â€” covered the true emergence
  time on 10/10 blind runs (bar 7/10; nominal 90%). Median lead 1,012 steps. Secondary
  Spearman on the new config: 0.988. K5-gate NO-FIRE.
- My own registered expectation ("the offset delta is most at risk under shift") was
  WRONG in the favorable direction: the precursor-to-cliff gap is ~config-invariant
  across this shift (975 -> 1,012) â€” itself a finding worth probing (is the gap set by
  optimization dynamics rather than config?).
COMMUNITY BAR: all five legs now stand. (1) real LMs: Pythia public suites + own fleets;
(2) manufactured negatives with FA accounting: 18 negatives, 0 false alarms; (3) per-seed
forecasts beat loss (50-step nowcast) and config-equation (constant): rho 0.98 at ~1,000-
step lead; (4) conformal intervals with measured coverage: 15/15 in-config, 10/10 under
config shift; (5) blind prospective rung, pre-registered, commit-stamped. EXPERIMENTS
COMPLETE; release decision is the user's per D2.

## R6/R7/R8 PRE-REGISTRATION â€” the three strengtheners (committed before their grids exist)

R7 THIRD CONFIG AXIS (a NEW LANGUAGE; frozen rule verbatim, R5 protocol): lang_seed
777 -> 888 (a different bigram table = a different data-generating process), all else at
the ORIGINAL fleet config (d256, p_rep 0.75, lr 1e-3); budget 20,000. Runs: rep seeds
201-210 + norep seeds 201-203 (grid6r7). Frozen rule unchanged: alarm at layer-0
prevtok >= 0.10; interval [t_alarm + 825, t_alarm + 1125]. Bars mirror R5: P7a >= 9/10
events & 0/3 negative events; P7b pre-event alarms >= 9/10 & conjunction FA 0/3; P7c
interval coverage >= 7/10; secondary Spearman >= 0.5. K7-gate: coverage <= 5/10 or
pre-event <= 7/10 -> the gap constant is language-local; report and calibrate per-language.

R8 GAP-ORIGIN PROBE (competing clocks, 20 runs, grid6r8; bigram lang 777, d256, p_rep
0.75, budget 32,000): lr in {5e-4, 2e-3} x seeds 301-305 at batch 64; batch in {32, 128}
x seeds 311-315 at lr 1e-3. Reference cell = the existing 30-seed fleet (lr 1e-3, batch
64, median gap 975). Gap = t_event - t_pv per run. Hypotheses: H1 (optimization clock)
gap_steps ~ 1/lr, insensitive to batch; H2 (token clock) gap_steps ~ 1/batch,
insensitive to lr. Discrimination rule (frozen): an axis "controls the gap" if its
low/high median-gap ratio >= 2 while the other axis' ratio <= 1.5 (lr ratio uses
5e-4 vs 2e-3, predicted 4x under H1; batch ratio uses 32 vs 128, predicted 4x under H2).
K8b: both ratios < 2, or both >= 2 -> no clean single clock; gap origin reported as
open/mixed.

R6 TRAP LANGUAGE (constants pending the running trigram smoke; predictions frozen NOW):
trigram language (next ~ table[hash(prev, current)]) makes previous-token context pay
for the TASK, independent of repetition -> the precursor should form even where the
capability cannot. Runs (after smoke sets budget): trigram rep seeds 1-10 + trigram
norep seeds 1-10 (grid6r6). Predictions: P-T1 (the trap is real): bare precursor rule
(prevtok >= 0.10) false-alarms on >= 8/10 trigram norep negatives. P-T2: the a-priori
conjunction (prevtok >= 0.10 AND indist_adv >= 0.10) restores FA <= 1/10. P-T3
(exploratory, all (rho, lead) pairs reported): among anchors {t_pv, t_ind, t_conj,
t_prefix (first prefix >= 0.05)}, at least one achieves Spearman >= 0.5 with median
lead >= 300 steps AND FA <= 1/10 on the trigram negatives. K-T: none does -> in trap
languages, forecasting needs signals beyond this battery; reported as the boundary.
Expectation stated now: t_pv itself should DEGRADE as a timer here (it forms early for
task reasons on every run) â€” the trap is expected to break the bare precursor and the
question is what survives.

## Disclosures
D1 R0/R1 events on public suites are defined after seeing R0 curves (relative criteria
   chosen to minimize arbitrariness); the blind rung is R5, as in P5.
D2 All kills reported internally; the release decision is the user's, gated on the bar.
D3 Downloads: public model checkpoints from huggingface.co (EleutherAI/pythia-*), ~150-350
   MB per checkpoint, ~3-7 GB per size for ~20 checkpoints; HF cache pinned to the E: drive.
D4 P5's min-negative-count rule applies to every FA number in P6.

### R6 constants FINALIZED after smoke iterations (v3; seed 0 = smoke, excluded)
lang=trigram TRI_MODES=2, budget 24,000, lr 1e-3, d256, p_rep 0.75 (positives) / 0.0
(negatives). Smoke v3: precursor crosses 0.10 at ~1k FOR TASK REASONS (saturates 0.99 by
4k) while the event lands at 10,100 — the trap is armed: on norep-trigram negatives the
identical task incentive should fire the bare rule (P-T1). Fleet: grid6r6, trigram rep
seeds 1-10 + trigram norep seeds 1-10. Design iterations v1 (unlearnable hash — no trap,
all-at-once assembly) and v2 (modes=8 — trap arms but induction starved beyond budget)
disclosed in src/train_lm.py and feed the competition-clock hypothesis (R8 notes).

### Ops note (2026-07-16): R7/R8 worker died at run 12/33 — GPU contention, no data loss
Two workers (R6 trap fleet + R7/R8) were launched concurrently on one 10GB RTX 3080. The
R6 process's allocator cache grew to ~8.6GB; the R7/R8 worker then failed to allocate for
its next run and exited (exit 1; traceback lost to the log redirect). NO corruption and no
scoring impact: the 11 completed R7 runs (10/10 positives + 1 negative) have intact
summaries; the partial norep_s202 had only metrics.jsonl (no summary.json) so the
idempotent runner re-runs it cleanly from scratch. Remediation: R7/R8 is now CHAINED to
start after R6 completes rather than racing it. Lesson (institutionalized): one training
worker per GPU; concurrency only across distinct devices.

## R7 VERDICT (2026-07-16) — THIRD AXIS PASSES [analysis/out6/r7_scored.json]
Frozen rule (calibrated on language 777, commit 377511b) applied to an ENTIRELY NEW
LANGUAGE (lang_seed 888), scored once:
- P7a PASS: 10/10 positives evented; 0/3 negatives evented.
- P7b PASS: 10/10 pre-event alarms; conjunction FA 0/3; BARE precursor FA also 0/3.
- P7c PASS: 9/10 interval coverage (bar 7/10) — EXACTLY the 90% nominal the conformal
  quantile was calibrated for, on a data-generating process it never saw. The single miss
  (seed 206, lead 1200) fell 75 steps past the interval's upper edge, i.e. late-side, not
  a structural failure.
- Secondary Spearman 0.9515; median lead 987.5. K7-gate NO-FIRE.
GAP CONSTANT NOW STABLE ACROSS THREE CONFIGS: 975 (config A: d256/p_rep .75/lang 777),
1,012 (config B: d320/p_rep .6/lang 777 — arch + data mix shifted), 987.5 (config C:
d256/p_rep .75/lang 888 — language replaced). Spread across all three: ~4%. The
precursor->emergence gap survives architecture, data density, AND the data-generating
process itself. This sharpens R8 (gap origin) from a curiosity into the program's central
open question: a quantity that invariant is being set by something, and it is not config.

## R6 VERDICT (2026-07-17) — THE TRAP IS REAL, AND THE FIX HOLDS [analysis/out6/r6_scored.json]
- P-T1 PASS AT CEILING: in the modal-trigram language the bare precursor rule false-alarms
  on 10/10 capability-blocked negatives (prevtok crosses 0.10 at 650-4,400 steps from the
  TASK incentive alone). The bigram-fleet headline rule is definitively broken in trap
  languages — predicted from mechanism at prereg, before any trigram fleet run existed.
- P-T2 PASS AT CEILING: the a-priori conjunction (prevtok >= 0.10 AND indist_adv >= 0.10,
  same eval) restores discrimination completely: 0/10 FA.
- P-T3 PASS with the campaign's most striking number: the conjunction anchor t_conj scores
  Spearman rho = 1.000 (PERFECT rank order, n=10) with median lead 1,900 steps (range
  1,525-2,100) at 0/10 FA. Secondary anchor t_prefix (induction-score ramp >= 0.05):
  rho = 0.988, median lead 550, 0/10 FA. Neither gate alone survives: bare precursor
  10/10 FA; behavioral ramp alone 2/10 FA (early transients on 2 seeds — which the
  same-eval conjunction inherently filters). K-T NO-FIRE.
- INTERPRETATION: in trap languages the forecasting hierarchy reorganizes exactly as the
  mechanism-factored view predicts. The precursor alone degenerates into a task-signature;
  the conjunction becomes a BETTER per-seed timer than the bare precursor ever was in
  bigram languages (rho 1.000 vs 0.977). This is the P5 two-gate lesson reproduced from
  first principles in a new domain, with the failure mode predicted, demonstrated, and
  repaired inside one pre-registered rung. The bigram fleet's P2d "wrong prediction" is
  retroactively contextualized: the trap I wrongly predicted there exists — in the
  language class actually built to contain it.
- PAPER IMPACT: the negatives claim upgrades from "0 FA everywhere" to the stronger,
  honest form: "bare precursor certified in bigram languages; broken (10/10 FA) in trap
  languages; mechanism-factored conjunction certified in BOTH (0 FA) with rho 1.000."

## R8 VERDICT (2026-07-17) — K8b FIRES AS REGISTERED; post-hoc reveals the unifying law
[analysis/out6/r8_scored.json, r8_posthoc_proportionality.txt]
AS-REGISTERED: no single external clock. lr ratio 2.27, batch ratio 2.35 — BOTH axes move
the gap ~2.3x, so neither passes the frozen owns-it rule (one >= 2, other <= 1.5). H1 and
H2 both rejected; K8b fires.
POST-HOC (labeled): the gap is a FRACTION OF TIME-TO-EMERGENCE, not a fixed interval.
Across ALL 80 valid-anchor runs (fleet A, gates B/C, all four R8 cells, trap fleet via
t_conj): anchor/event median 0.843, IQR [0.825, 0.865], range [0.739, 0.898]. One law —
t_event ~= 1.19 x t_anchor — explains the whole campaign: R5/R7's fixed-offset transfer
(those shifts preserved event timescale ~6.3-6.9k), R8's would-have-broken offsets (b128
gap 650, lr5e-4 gap 1,875), and K8b itself (both axes move t_event, and the announcement
scales with the road). Consistent with the competition-clock reading: precursor and
capability phases stretch together.
PAPER IMPACT: the deployable forecaster form is MULTIPLICATIVE (t_hat = 1.19 x t_alarm),
not additive; fixed-step offsets are timescale-local. The R5/R7 gates stand as registered
(their configs preserved timescale) with this scope stated.

## R9 PRE-REGISTRATION — proportional-rule confirmation (grid6r9 empty at commit)
FROZEN RULE (calibrated on the 80 retrodictive runs above; frozen verbatim now):
predict t_event in [1.11 x t_pv, 1.36 x t_pv] (full retrodictive envelope; median
multiplier 1.186 reported as point forecast). Test: a NEVER-SEEN axis value lr = 7e-4
(batch 64, bigram lang 777, d256, p_rep 0.75, budget 24,000), rep seeds 321-325 + norep
seeds 321-322 (hygiene). Bars: P9a >= 4/5 positives event with envelope coverage >= 4/5;
P9b 0/2 negative events and 0/2 conjunction FA. K9b: coverage <= 3/5 -> the proportional
law does not survive its first blind test; reported, and the paper ships the law as
retrodictive-only.

## R9 VERDICT (2026-07-17) — THE PROPORTIONAL LAW SURVIVES BLIND [analysis/out6/r9_scored.json]
P9a PASS AT CEILING: 5/5 events, 5/5 envelope coverage at the never-seen lr 7e-4.
Observed multipliers 1.160-1.226 vs retrodictive median 1.186. P9b PASS: 0/2 negative
events, 0/2 conjunction FA. K9b NO-FIRE.
THE CAMPAIGN IS CLOSED (R6 trap: real + conjunction rho 1.0; R7 third config: nominal
coverage; R8: K8b as registered + the fractional-gap law; R9: law blind-validated).
The paper's forecaster story is now complete and earned: fixed offset -> broken by its
own stress tests (R8 cells) -> replaced by the scale-invariant law t_event ~= 1.19 x
t_anchor -> validated blind at an unseen config (5/5). 87 total valid-anchor runs support
the law; anchors must be mechanism-composed in trap languages (R6). Fold-in begins.
