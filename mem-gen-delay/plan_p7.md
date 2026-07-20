# P7 — Causal control of emergence timing (prereg BEFORE any scaffold code exists)

Started 2026-07-18. Question: is the layer-0 prev-token precursor the RATE-LIMITING
CLOCK for induction emergence, or a passenger on a deeper shared clock? Every P5/P6
result so far is observational; the gap law (anchor at 0.843 of time-to-event, paper4)
is consistent with both readings. P7 intervenes.

## Lit position (kill-test done first; see also analysis/litcheck_esn.md)
- Singh et al., "What needs to go right for an induction head?" (arXiv:2404.07129):
  activation CLAMPING during training in a 2-layer attention-only model on synthetic
  Omniglot ICL. Establishes qualitatively that supplying/removing subcircuits changes
  the transition (one run: phase change ~7.5e4 vs 2e5 iters; knockouts stall). Single
  initialization, no seed statistics, clamp-from-start only, no placebo control, no
  quantitative timing law. THE foundation cite; our rung is its quantitative,
  controlled, law-testing successor on an LM task with an ARCHITECTURAL lever
  (attention-bias primitive, deployable like H3/short-conv) rather than activation
  surgery.
- "Patterning: The Dual of Interpretability" (arXiv:2601.13548): steers induction-
  circuit formation timing via DATA reweighting (susceptibilities). The data-side dual
  of our architecture-side lever. Cite.
- MIDAS/stacking (arXiv:2409.19044): initialization-carried structure speeds training
  + reasoning bias at scale. Circumstantial support; no circuit-timing measurement.
- Our unique assets: the multiplicative gap law (t_event ~ 1.19 x t_anchor) as a
  QUANTITATIVE causal bet; certified event rule + fleet statistics; manufactured
  negatives; specificity + data-necessity controls (absent in all of the above).

## Design (single rung R1; grid7c/)
Substrate: bigram language, standard config (lr 1e-3, d256, 4 heads, 16k steps) —
identical to grid6r2, whose 30 rep runs are the CONTROL distribution:
**T0 = 6,300 median t_event (n=30, IQR 6,206-6,662, range 5,650-7,400).**
Controls are reused, not rerun; validity guard = bit-identity check (see K-C2b).

Intervention: additive attention-score bias B on LAYER 0, HEAD 0 only, added pre-mask:
- hard:  B[i, i-1] = +8 (i>=1), fixed buffer (non-trainable) — the supplied primitive.
- seed:  same B at init but nn.Parameter (trainable, EXCLUDED from weight decay) —
  can be unlearned; tests stickiness.
- sink:  B[i, 0] = +8, fixed — PLACEBO: an equally opinionated, task-useless prior
  (controls for generic effects of low-entropy attention at init).
- near:  B[i, i-2] = +8, fixed — near-miss primitive (EXPLORATORY, no bet).
Arms: hard_s1..10, seed_s1..10, sink_s1..5 (rep condition); near_s1..5 (rep,
exploratory); norephard_s1..5 (norep condition + hard) — data-necessity corner.
35 runs total. Event rule unchanged: copy_adv >= 2.0 nats, sustain 2, completing eval.

## Predictions and kills (frozen now)
- P-C1 (clock): hard median t_event <= 0.5*T0 = 3,150.
- P-C1s (STRONG, law-causal): hard median within [0.08, 0.31]*T0 = [504, 1,953] —
  the gap-law residual (0.157*T0 ~ 989) within factor 2. If P-C1s holds, the post-
  anchor phase is an invariant assembly time and the law is causal, not correlational:
  emergence can be SCHEDULED by scaffold supply.
- P-C2 (passenger boundary): hard median >= 0.8*T0 = 5,040 -> precursor NOT rate-
  limiting; shared-clock/competition reading wins. Between 3,150 and 5,040 = partial
  contribution, reported as measured.
- P-C3 / K-C1 (specificity): sink median must be >= 0.8*T0. If sink <= 0.5*T0,
  K-C1 FIRES: speedup is a nonspecific optimization artifact; no clock claim
  regardless of P-C1.
- P-C4 (stickiness, weak bet): seed median in [hard median, T0]; additionally report
  whether the seeded head's prevtok0 decays below 0.5 before event on >=3/10 runs
  (window signal for R2).
- P-C5 / K-C3 (data necessity): norephard events 0/5. If >=2/5 event, K-C3 FIRES:
  scaffold alone manufactures the capability without repetition pressure —
  contradicts the trap/R-ORD reading; everything stops for re-examination.
- K-C2 (manipulation check, not an outcome): hard arm must show prevtok0 >= 0.8 at
  the FIRST eval (step 25) on every run; failure = implementation void (fix and
  relaunch permitted BEFORE any outcome scoring; outcomes never consulted).
- K-C2b (control validity): scaffold=none code path must reproduce grid6r2/rep_s1's
  first 4 metric rows bit-identically in a 100-step smoke; failure = fix before fleet.
- Scoring: sealed one-shot analysis/score_p7c.py after ALL 35 summaries exist; writes
  analysis/out7/p7c_scored.json; refuses on partial fleet or existing scorefile.
- Disclosure: near arm exploratory; seed-arm wd exclusion disclosed; controls reused
  from grid6r2 (same code path, bit-identity-guarded).

## R2 (conditional, NOT part of this prereg's bets): if P-C1 holds, map the receptive
window: hard bias switched ON at t_insert in {0, 1k, 2k, 4k} x 5 seeds; measure
t_event(t_insert) against the law. Prereg to be written separately after R1 verdict.

## P7 R1 VERDICT (2026-07-20) [analysis/out7/p7c_scored.json, sealed; prereg b8eb219]
Adjudicated with an 11-agent workflow (4 analyses + 4 adversarial verifications + 3
red-team lenses). ALL mechanism analysis below is POST-HOC unless marked pre-registered.

### 1. PRE-REGISTERED OUTCOMES (sealed one-shot, unchanged)
P-C1 (clock, hard <= 3150) FAILED: 3600. P-C1s (law band [504,1953]) FAILED decisively.
P-C2 (passenger, >=5040) no-fire. P-C3 specificity PASS. P-C4 PASS. P-C5 PASS.
K-C1 / K-C2 / K-C2b / K-C3 all clean. Prereg-designated zone: "partial contribution".

### 2. THE EFFECT IS REAL AND SURVIVED EVERY ARTIFACT ATTACK
6,300 -> 3,600 median (42.9%; 31.4-44.1% across 9 alternative capability definitions, all
with complete separation). Exact one-sided permutation p = 1.18e-9 (= 1/C(40,10), the
floor at these n); seed-matched paired p = 9.8e-4, 10/10 faster. Speedup 1.75x, bootstrap
CI [1.73, 1.83] (grid-locked to the 25-step eval grid).
Attacks that FAILED to break it: (a) generic optimization speedup - train_loss curves
superimpose to <= 0.010 nats through step 3000 and hard events at a WORSE loss (2.251 vs
2.176), killing any loss-threshold reading; (b) softmax saturation / attention entropy /
gradient scale / dead-head - sink and near match hard's biased-key softmax weight to 3
decimals and are equally non-trainable; (c) pre-mask addition - provably a no-op (zero
nonzero entries above the diagonal); (d) RNG contamination - step-0 loss identical to 5 dp,
scaffold consumes no RNG; (e) event-metric inflation - pre-emergence copy_adv floor
identical (0.0118 vs 0.0120, p=0.628); (f) transition-shape change - the 0.5->2.0 ramp is
125 steps in EVERY arm, so the transition is a pure time translation.
CAPABILITY VALIDITY: at its event the hard arm has HIGHER layer-1 prefix-matching than
controls (0.629 vs 0.595, p=0.0021), and re-scoring the event on the IN-DISTRIBUTION probe
(different offset, r=96) gives 42.7% - no positional-copy shortcut.
CONTROL REUSE (flagged as the largest unaudited dependency): re-verified deeper this
session - 40/40 eval records bit-identical to grid6r2/rep_s1 over 1,000 steps.

### 3. WHY P-C1s FAILED - THE TWO-GATE MECHANISM (post-hoc, quantitative)
The frozen composed anchor is t_conj = max(t_pv, t_ind), verified 60/60 runs.
  CONTROLS: t_ind 2850 (slack), t_pv 5325 (BINDS), t_event 6300.
  HARD:     t_pv 0 by construction, t_ind 2638 (BINDS), t_event 3600.
The precursor is not the clock; it is simply THE LATER OF TWO PREREQUISITES in the
unperturbed system. Deleting it reveals the second gate - an in-distribution behavioral
gate the intervention barely moves (2850 -> 2638, 7.4%, p=1.9e-5). Two-gate floor
prediction t_ind + assembly = 2638 + 938 = 3,576 vs observed 3,600 (0.7% error).
Our prereg's law-band bet assumed the precursor was the SOLE prerequisite; that auxiliary
assumption, not the law, is what failed. A pure serial-prerequisite model predicts 975 and
is refuted by 3.7x; conversion efficiency of the supplied head start is 50.7%.

### 4. WHAT THIS DOES AND DOES NOT SAY ABOUT THE GAP LAW (scope carefully)
DO NOT WRITE "the gap law is falsified". The scaffold sets t_pv = 0 BY CONSTRUCTION, so at
the composed anchor the multiplicative rule is INAPPLICABLE, not refuted. The shipped R9
rule is literally t_pv-based, so it is undefined here (not "7/10 covered" - that is a
t_conj-substituted variant and must be labelled as such). Also: our 30 controls are 37.5%
of the law's own n=80 calibration pool, so "controls reproduce 0.843" is IN-SAMPLE
consistency, never independent reproduction. Retire the 0.736-vs-0.739 comparison: 6/10
(not 8/10) hard runs fall below the range floor, 4/10 lie inside it, and 11/30 CONTROLS lie
outside the published IQR.
WHAT DOES SURVIVE, at the NON-degenerate prefix anchor: post-anchor residual is
375/350/375/375 steps across arms whose t_event spans 3,600-6,725. Fleet-wide (85
unscaffolded runs) that gap is near-proportional to t_event (OLS 0.1435*t_event + 98.3,
r=0.795), predicting 566-615 steps at t_event=3600; observed 938 is anomalously HIGH, and
grid6r8 b128 (t_event 4275, post-anchor 650) is an in-fleet counterexample to strict
constancy. HONEST STATEMENT: the assembly phase does NOT compress under this intervention
the way cross-config rescaling predicts. That is evidence about the assembly phase, from
one intervened cell at fixed lr/batch - not a refutation of a functional form.

### 5. CORRECTIONS TO CLAIMS MADE BEFORE THE RED-TEAM (must propagate everywhere)
(a) "Placebo and near-miss do nothing" is FALSE as written. Both point estimates are a
    DELAY (sink 6725, p=0.045 uncorrected / 0.020 pooled; near 6725, p=0.137; paired sink
    4/5 slower, near 3/5). Mechanism visible: the bias occupies the head that would
    otherwise become the prev-token head (prevtok_L0 at step 6000: ctrl 0.416, sink 0.193,
    near 0.146). This STRENGTHENS specificity - versus the matched-bias baseline the effect
    is 46.5%, not 42.9% - but "inert" is wrong. near also yields significantly BETTER final
    models (copy_adv 8.63 vs 6.89, p=6e-6), an unregistered endpoint.
(b) The seed arm is NOT a stickiness test. A +8 additive bias sits in softmax saturation
    with negligible gradient, so the prereg decay bar (prevtok0 < 0.5) was UNREACHABLE BY
    CONSTRUCTION. P-C4 "PASS" is arithmetic, not evidence. It is a same-seed replication
    (9/10 identical t_event); hard+seed is effective n=10, NEVER 20.
(c) Variance collapse: real versus controls and robust to fairness checks - exact
    enumeration of all C(30,10) subsets gives P(SD <= 50.07) = 9.85e-6, P(CV) = 1.15e-4,
    P(range) = 9.85e-6; CV 0.0140 vs 0.0561; Fligner on median-normalized values p=0.019.
    BUT NOT DEMONSTRATED SPECIFIC: sink also tightens ~2.8x and hard is not significantly
    tighter than sink (p=0.24); near does not tighten at all (CV 0.056). Correct claim: the
    scaffold collapses timing variance relative to controls; whether that requires the
    TASK-USEFUL primitive is unresolved at n=5 placebos.
(d) Data necessity (norephard 0/5) has near-zero power: unscaffolded norep is also 0/10. It
    excludes the strong "scaffold manufactures the capability" possibility and nothing more.
(e) The supplied prior is CONTESTED, not clamped - a disclosure that distinguishes this
    design from activation-clamping work: hard's prevtok0 falls 0.977 -> 0.841 at step 2150
    and recovers to 0.916 at event, entirely from learned QK opposing a FIXED buffer.
(f) beta=8 supplies a NEAR-ASYMPTOTIC head (attention weight ~0.907) versus 0.103 in
    controls when their own precursor anchor fires. "Supplying the precursor" is unwarranted
    until the dose axis is run; today's defensible phrasing is "clamping the converged
    prev-token pattern". R2 resolves this.
(g) The residual is NOT explained. During the 3,250-step wait before layer-1 prefix onset,
    train_loss is indistinguishable from controls (0.06x the control seed spread) and prefix
    is flat. It is also NOT a fixed threshold on loss or indist_adv: loss-time re-clocking
    still puts hard's prefix onset ~1,760 loss-equivalent steps early. Characterized, not
    explained; the logged probes do not resolve it.

### 6. PREREG LIT-CHARACTERIZATION ERROR (disclosed)
plan_p7.md's lit position states Singh et al. (2404.07129) is "single initialization, no
seed statistics". THIS IS WRONG - their Fig 7 / App I examine weight-clamping across seeds
5, 6, 7. The genuine gaps remain: no placebo clamp, no near-miss primitive, no
timing-variance statistics, no data-necessity corner, no timing law tested. Also correct in
print: their speedup is ~2.7x (2e5 -> 7.5e4) versus our 1.75x - ours is a weaker but far
more deployable lever (one additive bias versus activation clamping). MIDAS (2409.19044) is
weaker support than the prereg assumed (efficiency + reasoning quality, no circuit or timing
claim) - background only. Nearest architectural ancestors are ALiBi / T5 relative bias and
short-conv / token-shift primitives (H3, RWKV, Mamba, Canon layers); none measures
capability ARRIVAL TIME. Baherwani et al. 2606.25010 verbatim: "emergent capabilities arise
stochastically throughout training, with larger models acquiring them earlier on average" -
no induction-head claim; our variance result constrains but does not refute it.

### 7. STANDING
Defensible one-paragraph claim: supplying a converged prev-token pattern at init advances
induction emergence 6,300 -> 3,600 (1.75x, complete separation, threshold-robust,
artifact-hardened, capability-validated), with no acceleration from a task-useless bias or
an i-2 near-miss; but the precursor is a PREREQUISITE, NOT A CLOCK - it is the later of two
prerequisites, and deleting it exposes a second behavioral gate that sets a ~3,576-step
floor, which is why our own pre-registered law-band bet missed by 3.7x.
Two axes remain uncontrolled and are pre-registered next as R2: DOSE (is a
precursor-strength prior enough, or only a converged one?) and PLACEMENT (does the effect
require the primitive to be upstream of the match head, i.e. genuine composition?).

## P7 R2 PREREG (2026-07-20) — DOSE + PLACEMENT (frozen BEFORE runs/grid7d exists and
## before --scaffold_layer / --log_heads are implemented)
Motivation: the R1 red-team named exactly two uncontrolled axes that decide how R1 may be
described. Both are cheap and both can kill the current framing.

AXIS 1 - DOSE. beta=+8 supplies attention weight ~0.907 on the previous token: a
NEAR-CONVERGED head. Controls sit at 0.103 when their own precursor anchor fires and 0.698
at their event. If acceleration appears only at near-asymptotic beta, the honest claim is
"clamping a converged pattern", not "supplying the precursor".
AXIS 2 - PLACEMENT. attach_scaffold hard-codes blocks[0]. Nothing yet tests whether the
effect requires the primitive to be UPSTREAM of the match head (genuine circuit
composition) rather than merely injecting a position-shifted channel into the residual
stream. A prev-token bias on LAYER 1 cannot compose into induction and should be null.

### Design (runs/grid7d/, 30 runs, fresh seeds 21-25 everywhere)
- DOSE: dose{b}_s21..25 for b in {1, 2, 3, 4, 6} — hard pattern (B[i,i-1]=+b), layer 0
  head 0, fixed buffer, rep condition, otherwise identical to grid7c hard (16k steps,
  lr 1e-3, d256). 25 runs. The beta=8 point is grid7c hard (n=10, median 3,600) and is
  NOT re-run; beta=0 is the control distribution (T0=6,300, n=30, bit-identity re-verified
  40/40 records).
- PLACEMENT: hardL1_s21..25 — identical B[i,i-1]=+8 fixed buffer on LAYER 1 head 0. 5 runs.
- Instrumentation: --log_heads adds PER-HEAD prevtok/prefix vectors to metrics.jsonl.
  OPT-IN so default records stay byte-identical and the K-C2b guard stays usable. Used on
  all R2 runs; resolves suppression-vs-relocation, which max-over-heads cannot. Descriptive
  only, no bet attached.

### Predictions and kills (frozen now)
- P-D1 (dose-response is GRADED, not all-or-none): median t_event non-increasing in beta
  over {1,2,3,4,6,8}, AND beta=4 delivers >= 50% of the full acceleration, i.e.
  median t_event(beta=4) <= 4,950.
  K-D1 FIRES if median t_event(beta=4) > 5,670 (<25% of the acceleration): the effect
  requires a near-converged head. Then "supplying the precursor" is RETIRED from all
  write-ups in favour of "clamping the converged prev-token pattern", and R1 section 5(f)
  becomes the standing description.
- P-D2 (placement / composition): hardL1 median t_event >= 5,040 (0.8*T0 — null, inside
  the control band).
  K-D2 FIRES if hardL1 median <= 4,725 (>= 25% acceleration): the effect is NOT upstream
  circuit composition but a generic offset-1 channel into the residual stream. The
  composition framing dies and R1's mechanism section is rewritten.
- P-D3 (THE TWO-GATE FLOOR — the strong quantitative bet, and the first prospective test
  of R1's post-hoc mechanism): NO cell in this rung has median t_event < 3,400. The
  two-gate account puts the floor at t_ind + assembly ~ 3,576 regardless of scaffold
  strength or site.
  K-D3 FIRES if any cell median < 3,400: the two-gate account is incomplete and R1
  section 3 must be reopened.
- K-D4 (manipulation check, not an outcome): hardL1 runs must show layer-1 prevtok >= 0.8
  at the first eval; dose cells must show first-eval prevtok0 monotonically increasing in
  beta. Failure = implementation void; fix and relaunch permitted BEFORE any outcome
  scoring, outcomes never consulted.
- Scoring: sealed one-shot analysis/score_p7d.py after ALL 30 summaries exist; writes
  analysis/out7/p7d_scored.json; refuses on partial fleet or existing scorefile.
- Disclosure: beta=8 and beta=0 cells are REUSED from grid7c/grid6r2 rather than re-run at
  seeds 21-25, so the dose curve mixes seed sets (seeds 1-10 at beta=8, 1-30 at beta=0,
  21-25 elsewhere). Seed set is not expected to matter at n>=5 given control CV 5.6%, but
  it is a disclosed non-ideality, and monotonicity in P-D1 is therefore judged on the
  five FRESH cells plus the two reused endpoints, with the endpoints flagged in the table.

## P7 R2 VERDICT (2026-07-20) [analysis/out7/p7d_scored.json, sealed; prereg 1affb4a]
Adjudicated by a 6-agent workflow (2 analyses + 2 adversarial verifications + 2 red-team
lenses); every load-bearing number reproduced from disk to the digit in both verifies.
All mechanism analysis is POST-HOC unless marked. Dose cells share seeds s21-25 (paired);
beta=0 (grid6r2, n=30) and beta=8 (grid7c, n=10) endpoints are REUSED (disclosed).

### PRE-REGISTERED OUTCOMES (sealed, unchanged)
P-D1 PASS (beta4 3425 <= 4950), monotone check False. K-D1 no-fire. P-D2 PASS (hardL1
6700 >= 5040). K-D2 no-fire. P-D3 FAIL / K-D3 FIRES (dose2 3375, dose3 3350 below the
3400 floor). K-D4 manipulation PASS.

### 1. K-D1 NO-FIRE IS THE STRONGEST POSITIVE: a SUB-THRESHOLD prior already accelerates
beta=1 imposes prev-token weight 0.0787 at init - BELOW the 0.10 anchor the whole campaign
uses to declare the precursor "present" (verified 5/5 runs) - and still moves emergence
6,300 -> 3,825 (recovering 83.9% of the achievable acceleration). The network amplifies
that sub-threshold hint past 0.10 by step 100 (control: step 5,325) and to 0.205 by step
2,000 (4.9x control). CONSEQUENCE: "supplying the precursor" is vindicated over "clamping
a converged head" - a weak seed the network grows itself is enough. "Precursor present" is
a RATE, not a binary state; the 0.10 anchor is the wrong primitive. This is the sentence
R2 was run to earn, and it earned it.

### 2. THE DOSE CURVE IS U-SHAPED (ascending arm real; floor unresolvable)
Median t_event by beta: 0->6300, 1->3825, 2->3375, 3->3350, 4->3425, 6->3525, 8->3600.
- ASCENDING ARM (beta 3->8, stronger priors are WORSE) is REAL, established PAIRED INSIDE
  grid7d ALONE (no reused endpoint needed): seed-paired diffs b3->b4 = [75,75,75,75,75],
  b3->b6 = [175,175,125,150,175], b4->b6 = [100,100,50,75,100] - 15/15 positive, ZERO
  reversals. Unpaired corroboration b3 vs b8 MWU p=0.0025 (asymptotic; exact floor
  6.7e-4), pooled {2,3,4} vs {6,8} p=8e-6. Spearman over {3,4,6,8} rho=+0.84.
- FLOOR (beta 2 vs 3) is ONE 25-step tick apart and UNRESOLVABLE at n=5 on a 25-step grid
  (MWU p=0.59, tied on the upstream clock). The optimum is a flat plateau over beta in
  [2,3], not a point. Over-dosing 3->8 costs 250 steps (8.5% of achievable acceleration);
  under-dosing 3->1 costs 475 (16.1%).
- MECHANISM (post-hoc, threshold-robust): the ENTIRE U lives in when layer-1
  prefix-matching turns on. t_event = t_pfx + a beta-independent ~250-step lag (pooled lag
  median 275, sd 20, range [225,300] against a 2,912-step spread in t_pfx). ADOPT-vs-FIGHT
  crossover, 35/35 runs zero exceptions: beta<=3 the biased head is AMPLIFIED above its
  imposed value; beta>=4 it is SUPPRESSED (the network fights a too-rigid prior). Dip
  DEPTH does not order the delay; the pre-induction LEVEL the head is pinned at does.
  Best single-parameter post-hoc fit is softmax plasticity -p*(1-p*) (rho 0.85) - FLAGGED
  WEAK, curve-fit on 7 medians, several alternatives fit equally; it is an R3 hypothesis,
  not a finding. No relocation to other heads at any dose (biased head is L0 argmax 25/25).

### 3. K-D3 FIRED - R1's TWO-GATE MODEL IS SUPERSEDED, mechanism now identified
The floor breach is STRUCTURAL, not a bad threshold. Across beta the second gate t_ind is
essentially FLAT (~2,660-2,725, MWU vs beta8 all p>=0.57); what varies is the post-anchor
ASSEMBLY interval, which R1 treated as a fixed 938-step overhead. Re-scoring R1's rule on
every R2 cell: accurate only at beta=8 (the cell it was fitted on), over-predicting by
138-263 steps everywhere else. Measured t_event - t_ind = 1100/700/675/750/800/962 for
beta 1/2/3/4/6/8 - NOT constant. The quantity R1 called fixed overhead IS the beta-
dependent term that generates the U. R1 VERDICT section 4's "assembly phase does not
compress" is now OVERTURNED by direct dose evidence: it compresses to ~675-800 steps at
mid-beta. Superseding model: t_event = t_pfx(beta) + ~250, and t_pfx is gated by how fast
layer-1 prefix-matching forms once the induction advantage exists.
CORRECTION (verify, must propagate): the "complete range separation, exact perm p=0.00067"
framing of the assembly compression does NOT survive a robustified gate. Requiring
indist_adv>=0.10 for 2 CONSECUTIVE evals (matching R1's own copy_adv event rule) moves
beta8's t_ind +62.5 while the dose cells do not move; assembly deltas attenuate to
-150/-125/-100, range separation is LOST, exact perm p -> 0.0013/0.0017/0.0043, and dose6
loses significance (p=0.21). DIRECTION and p<0.005 SURVIVE; the "disjoint ranges" wording
does not. Also: dose2/3/4 share seeds, so those three p-values are dependent, not
independent replications.

### 4. K-D2 NO-FIRE, BUT THE COMPOSITION READING IS UNDER-DETERMINED (red-team)
hardL1 (converged +8 prev-token bias on LAYER 1) = 6,700, no acceleration, and the biased
L1 head never becomes an induction head (prefix max-ever 0.029-0.052 across 5 runs) while
layer 0 grows its OWN prev-token head at near-control timing. That is consistent with the
composition story. BUT 6,700 is statistically IDENTICAL to R1's task-useless placebos
(sink 6725, near 6725; hardL1 vs pooled placebo dMedian p=0.77) and reproduces their exact
~400-step head-burning delay - so it is equally consistent with "layer-1 head-0 is a
generically bad site for ANY fixed bias." The design lacks the discriminating control (an
L1 PLACEBO - a task-useless bias on layer 1). DEFENSIBLE CLAIM: "a converged prev-token
bias on layer 1 does not accelerate emergence and does not itself become the induction
head." NOT DEFENSIBLE from this data: "the effect REQUIRES the primitive upstream = proven
circuit composition." Retire the strong wording; the L1-placebo control is an R3 item.

### 5. STANDING AFTER R2
Confirmed and strong: (i) a sub-threshold prev-token seed causally accelerates emergence
(K-D1) - the precursor is genuinely causal at precursor strength; (ii) the dose-response
is U-shaped with an interior optimum near natural anchor strength, real on its ascending
arm; (iii) the accelerant does not relocate and the biased head is adopted or fought
depending on dose. Overturned/superseded: R1's fixed-assembly two-gate model; the
rate-limiter is now localized to layer-1 prefix-matching onset. Under-determined (-> R3):
composition-vs-site-quality (needs L1 placebo); the exact floor location (needs finer beta
on a 10-step grid); the plasticity mechanism (needs a time-varying-scaffold test that
imposes beta=8 then decays it - the plasticity account predicts recovery of beta=3 timing,
the distance-from-converged account does not). No R3 launched; this is the R2 stopping
point. All claims scale-local: 2-layer d256 bigram LM, one lang_seed, one task.
