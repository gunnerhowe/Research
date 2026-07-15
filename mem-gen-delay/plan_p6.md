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
   behavioral copy advantage (2nd-half vs 1st-half CE on repeated random sequences — the
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
   runs with identical logging: (a) data ablation — within-context repeated-bigram
   stripping (removes the IH training signal); (b) architectural — 1-layer models (IH
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

## Disclosures
D1 R0/R1 events on public suites are defined after seeing R0 curves (relative criteria
   chosen to minimize arbitrariness); the blind rung is R5, as in P5.
D2 All kills reported internally; the release decision is the user's, gated on the bar.
D3 Downloads: public model checkpoints from huggingface.co (EleutherAI/pythia-*), ~150-350
   MB per checkpoint, ~3-7 GB per size for ~20 checkpoints; HF cache pinned to the E: drive.
D4 P5's min-negative-count rule applies to every FA number in P6.
