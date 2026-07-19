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
