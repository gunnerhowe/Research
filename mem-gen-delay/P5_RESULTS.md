# P5 RESULTS DOCUMENT — "Does Interpretability Buy Warning Time?"
## Paper-grade internal documentation (NOT for release; the release bar is P6)

Written 2026-07-16. Every number regenerates from the artifact cited beside it.
Prereg chain (all pushed to github.com/gunnerhowe/Research):
2cf62e3 (plan+kills BEFORE code) -> 96f9c0a-era P4 infra -> freeze commit BEFORE test ->
5595d3e (R2b/R3 prereg BEFORE runs) -> cbea4a6 (R3/R2b verdicts) -> 2365021 (R4+R5 prereg)
-> 10082d3 (R5 verdict + R5b prereg) -> 7db1973 (R5b verdict; experiments complete).

## The question
Can internal (representational) signals FORECAST delayed generalization before behavior
moves — scored as forecasts (lead time at controlled false-alarm rate), raced against
scalar training signals, stress-tested under interventional shift, and validated
prospectively?

## Corpus
495 runs total: 466 pre-existing (grid 95 alg / grid2 100 alg / grid3 188 alg (13 seq
excluded) / grid4 35 mnist / grid4b 48 mnist) + 20 prospective (grid5r3) + 10 label-free
(grid5r5) + 9 confirmation (grid5r5b). ~210k+ synchronized eval records: behavioral
(train/test acc, CE), scalar (wnorm, logit_scale, conf), representational (fourier_top8,
fourier_gini, cos_gap, fisher; MNIST cos_gap), label-free spectra (grid5r5/b only:
eff_rank, part_ratio, top1_frac). Interventions shift t_gen ~100x and decouple the norm
clock (the corpus's unique property; no public corpus has this).

## Protocol (frozen in plan_p5.md before any analysis)
Event = t_gen (test acc >= 0.95 alg / 0.85 mnist). Dynamic forecasting: alarm from history
prefix only; warmup 5 evals. Thresholds/models fit on TRAIN (even seeds) under FA <= 5%
over train negatives; TEST = odd seeds, evaluated once. Lead = t_gen - t_alarm (miss = 0);
censored runs scored at budget. Champions selected on train, compared on test.

## Rung-by-rung results (artifact -> analysis/out5/)

### R1 — retrodictive benchmark [r1r2_stats.json, frozen_eval.json]
- MNIST single signals: wnorm <= 91.92 gives 1,000-step test lead; d.cos_gap 4,000 (but FA
  1/3 on tiny negative set). MNIST frozen MULTIVARIATE (SRq: S+R levels/slopes + wnorm^2 +
  cos_gap x norm interactions; pw20/it8000/W=median t_gen; tuned train-side only, odd seeds
  never loaded during tuning): TEST median lead 10,400 = 59.8% of delay, miss 0/35, FA 1/3
  where the 1 is base_c92_s1 (censoring-noise: still climbing at budget, crossed 0.80 at
  43k). Structural negatives silent (maxP ~0.49 vs tau 0.861). K1 no-fire (SRq +940% over
  best scalar), K2 no-fire.
- ALGORITHMIC: NO feasible operating point at FA <= 5% for ANY forecaster family
  (single-signal, linear multivariate, quadratic multivariate). Cause (diag_p5_algzero.py):
  trap negatives — supcon@clamp35 runs build COMPLETE structure (fourier 0.996 > positives'
  median 0.69 AT t_gen) yet never generalize (low-norm inversion, Paper 3); band arms fire
  0.81-0.91. Plus cadence: t_gen quartiles 100/2,300/8,125 at 50-epoch evals = no room for
  lead on fast runs. K1/K2 FIRE as "infeasible" -> forecastability boundary.
- Negative taxonomy formalized: BUDGET-CENSORED (c92 arms: grokked at 84k/92.8k/93.6k on
  3 later seeds — slow, not blocked) vs STRUCTURALLY BLOCKED (shufpair: ceiling 0.76-0.79,
  norm 133-137).

### R2 — interventional shift [r1r2_stats.json r2_*, frozen_eval.json r2]
Train on control-family only, test on prior arms (where Papers 1-4 established the norm
clock decouples: prior arms grok at frozen ~92 / growing ~104 norm).
- wnorm forecaster COLLAPSES (0 lead, 51% miss single-signal; multivariate 0 lead, 60-100%
  miss). Bare content probe SURVIVES: cos_gap 4,400 lead / 4.7% miss; d.cos_gap 16,400 / 0
  miss.
- THE CENTRAL FINDING: robustness-vs-false-alarm tradeoff. The context features (norm) that
  let a forecaster control false alarms in-distribution are exactly the features that break
  under intervention. A monitor tuned quiet on the training distribution goes blind when
  the recipe changes.

### R2b — mechanism-factored two-gate [r2b_twogate.json]
Alarm = content gate AND norm-viability window with A-PRIORI mechanism constants (alg
[40,inf): c35 inversion below; mnist (0,110]: wrong-structure explosion above). Only the
content threshold is fit.
- MNIST under shift: 12,400 median lead (53% rel; d.cos_gap 16,400 / 94% rel), 0 miss.
  The tradeoff is RESOLVED where the mechanism is understood.
- ALG: veto fixes FA (fourier 0% test FA) but buys no lead -> K8 FIRES; boundary final.

### R3 — prospective validation [r3_scored.json; grid5r3 = 20 runs launched AFTER 5595d3e]
ALL FOUR PRE-REGISTERED PREDICTIONS PASSED:
- P-R3a: frozen SRq on new baseline-family runs: median lead 18,000 (bar 5,200) — BETTER
  than retrodictive 10,400 — including UNSEEN clamp-60 pin (leads 14,400-21,600).
- P-R3b: frozen cos_gap threshold on prior arms: 8,400 (bar 2,200); d.cos_gap 35,400.
- P-R3c: control-trained SRq missed 100% of prior arms (bar >= 40%) — tradeoff prospective.
- P-R3d: 0 alarms on all 3 structural negatives x all 3 frozen artifacts.
- c92 exempt class: 1/2 alarmed (consistent with censoring-noise). K4 no-fire.

### R4 — cross-domain transfer [r4_transfer.json]
K5 FIRES: no signal transfers in either direction at target FA <= 5%, even per-domain
z-standardized. alg->mnist thresholds trap-hardened (miss 0.52-0.87 or FA 0.875-1.0);
mnist->alg permissive (FA 0.20-1.0). CALIBRATION IS REGIME-SPECIFIC; the portable
ingredient is mechanism (R2b), not fitted thresholds.

### R5 — label-free battery [r5_scored.json; grid5r5, --log_spectra instrumentation]
- AS-REGISTERED K6 comparison INVALID (disclosed): single fit-side negative made the FA cap
  vacuous; degenerate level-thresholds won the fit and false-alarmed on validation (FA 1/1).
  Protocol lesson: FA discipline requires a minimum negative count; add to any future spec.
- FA-valid exploratory reading: d.top1_frac (slope of top eigenfraction, no labels) =
  36,200-step validated lead (97% of delay), 0 miss, 0 FA — ~5x task-aware cos_gap (7,200).

### R5b — pre-registered confirmation [r5b_scored.json; grid5r5b, frozen thresholds]
K9 FIRES — the label-free superiority claim DIES: d.top1_frac cleared its lead bar (34,200
>= 18,100) but false-alarmed on 1/3 fresh structural negatives, and scored 0 lead on 2/3
BASELINES while leading ~the full delay on all supcon arms. Re-reading: the signal is
largely an AUX-ARM DETECTOR (contrastive term drives sharp early spectral collapse), not
universal pre-emergence structure. cos_gap passed its lead bar (13,400) but ALSO
false-alarmed 1/3 and led 0 on all baselines. RUNG LESSON: single-threshold probes
calibrated on small negative sets are fragile out-of-sample. (R1-R3 core untouched:
different machinery, larger negative sets, R3 prospective structural FA = 0.)

## Scoreboard
| rung | claim | verdict |
|---|---|---|
| R1 mnist | forecastable in-distribution | CONFIRMED (10,400 lead, 60% of delay) |
| R1 alg | forecastable | INFEASIBLE at 5% FA (traps + cadence) — boundary |
| R2 | tradeoff: FA-control vs shift-robustness | CONFIRMED (the core finding) |
| R2b mnist | mechanism-factoring resolves tradeoff | CONFIRMED (12.4-16.4k, 0 miss) |
| R2b alg | mechanism veto rescues alg | K8 FIRED (FA fixed, no lead) |
| R3 | prospective validity | ALL 4 PREDICTIONS PASSED (18k lead unseen configs) |
| R4 | calibration transfers across regimes | K5 FIRED (it does not) |
| R5/R5b | label-free probes suffice | K9 FIRED (aux-arm detector; fragile) |

## What this establishes / what it does not
ESTABLISHES: scored emergence forecasting is possible in-regime with mechanism-informed
context (prospectively); the tradeoff is real and named; negative-set quality is the
binding resource; calibration is not portable; "early warning" claims without FA
accounting are untrustworthy (we produced three seductive numbers that died under it).
DOES NOT: frontier applicability (model systems only); unknown-capability warning (all
probes capability- or intervention-specific); label-free sufficiency (killed).

## Known imperfections (for any future write-up's limitations section)
R5 single-negative degeneracy (protocol design error, caught at validation); alg kill
criteria didn't pre-name "infeasible" as an outcome class; corpus inherited 50-epoch
cadence + budget from Papers 1-4 rather than forecasting-first design; c92 budget too
short for its class; W/warmup choices under-justified; multivariate first implementation
underfit (caught train-side); P4-era worker race (no corruption, disclosed).
