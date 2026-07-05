# PLAN — Rice's Formula for Event-Driven Efficiency (paper #4 of the Kac-Rice program)

Working plan per info.txt. Every deviation from the brief is recorded in the
"Deviations & decisions" section below, dated.

## Status board

- [x] Package skeleton (src/eventrice), vendored estimator, delta cells
- [x] tests/test_correctness.py green via `python -m pytest tests/ -q` (14)
- [x] E0 estimator validation — **GATE V PASS** (Rice on known spectra
      <=1.9%; OU discrete-Gaussian <=1.0%; multisine <=0.2%; threshold 3%)
- [x] Data + base networks: SC2 GRU 91.6-91.9% (8 seeds), psMNIST GRU
      98.6-98.7%, enwik8 TransformerLM (12k steps, 20M chars)
- [x] E1 SC2 + psMNIST: **GATE P direction confirmed** — empirical (b)
      1.6-7.5% median rel err; Rice (a) breaks with depth/non-Gaussianity
      (14-73%), iid (c) 2.3-17x off. Event rates: open-loop calibration
      predictor 0.3-10% in the deployable regime; synthetic-universal gamma
      off by up to ~50% on real traces (heavy-tailed increments) — honest
      finding, empirical curves used for allocation.
- [ ] E1 enwik8 + formal GATE P over all three tasks
- [ ] E2 analytic threshold allocation vs grid-search incumbent
- [ ] E3 budget training (headline, 8 seeds SC2) — Pareto vs post-hoc
- [ ] E4 null controls: L1-delta, rate reg, post-hoc sweep; delta histograms
- [ ] bench_timing.py overhead; make_figures.py; gen_paper_numbers.py
- [ ] paper/main.tex compiled; regenerate-and-diff check passes
- [ ] Pre-submission arXiv sweep re-run (see Kill-check log)

## Task/model choices (brief allows a pick)

- **Primary 1: Google Speech Commands v2**, 35-class keyword spotting,
  log-mel/MFCC front end, 2-layer GRU (canonical for DeltaKWS-style
  comparisons). Features precomputed once and cached (float16) so epochs are
  seconds on the 3080.
- **Primary 2: pixel-stream task = row-sequential MNIST** (28 steps x 28
  features, GRU). Chosen over DVS128 Gesture because it loads cleanly from
  torchvision (DVS128 requires a manual IBM sign-up download; brief says
  "choose whichever loads cleanly"). Rows of natural digits are temporally
  correlated, which is the regime where delta coding wins.
- **Secondary: enwik8 slice char-LM** with a small delta-transformer
  (delta-encoded inputs to QKV/out/FFN projections across the sequence/time
  axis). Data reused from E:/GitHub/Research/ignore-temp-context-info-seq
  data_cache (enwik8.npy). Secondary = full E1 treatment; E2/E3/E4 run on the
  two primaries (recorded as a scoping decision, 2026-07-05).

## Experiment ladder (gates per info.txt)

- **E0 (GATE V)**: random-phase-sinusoid GPs + OU process; Rice from known /
  empirical spectral moments vs dense-grid hard counts, multiple levels,
  3 seeds, target <=3%. Multisine exact counts. eps sweep (bias-variance,
  documented once). PLUS: delta-cell (send-on-delta w/ memory) event counts vs
  ladder-crossing prediction — the hysteresis correction curve vs theta/sigma;
  and closed-loop vs open-loop event counting decomposition.
- **E1 (GATE P)**: base (non-delta) nets; record per-layer/per-channel
  activation traces; measured delta-event rates (closed loop, faithful cell)
  at a grid of thresholds vs three predictors: (a) Rice from fitted spectral
  moments (Gaussian), (b) empirical crossing-density profile from a held-out
  calibration split (no Gaussianity; the differentiable one), (c) Gaussian-iid
  baseline (no temporal structure). Report where (a) breaks (heavy tails).
- **E2**: analytic per-layer threshold allocation by inverting predicted rate
  curves vs grid/random search incumbent; accuracy-vs-events Pareto + tuning
  cost in forward passes (matched-budget and asymptotic).
- **E3**: fine-tune with L_task + lambda * one-sided crossing budget
  (quantile-ladder levels per paper3 Rule 1; per-layer budgets from E2
  allocation shape). Pareto accuracy vs events/step, >=3 seeds (8 for the
  SC2 headline if runtime allows), paired across shared inits.
- **E4**: (i) L1 penalty on |a_t - a_{t-1}|; (ii) global rate regularizer;
  (iii) post-hoc threshold sweep on same architecture. Delta-magnitude
  histograms under (i) vs budget to test the mechanism claim directly.

## House discipline checklist (from info.txt, enforced)

1. numbers.tex auto-generated; regenerate-and-diff before submission.
2. tests green via plain pytest.
3. >=3 seeds, mean±sd, paired where inits shared.
4. Split-half noise floors for crossing-rate estimates on short traces.
5. Pre-registered kill conditions as written in info.txt (V, P, E4-tie).
6. Delta cell faithful to Neil et al. 2016 (threshold on |a_t - a_hat|,
   anchor update on event); theta=0 must reproduce the dense GRU exactly
   (unit test).
7. Paper: honest abstract, "What we do not claim", Limitations,
   Reproducibility with DOI/URL slot + environment block.
8. arXiv sweep re-run before submission (log below).
9. bench_timing.py overhead measured and reported.

## Kill-check log

- 2026-07-05: atlas kill-check clean vs 3.09M-abstract index + web (per
  info.txt handoff). Pre-submission re-sweep: PENDING (do before shipping).

## Deviations & decisions

- 2026-07-05: DVS128 Gesture replaced by row-sequential MNIST (clean loading;
  allowed by brief).
- 2026-07-05: Secondary task (enwik8) scoped to E1 only; E2-E4 on primaries.
- 2026-07-05: Temporal estimator implemented in two forms: (i) midpoint
  co-area form (direct vendoring of kac-rice crossing_density with
  |da/dt| by finite differences), and (ii) an exact piecewise-linear "segment"
  form using Gaussian-CDF differences per step interval, which converges to
  the hard sign-change count as eps->0 and remains differentiable. E0
  documents both; the segment form is the default training objective because
  activation traces are coarsely sampled (|delta a| can exceed eps within one
  step, which the midpoint rule underestimates).
- 2026-07-05: Predictor (a) uses the continuous Rice formula from FD-fitted
  spectral moments (as specified); the exact discrete-time stationary-Gaussian
  crossing probability (bivariate-normal form) is also implemented in rice.py
  and reported in the appendix (continuous Rice underpredicts by up to ~10%
  when lag-1 correlation is weak; the discrete form removes that bias).
- 2026-07-05: 35-class SC2 setup (all keywords) rather than the 12-class
  (10 + silence + unknown) variant, to avoid synthetic silence-clip
  generation; stated in the paper.
- 2026-07-05: E5 (stretch) realized as an energy-per-decision table for SC2
  keyword spotting (modeled energy, Horowitz constants, dense vs delta at
  chosen operating points) computed from E2/E3 results — no new dataset, no
  silicon claims. DVS demo dropped (E1-E4 land on two primaries + secondary).
- 2026-07-05: enwik8 base LM retrained at 12k steps / lr 1e-3 / 20M-char
  slice after the first 3k-step run underfit (3.43 bpc); final ~2.3-2.5 bpc
  expected. Base GRU seeds extended to 8 on SC2 for the E3/E4 headline
  (house rule 3); psMNIST stays 3 seeds; enwik8 3 seeds (secondary).
