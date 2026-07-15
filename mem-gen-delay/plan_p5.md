# P5 PLAN + PRE-REGISTRATION: "Does Interpretability Buy Warning Time?"
## Forecasting emergence timing from internal signals, with scalar baselines, false-alarm
## accounting, interventional shift, and prospective validation

Committed BEFORE any forecasting analysis is run (R1 code does not exist at commit time).
Novelty basis: lit_notes_p5.md (R0, complete). Companion corpus: runs/grid{,2,3,4,4b},
466 runs, ~210k eval records, probes logged every 50-200 epochs/steps.

## Framing

Emergence forecasting = dynamic time-to-event prediction. Event = behavioral generalization
(t_gen: test acc >= 0.95 algorithmic / 0.85 MNIST, as in Papers 1-4). At each eval time t a
forecaster sees the run's history up to t and outputs (a) an ALARM decision and (b) P(event
within horizon W). Scoring on held-out runs:
- LEAD TIME = t_gen - t_alarm for the first alarm (alarms after t_gen score 0; no alarm on a
  grokking run = miss).
- FALSE-ALARM RATE = fraction of never-generalizing runs (censored at budget) that alarm.
- Thresholds are fit on TRAIN runs to maximize median lead subject to FA <= 5% on train
  negatives; all reported numbers come from TEST runs.
- Calibration of P(event within W): reliability slope + Brier, W = 0.2 x median t_gen of
  grokking train runs per domain.
- Primary summary per signal set: median lead time (absolute and as fraction of t_gen) at
  the <=5%-FA operating point, on held-out seeds.

Signal sets (levels + trailing-5-eval slopes for each signal):
- S (scalar/cheap, computable from loss+outputs+weights without internals):
  wnorm, train_ce, test_ce, logit_scale, conf, osc(train_ce) = std of residual from a linear
  fit over the trailing 50 evals (Notsawo-style oscillation feature).
- R (representational, needs hidden reps + task structure):
  algorithmic: fourier_top8, fourier_gini, cos_gap, fisher. MNIST: cos_gap.
- S+R combined.
Forecasters: (i) per-signal threshold alarms (direction fit on train; transparent, primary
for the lead-time table); (ii) pooled logistic regression per domain on standardized
features for P(event within W) (secondary, for calibration + combination).
Split: TRAIN = even seeds, TEST = odd seeds (all grids). No peeking at test until thresholds
and code are frozen; the analysis script prints train and test blocks separately.

## Rungs, each with its kill criterion

R0 (DONE): lit kill-test. Kill = a prior paper already scores lead time vs scalar baselines
    -> none found (lit_notes_p5.md); proceed.

R1 RETRODICTIVE BENCHMARK (zero new GPU): the protocol above on all 466 runs.
    P1: in-distribution, wnorm is a strong forecaster in free-norm control runs (delay law).
    P2 (uncertain): R or S+R beats best-S in median lead at matched FA in-distribution.
    K1: if no R/S+R signal beats best-S by >= 20% relative median lead in EITHER domain,
        in-distribution probes add nothing -> the paper's in-distribution claim becomes the
        honest negative "scalar clocks suffice in-regime" (benchmark still ships).
    K2: if NO signal (incl. scalars) achieves median lead >= 10% of median t_gen at 5% FA,
        the regime is unforecastable at these budgets -> report as such (tension with
        Notsawo's whether-result, interesting either way).

R2 INTERVENTIONAL SHIFT (the sharp rung; zero new GPU): train forecasters ONLY on
    control-family runs (baseline, shuffled, wrong, band, norm-matched/replay, clamp arms,
    augce), test on contrastive-prior arms (supcon_true/comm/window/adaptive/anneal in
    algorithmic; supcon_aug/label/nn in MNIST), where Papers 1-4 established the norm clock
    DECOUPLES from the outcome (groks at frozen ~92 / growing ~104 norm).
    P3 (pre-registered, mechanism-backed): norm-led scalar forecasters degrade severely
    under this shift (missed alarms or FA explosion); representational probes retain >= 50%
    of their in-distribution median lead.
    K3: if probes ALSO lose > 50% of their lead under shift, the robustness claim of
    internal-signal early warning fails in these regimes -> honest negative, reported.

R3 PROSPECTIVE VALIDATION (modest GPU): freeze the R1/R2 forecasters + thresholds (commit
    hash), pre-register point forecasts + intervals for ~20-30 NEW runs (fresh seeds; at
    least 2 conditions and 1 lambda value never seen by the forecaster), then run them.
    K4: prospective coverage < 60% of nominal or median prospective lead < 50% of
    retrodictive lead -> retrodictive numbers were overfit; report the gap.

R4 CROSS-DOMAIN TRANSFER (zero new GPU): fit on algorithmic, test on MNIST (features
    domain-standardized; cos_gap is the shared probe) and vice versa.
    K5: no signal transfers (lead ~ 0 at matched FA) -> forecasting is regime-specific;
    report as scope limit.

R5 LABEL-FREE BATTERY (modest GPU, only if R1/R2 positive): add label-free internal signals
    (penultimate-spectrum effective rank, participation ratio, optionally 2RDM heat capacity
    from arXiv:2603.29805 and LLC if compute allows) to train.py/train_mnist.py logging;
    piggyback on R3's new runs.
    K6: label-free signals recover < 50% of task-aware-probe lead -> early warning needs
    capability-specific probes; report (this is itself decision-relevant for monitoring).

R6 (STRETCH, only if R1-R4 land) INDUCTION-HEAD SEED-VARIANCE FORECASTING: at fixed config
    (where arXiv:2511.16893's config-based equation predicts a single formation time),
    forecast PER-SEED timing residuals from within-run previous-token-head strength.
    2-layer attention-only models on the 3080. K7: per-seed residuals unforecastable from
    precursors -> config-level prediction is the ceiling; report.

## Disclosures (D)
D1: R1/R2 are retrodictive analyses of runs that already exist and were partially inspected
    during Papers 1-4 (e.g., we know probe-lead anecdotes and the norm-decoupling fact used
    in P3). The analysis SPEC and kill criteria above are what is pre-registered; the runs
    are not blind. R3 is the blind rung.
D2: All rungs reported regardless of outcome; kills convert claims to their honest negative
    forms, never delete rungs.
D3: Test-seed results computed once per frozen spec; any spec change after seeing test
    results is disclosed as an amendment with its own commit.
D4: Probes are task-aware; framed as capability-specific monitoring (R5 addresses
    label-free).

## Deliverable
paper4/ manuscript: "Does Interpretability Buy Warning Time? A Forecasting Benchmark for
Emergence Timing" + released benchmark (metrics.jsonl corpus + forecasting harness).
House rules: gen_numbers/verify_regen from line one; censoring scored at budget; counts not
p-theater at small n.
