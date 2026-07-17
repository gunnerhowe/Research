# The Emergence-Forecasting Benchmark (v0.1)

The corpus, frozen rules, and scoring harness behind *"Capability Emergence Can Be
Forecast"* (paper4/). Everything needed to reproduce every number, or to evaluate a new
forecaster under the same discipline.

## What forecasting means here
An **event** is an absolute behavioral crossing (copy advantage >= 2.0 nats, two
consecutive evals — never a fraction of a run's own maximum). A **forecaster** sees the
log prefix and may alarm once; **lead** = t_event - t_alarm; **false-alarm rate** is
measured on *negative* runs where the capability cannot form. Forecast quality is a
(ranking, lead) pair with conformal intervals whose empirical coverage is reported.
Minimum five negatives on any side of any threshold fit. Blind validation = constants
frozen and commit-stamped before the runs exist.

## Corpora (all logs in-repo; probes logged every 25 steps unless noted)

| corpus | contents | role |
|---|---|---|
| `runs/grid6r2` | 30 positives + 10 data-ablated + 5 architectural negatives (bigram language, config A) | calibration fleet |
| `runs/grid6r5` | 10 positives + 3 negatives, config B (d320, p_rep .6) | blind gate B |
| `runs/grid6r7` | 10 positives + 3 negatives, new language (seed 888) | blind gate C |
| `runs/grid6r6` | 10 positives + 10 blocked negatives, **trap language** (modal trigram) | anchor-composition rung |
| `runs/grid6r8` | 4 x 5 runs: lr {5e-4, 2e-3}, batch {32, 128} | gap-origin cells |
| `runs/grid6r9` | 5 positives + 2 negatives at unseen lr 7e-4 | blind gate for the gap law |
| `runs/p6r0` | Pythia 70m/160m/410m (+deduped) checkpoint probe trajectories | public-suite evidence |
| `runs/p6ord` | OLMo-1B + OLMo-2-1B checkpoint probe trajectories | cross-family ordinal test |
| `runs/grid*`, `runs/grid5*` | 466-run grokking corpus + P5 prospective/label-free fleets | methodology substrate |

Per-run schema: `metrics.jsonl` rows with `step, tokens?, train_loss/accs, copy_adv,
indist_adv?, prefix_by_layer, prevtok_by_layer` (+ `wnorm`/probes in grokking corpora);
`summary.json` holds config + `t_event` under the frozen rule.

## Frozen rules and results they produced
- Precursor anchor: early-layer previous-token score >= 0.10. Per-seed timing rho 0.977
  (n=30), median lead 975 steps; blind gates 10/10 and 9/10 coverage.
- Composed anchor (trap-safe): precursor AND in-distribution copy ramp >= 0.10 at the
  same eval. 0 FA in both language classes; rho 1.000 in the trap fleet.
- The gap law: anchor fires at 0.843 of time-to-emergence (n=80, IQR 0.825-0.865);
  multiplicative envelope [1.11, 1.36] x t_anchor blind-validated 5/5.

## Reproduce / evaluate
- Training: `src/train_lm.py` (fleets; `--lang trigram` for the trap class);
  `src/probe_pythia.py` (public checkpoint suites, generic `--revisions`).
- Scoring: `analysis/score_r{2,4,5,6,7,8,9}_p6.py` — each is a sealed one-shot evaluator
  matching a pre-registered spec in `plan_p6.md`.
- Numbers in the paper regenerate via `paper4/gen_numbers.py` (byte-verified by
  `paper4/verify_regen.py`).
- To evaluate your own forecaster: consume `metrics.jsonl` prefixes causally, alarm once
  per run, and score with the harness conventions above. If you tune anything, tune on
  even seeds and report odd seeds once.

## Provenance
Every spec freeze precedes its data in the public commit history (chain documented in
`plan_p6.md` and `P5_RESULTS.md`, from `2cf62e3` onward). Kill criteria that fired are
reported in the paper's ledger, not deleted.
