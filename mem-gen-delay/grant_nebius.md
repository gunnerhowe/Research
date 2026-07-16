# Nebius grant application draft (2026-07-17)

## Project name
Forecasting Capability Emergence in Language Models: A Multi-Seed, Negative-Certified
Benchmark at Scale

(short form if the field is tight: "Forecasting Capability Emergence in Language Models")

## Estimated project cost
$50,000 in compute credits.

Itemization (H100 on-demand ~$2.5/GPU-hr):
- Multi-seed pretraining fleets, 70M-410M params to ~10B tokens (30+ seeds per config,
  3 configs, dense checkpointing through capability phase changes): ~1,500 H100-h
- Manufactured capability-blocked negative runs (data-ablated + architectural) at the
  same scales: ~700 H100-h
- Second + third capability tracks (in-context learning task battery, arithmetic /
  syntactic circuits), incl. precursor-identification probing sweeps: ~2,500 H100-h
- 1B-param validation fleet (10 seeds + negatives): ~1,200 H100-h
- Gap-origin scaling study (lr/batch/width factorial): ~800 H100-h
- Design-iteration / re-run headroom (empirically ~1.5x in our program): ~3,000 H100-h
- Total ~10,000 H100-h ~ $25k GPU + ~$3-5k storage/egress (dense multi-seed checkpoint
  corpus, 5-15 TB, public release) + eval/CPU nodes -> $30-35k core; $50k covers the
  extended capability tracks and public-corpus hosting through release.

(Floor option if asked: $15k delivers the single-capability core at 70M-410M.)

## Project summary (259 words)

Frontier labs cannot currently predict when a training run will acquire a specific
capability: loss improves smoothly while abilities appear abruptly. We have built the
first evaluation discipline that treats capability emergence as a forecasting problem —
lead time at controlled false-alarm rates, conformal prediction intervals, and blind
pre-registered validation — and demonstrated, on consumer hardware, that emergence timing
is forecastable. Across 30 small transformers at identical configuration, the formation
time of a mechanistic precursor circuit (the previous-token head) forecasts each seed's
induction-head emergence at Spearman rho = 0.977 with ~15% of training as advance
warning; a frozen forecasting rule then passed a blind pre-registered gate on a
never-seen configuration with 10/10 interval coverage. Critically, all false-alarm rates
are certified against manufactured capability-blocked negatives — training runs
constructed so the capability cannot form — a resource that exists in no public
checkpoint suite, and our probing of public Pythia suites shows why public artifacts
cannot answer this question: one run per configuration.

This grant scales the methodology from synthetic languages to real language-model
pretraining. We will train multi-seed fleets (70M-1B parameters, ~10B tokens each) on
real text with dense checkpointing through capability phase changes, manufacture blocked
negatives at scale via targeted data ablation, extend from induction to in-context
learning and arithmetic capability tracks, and release the first multi-seed emergence
corpus — checkpoints, circuit probes, negatives, and forecasting harness — as a public
benchmark. Every claim follows our established discipline: pre-registered predictions
and kill criteria, commit-stamped before data exists. Deliverables: the public corpus,
the benchmark harness, and the flagship paper.
