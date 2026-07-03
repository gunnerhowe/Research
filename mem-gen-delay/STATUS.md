# Project status: COMPLETE (draft ready for author review)
Updated: 2026-07-02

## Deliverable
paper/main.pdf — 12-page arXiv-ready draft, compiles clean via tools\tectonic.exe.
"Structure-Specific Representational Priors Causally Control the Grokking Delay"

## Final dataset (80 runs, runs/grid/)
- baseline n=10: 10/10 grok, median t_gen 19,950 (range 5,500-37,550)
- supcon_true n=30 (λ∈{0.1,0.3,1.0} × 10 seeds): 22/30 grok; λ=1.0 median paired ratio 0.80,
  fastest grok 2,000 epochs (2.75x vs same-seed baseline)
- supcon_shuffled n=20: 0/20 grok (Fisher vs true p=1.3e-7)
- norm_matched n=15: 0/15 grok (p=1.9e-6); logit-scale saturation collapse
- grokfast n=5: 5/5 grok, ratios 0.43-0.85
- Representation-timing invariant verified 80/80 (see paper footnote): no grok before probes
  move; no completed Fourier rise without grok.

## Analysis
analysis/out/: results.csv, stats.json (incl. paired Wilcoxon + Fisher), figs 1-6.
Regenerate: $env:PRIMARY_LAM="1.0"; python analysis/analyze.py

## Before arXiv submission (user actions)
1. Confirm author name/affiliation/email in paper/main.tex (currently placeholder from session).
2. Decide on optional long-horizon (150k) reruns of censored runs (chip spawned; likely
   unnecessary — censoring handled conservatively).
3. Consider public code release (repo is self-contained; README has repro commands).
