# Project status: E0-E2 COMPLETE (K1 fired — honest negative with characterization); E3 tail running
Updated: 2026-07-09 ~03:15

## Verdicts (pre-registered in PLAN.md, committed before results)
- **K1 FIRES** (null-control failure): sigma1 branch AUROC M2 0.554 [0.523,0.585] vs
  pause nulls M3 0.556 / M4 0.634 (nulls equal or HIGHER); per-problem branch
  separation LARGER in nulls (3.25x / 25.4x); pruned twins unresponsive (MWU p=0.71,
  twins solved at 97.5%); within-step stratification collapses ALL predictors incl.
  trivial baselines to chance (sigma1 0.492, |dc| 0.485). Spectral structure is
  architectural/positional, not reasoning-specific.
- **K2 does NOT fire**: v1(J_t) perturbations 3.3x/7.0x random at eps 0.1/0.3
  (p~1e-80/1e-96); eigendirections NOT privileged (non-normal signature). BUT absolute
  effects tiny: zero answer flips anywhere; single-thought mean-replacement median
  |dmargin| 0.002 vs margins ~13.9 (3,000 ablations, 0 flips). Thoughts locally
  near-inert.
- **K3 formal criterion not met** (AUROC CIs exclude 0.5) but the residual signal is
  the step-position confound; reported with both letter and spirit. Routedness: local
  Jacobian chain loses ~half the influence by k=3 (median R 0.44-0.51); pooled EDMD
  has near-unit persistent modes, no reasoning signal.
- Valid-split replication: sigma1 0.538 raw / 0.481 stratified (matches test).

## Headline characterization (the positive content)
Step maps strongly contracting (mean rho 0.34) yet highly non-normal (sigma1 2.97,
kappa ~9, Henrici ~0.96); thought vector rides its own top-4 eigen-subspace (91% of
norm); causal signal concentrated in non-normal amplification directions; influence
routed through attention KV. Dynamical vindication of curriculum-is-the-mechanism.

## Running (single background chain)
- E3 epoch sweep (checkpoints download to E:\hf_cache_lrspec after C: filled), E3
  analyze, post-hoc all-thoughts ablation probe (declared in PLAN 6b).

## Remaining
- E3 section + epoch numbers into paper; final make_figures + verify_regen; final
  compile; laymen.md done; final recency sweep note; atlas link-back note.

## Gotchas / environment notes
- C: drive hit 100% during epoch-checkpoint downloads (HF cache ~71GB, mostly OTHER
  projects' models — Qwen/Mistral/Nemotron/Phi — untouched). Deleted only
  models--bmarti44--coconut-curriculum-checkpoints (3.5GB, already copied to models/).
  HF_HOME for this project's remaining downloads: E:\hf_cache_lrspec.
- GPU shared with concurrent sessions (qwen7b eval, exp0_existence.py,
  faithful-selection job). Full 768-row batched Jacobian spills VRAM under co-tenancy
  (40x slowdown) -> chunked VJPs (LRSPEC_JAC_CHUNK=96). One transient CUDA
  illegal-memory-access absorbed by capture resume + run_all multi-retry.
- transformers 4.55 GPT-2 returns legacy tuple KV caches; harness handles both.
- M1 CoT eval needs max_new_tokens=128 (64 truncates 5-6-hop answers) and has vocab
  50257 (no special tokens); still 6pp below published 83.0 under our protocol —
  quoted with caveat, context-only.

## Deliverable state
paper/main.pdf compiles (228 KiB, 6 figures, 177 auto-generated macros); results/
JSONs committed for sanity + E0 + E1 + E2.
