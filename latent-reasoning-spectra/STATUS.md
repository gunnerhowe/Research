# Project status: COMPLETE — honest negative with strong characterization (all deliverables shipped)
Updated: 2026-07-09

## Outcome (all pre-registered in PLAN.md, committed before results)
- **K1 FIRES**: spectral invariants of the latent step map do NOT predict reasoning
  structure. sigma1 branch AUROC M2 0.554 [0.523,0.585] vs pause nulls 0.556/0.634
  (nulls equal/higher; separation 3.25x/25.4x LARGER in nulls); pruned twins
  unresponsive (MWU p=0.71); within-step stratification collapses everything incl.
  trivial baselines to chance; valid split replicates (0.538, CI brackets 0.5);
  AUROC flat across curriculum while accuracy goes 0->99%.
- **K2 does NOT fire**: v1(J_t) causally privileged 3.3x/7.0x vs random
  (p~1e-80/1e-96); eigendirections NOT privileged (non-normal signature). Absolute
  effects tiny: 0 flips anywhere.
- **K3 formal criterion not met** (CIs exclude 0.5) but signal shown positional;
  routedness: local chain loses ~half the influence by k=3 (median R up to 0.51).
- **Headline discovery (post-hoc probe, declared in PLAN 6b)**: transplanting ALL
  SIX thoughts from a random donor problem changes 0/500 answers (zeros: 4/500).
  The recycled content is collectively inert — the latent segment contributes
  structure, not state. First operator-side confirmation of curriculum-is-mechanism.
- Characterization: step maps contracting (rho 0.34) + highly non-normal (kappa ~9);
  thought rides its own top-4 eigen-subspace (91% of norm).

## Deliverables
- paper/main.pdf — 14 pp, "Contracting, Non-Normal, and Not the Mechanism: A
  Pre-Registered Spectral Audit of Latent Chain-of-Thought Reasoning"; 202
  auto-generated macros (paper/numbers.tex, verify_regen.py passes), 7 figures.
- paper/arxiv_abstract.txt ready.
- results/*.json committed (sanity, exp0_gate, exp1_causal, exp2_koopman,
  exp3_robustness, exp_posthoc_allablate).
- src/lrspec + tests (11 pass), experiments/, PLAN.md pre-registration, README
  repro commands, laymen.md, lit_notes.md (recency sweep 2026-07-08: cell open).

## Before arXiv submission (user actions)
1. Confirm author name/affiliation/email in paper/main.tex.
2. Re-run the recency sweep (lit_notes.md queries) on submission day — field ships
   ~weekly; sweep was clean as of 2026-07-08.
3. Optional: Zenodo code bundle + arXiv source zip (house pattern from doob-synapses).
4. Atlas link-back (per info.txt): on ship, tell the atlas assistant to ingest as
   own-lab node, outcome = HONEST NEGATIVE with characterization (K1 fired; K2 not;
   thought content collectively inert; routedness ~0.5 at k=3).

## Environment notes for future sessions
- C: hit 100% (HF cache ~71GB is mostly OTHER projects' models — untouched); this
  project's HF cache entry deleted after copying to models/; epoch checkpoints
  cached at E:\hf_cache_lrspec.
- GPU co-tenancy: use chunked VJPs (LRSPEC_JAC_CHUNK=96); full 768-row batched
  backward spills VRAM and is ~40x slower under contention.
- transformers 4.55 GPT-2 returns legacy tuple KV; harness handles both. M1 CoT
  eval: max_new_tokens=128, vocab 50257.
