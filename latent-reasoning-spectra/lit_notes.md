# Literature notes + recency sweeps

> **Pre-submission checklist:** re-run the sweep queries below on submission day.
> Last clean sweep: 2026-07-08 (cell open). The 2606 Coconut-analysis cluster ships
> ~weekly and the SPAR Spring-2026 project (mentor Uzay Macar) had no output yet.

## Recency sweep 2026-07-08 (pre-writing; live web)

Queries: "Koopman latent reasoning chain of thought 2026", "spectral Coconut continuous
chain-of-thought interpretability Jacobian", "dynamical systems chain of thought hidden
state trajectory eigenvalue June 2026" (newest-first).

**Verdict: the cell is still open.** No paper found that characterizes the latent-thought
autoregressive map c_{t+1}=F(c_t) spectrally, or predicts anchor/branch structure from
dynamical invariants with DAG validation. Nearest hits, all checked:

- arXiv:2604.04902 "Are Latent Reasoning Models Easily Interpretable?" — token ablation +
  decoding of latent traces (65–93% recoverable); NO spectral/Jacobian/Koopman machinery,
  no DAG prediction. Complementary, cite as recent latent-interp activity.
- arXiv:2602.01196 "Unraveling the Hidden Dynamical Structure in Recurrent Neural
  Policies" — local Jacobian eigenspectra along trajectories, but for RL policies (control
  tasks), not language latent reasoning. Method-adjacent; different domain and no
  reasoning DAG / causal validation.
- arXiv:2509.25239 formal CoT-vs-latent-thought comparison (theory, no dynamics).
- ReLaX (arXiv:2512.07558): confirmed phantom (RL latent exploration; no Koopman/DMD).

## Verified load-bearing citations (fetched abstracts 2026-07-08)

- **Coconut** arXiv:2412.06769 — substrate. Hidden-state recycling, ProsQA.
- **Aswal et al.** arXiv:2606.12689 "Observable Patterns Are Not Explanations: A
  Causal-Geometric Analysis of Latent Reasoning Models" (Aswal, Palmeira Ferraz, Zhou,
  Peyrard). Coconut+CODI vs matched controls; BFS-like frontiers/decodable patterns appear
  in controls too; causal effects concentrate in LOW-RANK directions, step-to-step geometry
  grows structured with influence; thesis: patterns ≠ mechanism, need matched controls +
  causal tests. => our discipline gate; adopt, don't fight.
- **Li et al.** arXiv:2602.08783 "Dynamics Within Latent Chain-of-Thought: An Empirical
  Study of Causal Structure" (Li, Bai, Chen, Li, Yang, Lin, Zhang). SCM + step-wise
  do-interventions; finds staged functionality with NON-LOCAL ROUTING; anchors exist
  causally. => phenomenon established here; also the routed threat behind our E2/K3.
- **Fernando & Guitchounts** arXiv:2605.14258 — full Jacobian eigendecomposition of the
  residual stream, DEPTH-as-time, production LLMs; monotonic spectral gradient through
  depth, non-normal rotation-dominated early layers. NO CoT/Coconut/latent recurrence.
  => same toolkit, different dynamical object; our single most important distinction.
- **RKSP** arXiv:2602.22988 — whitened DMD over LAYER-wise residual snapshots; "near-unit
  spectral mass" diagnostic predicts training divergence (AUROC 0.995); Koopman spectral
  shaping. => borrow near-unit-mass idea (credited); depth axis + training-stability goal,
  not reasoning interpretability.
- **Carson & Reisizadeh** arXiv:2506.04374 — SDE at sentence/token level, PCA+GMM regimes;
  discrete tokens, no Jacobian/Koopman/DAG.
- **Sun et al.** arXiv:2604.05655 — discrete-CoT trajectory geometry, step-specific
  subspaces, mid-trajectory correctness AUROC 0.87; no spectral machinery. Source of
  validation-metric style.
- **Curriculum checkpoints** — bmarti44/coconut-curriculum-checkpoints (HF): "The
  Curriculum Is the Mechanism: Dissecting COCONUT's Latent Thought Gains on ProsQA."
  M1 CoT 83.0 / M2 Coconut 97.0 / M3 pause-curriculum 96.6 / M4 pause-multipass 94.8 on
  ProsQA test (McNemar M2-vs-M3 p=0.845). Pause models = built-in matched null controls.
- Attractor/equilibrium reasoner architectures (2605.12466 "Solve the Loop", 2605.21488,
  2606.18206, 2605.18820): design fixed-point dynamics; we analyze emergent dynamics.
  Cite-and-distinguish (design vs analysis).
- SPAR Spring-2026 "Interpreting latent reasoning" (mentor Uzay Macar): no published
  output found as of 2026-07-08. The clock.

## Notes for the paper's distinguishing statement

Prior work: (i) anchors + routed effects established causally (2602.08783); (ii) low-rank
causal-direction geometry (2606.12689); (iii) Jacobian/Koopman spectra on the DEPTH axis
(2605.14258, 2602.22988). We: FIRST spectral characterization of the latent-reasoning
autoregressive map c_{t+1}=F(c_t) along the THOUGHT axis, with invariants that PREDICT
anchor/branch structure, validated on ProsQA's known DAG with matched pause-token +
pruned-linear-chain nulls and confirmed by causal interventions (their own discipline).
