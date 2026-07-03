# Literature notes & positioning
Project: "Structure-specific representational priors causally control the grokking delay"
Date compiled: 2026-07-01

## The three must-reads (position explicitly against each)

### 1. Radial Suppression (arXiv:2606.32000, Jun 30 2026)
"Radial Suppression Accelerates Algorithmic Generalization: A Geometric Analysis of Delayed Generalization"

- Intervention: soft norm penalty on hidden reps `L_norm(h) = (1/d)(||h||2 - sqrt(d))^2`, constraining
  reps to a sqrt(d)-radius hypersphere. lambda=0.05 (MLP), 0.01 (nanoGPT).
- Setup: (a+b) mod P, P in {97,137,211}, 2-layer MLP (d=512) + 2-layer transformer (d_model=128, 4 heads),
  train frac 0.5, AdamW lr=1e-3 wd=1e-3, batch 256, 5 seeds.
- Results: 6.3x speedup (MLP P=97: 15,540 -> 2,460 epochs), 1.5x (transformer), 2.3x (nanoGPT 3-digit add).
  Fourier coherence reached 14x earlier; effective rank INCREASES (135->443); Hessian trace 30x lower.
- Mechanism claims: CE drives "radial inflation"; penalty redirects gradient angularly -> faster Fourier
  feature assembly; implicit ANISOTROPIC weight regularization (∝ input variance); curvature reduction.
- Controls: vs strong WD (1e-1), vs MAN, LayerNorm interaction, lambda/radius/frac/moduli sweeps.
- Stated limitation (their words): evidence is CORRELATIONAL among mechanisms; cannot isolate which
  mechanism drives acceleration. All interventions are STRUCTURE-AGNOSTIC (pure geometry).

**Our positioning:** Radial suppression is a structure-agnostic geometric intervention — it cannot tell
whether acceleration comes from generic geometry (norm control, curvature) or from earlier formation of
task-specific structure. Our design decouples these: SupCon on L2-normalized reps carries a built-in
radial-suppression component (cosine similarities are radius-blind), and the shuffled-structure control
keeps that geometric component identical while destroying the structural content. If true-structure
accelerates and shuffled does not (or hurts), the lever is the *structure*, not the geometry. This
directly answers their stated limitation.

### 2. Two Speeds (arXiv:2605.27078, May 26 2026)
"Two Speeds of Learning: A Representation-Readout Decomposition of Grokking and Double Descent"

- Method: decompose f(x) = W · phi_theta(x) (readout vs encoder); track critical dimension N_crit,
  manifold geometry (GLUE: D, R, rho_c, rho_a), linear-probe gap, last-layer NTK-label alignment
  (train-vs-test difference).
- Setup: modular addition p=113, S5 composition, sparse parity k=3 n=40, small-sample MNIST; MLPs +
  transformers, 3 seeds.
- Claims: representation learning is slow and CONTINUOUS (not silent, not a phase transition); readout is
  fast and train-biased early; grokking = readout direction equalizing as representation quality catches
  up. 20-80% of N_crit drop occurs BEFORE grok onset.
- Limitation (their words): purely observational on canonical tasks; decomposition only at final layer;
  "incorporating task-specific structure ... could reveal more geometric signatures" = future work.

**Our positioning:** Two Speeds predicts that the grokking delay is set by the slow representation
timescale. That account is observational; ours is the interventional test. If delayed generalization is
caused by slow representation-structure formation, then directly accelerating class-structured
representation formation (SupCon-aux with true positives) must shrink the delay, and accelerating
formation of the WRONG structure (shuffled) must not. We also adopt their framing to interpret probes:
cluster formation timing (our metric) is the representation speed; epochs-to-generalization is the
readout catching up.

### 3. Weight-Norm Causal Delay Law (arXiv:2606.13753, Jun 11 2026)
"The Weight Norm Sets the Grokking Timescale: A Causal Delay Law"

- Intervention: sustained per-step projection of TOTAL weight norm ||W|| to rho * ||W||_c after t_int;
  AdamW moments not reset. 2-layer MLP d=128 H=256, (a+b) mod p, p in {29..97}, frac 0.30-0.65 (primary
  0.40), full-batch AdamW, wd=1.
- Law: T_grok ∝ exp(alpha * rho), alpha ≈ 7.5 (MLP), 15.45 (no-LN attention); valid rho in [0.85,1.25].
  Network groks AT WHATEVER NORM it is held at; ||W||_c is just where free weight decay lands, CV 1-2%,
  ∝ p^0.38. Norm moves delay ~19x; lr only ~2x.
- KEY CONTROL FACT: LayerNorm DESTROYS norm-dependence (total norm not concentrated; clamping any group
  gives <=2x effects). => to make a norm-matched control meaningful, use an architecture WITHOUT LayerNorm.
- One-shot rescale does ~nothing (<=1.2x); only sustained clamps matter.
- Related: arXiv:2606.18465 "What Does the Weight Norm Control in Grokking? Logit-Scale Mediation under
  Cross-Entropy" (Jun 16) — norm acts through logit scale / softmax saturation. Track logit scale too.

**Our positioning:** The strongest deflationary objection to any grokking-acceleration result is "your
loss just changed the weight-norm trajectory, and the norm sets the timescale." We kill this with a
weight-norm-matched control: replay the per-step total-norm trajectory recorded from the SupCon-true run
onto a standard-CE run (sustained projection, matched init/seed/data-split, moments preserved). If norm
were the mediator, the replay should reproduce the acceleration. We use a no-LayerNorm 1-layer
transformer (Nanda-style) so the norm intervention is functionally meaningful per their LayerNorm caveat.
We also log logit-scale trajectories to address 2606.18465's mediation account.

## Supporting canon

- Power et al. 2022, "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
  (arXiv:2201.02177) — the phenomenon; transformers on modular arithmetic.
- Nanda et al. 2023, "Progress measures for grokking via mechanistic interpretability"
  (arXiv:2301.05217) — 1-layer no-LN transformer, p=113, frac=0.3, AdamW lr=1e-3 wd=1.0 full-batch;
  Fourier multiplication algorithm; progress measures (restricted/excluded loss); grokking = gradual
  circuit formation + cleanup. Our architecture and Fourier probes follow this.
- Liu et al. 2022, "Towards Understanding Grokking: An Effective Theory of Representation Learning"
  (arXiv:2205.10343) — representation quality (structured embeddings) governs generalization; Goldilocks
  zone. Direct conceptual ancestor: we intervene on representation structure.
- Liu et al. 2023, "Omnigrok: Grokking Beyond Algorithmic Data" (arXiv:2210.01117) — large init norm +
  small-data => grokking; norm story precursor.
- Varma et al. 2023, "Explaining grokking through circuit efficiency" (arXiv:2309.02390) — gen circuit
  more "efficient" (more logit per unit norm); weight decay drives transition.
- Lee et al. 2024, "Grokfast: Accelerated Grokking by Amplifying Slow Gradients" (arXiv:2405.20233) —
  EMA low-pass filter on gradients amplifies slow (generalizing) component; ~50x acceleration claims.
  OUR COMPARISON BASELINE. Grokfast-EMA: g_hat = g + lamb * ema(g), alpha=0.98, lamb=2.0 typical.
- Khosla et al. 2020, "Supervised Contrastive Learning" (arXiv:2004.11362) — SupCon loss, L_out variant:
  for anchor i, positives P(i) = same-class in batch; L = sum_i -1/|P(i)| sum_{p in P(i)}
  log( exp(z_i.z_p/tau) / sum_{a != i} exp(z_i.z_a/tau) ), z = L2-normalized projection.
- NeuralGrok (arXiv:2504.17243) — learned gradient transformation accelerates grokking (optimizer-side,
  not representation-side).
- Also relevant recent: 2606.12966 "Circuit Synchronization Precedes Generalization" (Fourier circuit
  synchronization precedes grok by 500-3000 steps — supports representation-timing account);
  2605.09724 "Model Capacity Determines Grokking through Competing Memorisation and Generalisation
  Speeds"; 2606.08985 "Beyond Neural Collapse: Task-Intrinsic Geometry Governs Neural Representations in
  Modular Arithmetic" (cyclic rank-2 geometry from CE).

## Scoop check (2026-07-01)
No paper found that adds a contrastive/SupCon auxiliary loss on hidden representations during grokking
training. Nearest neighbors: Radial Suppression (geometric penalty on reps, structure-agnostic);
Feature Repulsion 2605.08119 (analysis of natural repulsion dynamics, not an intervention); NeuralGrok /
Grokfast (gradient-side). Niche open as of today.

## The contribution in one paragraph
Prior accelerators are structure-agnostic: they act on optimization (Grokfast, NeuralGrok), on parameter
norm (weight decay, delay law), or on representation geometry (radial suppression). Whether the grokking
delay is specifically the time to form *task-structured* representations — rather than generically
favorable geometry — has not been causally tested. We inject a structure-specific representational prior
(SupCon-aux whose positive sets encode the task's true equivalence structure) and compare against an
exactly-matched shuffled-structure prior with identical loss form, strength, and geometry. Structure
specificity is the contribution; the weight-norm-matched control and Grokfast baseline rule out
optimization/norm mediation.

## Design decisions locked
- Architecture: 1-layer transformer, d_model=128, 4 heads, d_mlp=512, NO LayerNorm (Nanda-style; makes
  norm-matched control meaningful).
- Task: (a+b) mod p, p=97 primary (9409 examples, fast on 3080); p=113 replication if time.
- Split: 30% train (strong delay), full-batch AdamW lr=1e-3, wd=1.0, betas (0.9, 0.98).
- SupCon: on mean-pooled or last-token hidden state, projection to 64-d, tau=0.1; lambda swept
  {0.1, 0.3, 1.0}. Positives = same c = (a+b) mod p (true) vs fixed random pseudo-classes of identical
  size distribution (shuffled).
- Thresholds: train-fit = 99% train acc; generalization = 95% test acc; delay = difference in epochs.
- Probes every N epochs: embedding Fourier power concentration (Gini + top-k fraction), CKA(t, final),
  class-cluster ratio (between/within class distance of test reps), total weight norm, logit scale.
- Seeds: >=5 per condition.
