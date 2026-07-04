# Phase 3 Spec — Differentiable Minkowski profiles of neural fields (topology at Monte-Carlo cost)

**Status:** YELLOW, headline-candidate, gated on validation. Preliminary kill-check run
2026-07-03 (three web angles; see §Occupants) — the narrow claim survived, but this spec has
NOT yet had the full gauntlet (local abstract index + exhaustive web + adversarial pass).
**That is the next reviewer's first job before any code is written.** Phase 1 (GREEN, complete,
arXiv-ready) and Phase 2 (complete modulo receipts; honest-negative outcome) both feed this.

**Do not start Phase 3 until (a) Phase 2's paper is finalized, (b) the full kill-check
confirms §Occupants is complete, and (c) GATE V below passes.**

---

## One-paragraph idea

A field's excursion sets {f ≥ u} carry exactly d+1 fundamental integral-geometric
descriptors (Minkowski functionals): in 2D — area M₀(u), boundary length M₁(u), Euler
characteristic M₂(u); in 3D — volume, surface area, integrated mean curvature, Euler
characteristic. Phases 1–2 built and validated the differentiable Monte-Carlo estimator for
**M₁** (level-crossing / level-set density via smoothed co-area) and used it in both
directions (demand → INR spectral bias; one-sided cap → PINN budget). Phase 3 completes the
family with **M₂ — topology — via the Gauss–Bonnet/co-area integrand**: a smooth, mesh-free,
complex-free, O(N)-per-batch estimator of the Euler-characteristic *profile* χ(u) of any
neural field, evaluated at scattered sample points with autograd derivatives only. Used as a
loss, it is training-time **topology control for neural implicit shapes without persistent
homology**: "this SDF has two spurious handles and a floating component — remove them," as a
cheap penalty term. The quantitative headline axis is COST: the incumbent (STITCH) builds a
cubical complex and runs persistent homology *every training iteration*; ours is one extra
autograd derivative on the batch.

## The mechanism (concrete)

2D, excursion set A_u = {x : f(x) ≥ u}, level curves ∂A_u. Gauss–Bonnet:
χ(A_u) = (1/2π) ∮_{∂A_u} κ_g ds, with κ_g the (signed) curvature of the level curve,
κ = div( ∇f / ‖∇f‖ ). Combining with the co-area factor (ds = δ(f−u)‖∇f‖ dx):

  **χ̂_ε(u) = (1/2π) · (V/N) · Σᵢ δ_ε(f(xᵢ) − u) · κ(xᵢ) · ‖∇f(xᵢ)‖**

where κ‖∇f‖ expands to (f_yy f_x² − 2 f_xy f_x f_y + f_xx f_y²)/‖∇f‖² — bounded wherever
∇f ≠ 0 on the level set; second derivatives via autograd (double backprop, same cost class
as Phase 2's PINN residual). 3D (the SDF case): χ(∂A_u) = (1/2π) ∫_{∂A_u} K dA with K the
Gaussian curvature of the level surface (expressible in f's Hessian and gradient; χ(A_u) =
χ(∂A_u)/2 for closed boundaries). M₀ is the trivially smoothed indicator mean; M₁ is
Phase 1's estimator, already validated to 5% (`src/kacrice/crossing.py`, tests).

**Loss family:** match, cap, or demand the profile vector (M₀(u_j), M₁(u_j), M₂(u_j)) at
L fixed levels — the same level/bandwidth/normalization machinery as Phases 1–2
(quantile-or-fixed levels, Gaussian δ_ε, relative normalization, one-sided variants). For
SDFs the natural target: χ(0-level) = known genus/component count of the shape class;
one-sided "no extra components/handles" caps for repair.

## Occupants (preliminary kill-check, 2026-07-03) — and the surviving sliver

- **Minkowski Image Loss** (arXiv 2604.11422, Apr 2026): differentiable area/perimeter/
  components losses — **on pixel grids** (sigmoid-relaxed morphology), weather domain.
- **DECT** (arXiv 2310.07630, ICLR 2024): differentiable Euler Characteristic Transform —
  **on discrete complexes** (point clouds/meshes/graphs), classification-oriented.
  Cite; do NOT claim "first differentiable EC."
- **STITCH** (arXiv 2412.18696): topology-constrained neural SDF fitting — **persistent
  homology on cubical complexes, per training step**. This is the head-to-head incumbent.
- **Persistent-homology losses** generally: established, expensive, non-smooth, complex-bound.

**Survives (the entire novelty claim — state it this narrowly):** smooth Monte-Carlo
estimation of Minkowski/EC *profiles* of **continuous neural fields** at **scattered
points** via the Gauss–Bonnet/co-area integrand, differentiable in θ, no grid, no complex,
no filtration — and its use for **training-time topology control at a fraction of PH cost**.
The reviewer must verify no occupant exists for THIS formulation (search angles: "Gaussian
kinematic formula loss", "Euler characteristic density estimator neural", "level set
topology regularization implicit", "differentiable integral geometry", cosmology ML
Minkowski emulators, and the citation graphs of the three occupants above).

## Gates (from Phases 1–2, plus new)

- **GATE V (hard, new):** χ̂_ε matches exact Euler characteristics of analytic test fields
  (random trigonometric fields and Gaussian-bump mixtures with countable, hand-verifiable
  excursion topology across levels) within ~10–15% at N ≤ 2×10⁵ samples, in 2D AND 3D.
  The known risks it tests: κ's 1/‖∇f‖² near critical points (which sit exactly where
  topology changes — the estimator is integrable through them in expectation, but variance
  may be brutal); boundary corrections (validate on periodic domains first; finite domains
  need the geodesic-curvature boundary term — implement or dodge via padding, but decide
  honestly); δ_ε bias near critical levels where χ(u) jumps.
- **GATE S (hard):** the M₂ loss steers: gradient descent on a toy field moves χ̂ toward a
  target profile without destroying M₀/M₁ (the Phase-2 in-vitro pattern — build the positive
  control FIRST this time; Phase 2's lesson is that the pathology, not the tool, is the
  usual missing ingredient).
- **Carried over:** Phase 1 GATE A/B analogues hold for M₁ (validated); the ε/L/β
  machinery, relative normalization, and one-sidedness proofs transfer unchanged.

## Experiments (continue repo numbering; order of decisiveness)

- **exp8_validation** — GATE V. Publishable as a figure regardless of outcome.
- **exp9_invitro** — GATE S: 2D implicit-curve fit from sparse samples engineered to
  produce spurious components; show the one-sided χ-cap removes them where curvature/
  smoothness regularizers don't. (Positive control before any in-vivo claim.)
- **exp10_sdf — THE DECISIVE ONE.** Neural SDF from noisy/sparse point clouds (shapes with
  known genus: sphere, torus, double-torus, plus a thin-structure stress case). Configs:
  Eikonal-only; +curvature/smoothness null controls; +χ-profile cap (ours); +STITCH-style
  PH loss (faithful reimplementation on cubical complexes — the CABLE/SemRF precedent:
  reimplement faithfully or cite-compare, never sloppily). Metrics: Betti numbers of the
  extracted mesh vs GT (topology correctness), Chamfer-L2 (geometry preserved), and
  **wall-clock per-iteration overhead** — the cost axis is the headline claim; measure it
  with the Phase-1 timing-artifact discipline (script + JSON in repo, no ad-hoc numbers).
- **exp11_ablations** — ε, L, N, curvature clamping, 2D vs 3D estimator variance.
- **stretch** — Minkowski-profile losses for stochastic-field emulation (cosmology/weather
  style): match all three profiles of a target ensemble; positions against 2604.11422's
  grid-bound loss on its own turf, off-grid.

## Kill conditions

1. GATE V fails (variance/bias unusable at practical N) → RED, stop; publish the estimator
   analysis as a short note if the failure mode is instructive.
2. exp10: topology repair no better than cheap curvature/smoothness regularizers → the
   topology term adds nothing over geometry; report the null (Phase-2 shape).
3. PH baseline matches ours at comparable wall-clock → redundant fix; the cost claim dies
   and with it the paper's spine; report honestly.
4. Works only at grid-scale sample counts (N ≳ voxel counts) → structural claim collapses
   to parity with cubical methods; fold into a methods note.

## Honesty rails (Phase-1/2 lessons, binding)

- Phase 1's own result pattern — "mesh-free structural advantage" hypotheses tend to land
  at PARITY on accuracy — is the prior. The win condition here is therefore pinned to the
  **cost axis and the no-complex axis**, which are measurable, not vibes.
- Do not claim "first differentiable topology" (DECT, PH losses exist). Do not claim
  cosmology impact without running the stretch. Positional sentence: "the first
  complex-free, training-time Euler-characteristic control for continuous neural fields."
- Pre-register these kill conditions in the paper as Phases 1–2 did; the honest-reporting
  pipeline (incremental JSON, resume-safe runners, figures-from-raw-logs) carries over
  unchanged (`experiments/run_paper.py` / `run_phase2.py` as templates).

## Code deltas (small; the point of the program)

Reuse unchanged: `field_and_grad`, `gaussian_delta`, `crossing_density`,
`CrossingBudgetLoss` normalization patterns (`src/kacrice/crossing.py`).
New: `minkowski.py` — `curvature_density` (κ‖∇f‖ via autograd Hessian-vector terms),
`euler_profile` (2D/3D), `MinkowskiProfileLoss` (match/cap/demand modes);
`sdf.py` — point-cloud SDF fitting harness (Eikonal + supervision), marching-cubes
extraction + Betti-number computation for evaluation only (evaluation may use meshes;
the LOSS never does).

**Effort:** GATE V ~1 day; exp9 ~1 day; exp10 ~3–5 days on the 3080 (SDF fits are small
MLPs; PH baseline is the slow part — budget it). **Paper shape:** standalone
("Differentiable Minkowski Functionals for Neural Fields") if exp10 delivers topology
repair at ≥10× PH cost advantage; otherwise methods-note + fold into the program line.
**Atlas:** new card; link Phase 1 (M₁ demand), Phase 2 (M₁ cap, honest null), Phase 3 (M₂).
