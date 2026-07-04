# Phase 2 Spec — Crossing-density BUDGET as a PINN stabilizer (Catastrophic Divergence on CGLE)

> **OUTCOME (2026-07-04): kill condition #1 fired, with receipts — atlas card → RED.**
> Vanilla failure reproduced on the ASPEN benchmark (rel L2 0.867±0.016, 3 seeds) and on a
> BF-unstable chaotic testbed (1.31), but in every run, at every depth (30k and the full
> published 100k), under both input conventions, the failure was UNDER-oscillatory
> (propagation loss / decorrelation; crossing profiles below budget at all levels). The
> one-sided budget is verifiably inert on such failures (bit-identical trajectory on a
> never-activated seed; ≤0.016 shift when transiently activated); gradient damping likewise.
> The mechanism itself is validated in vitro (SIREN over-oscillation clamped to budget, test
> error improved on all seeds). Published as the standalone honest-negative paper in
> `paper2/` ("Validated Mechanism, Absent Pathology") rather than a Phase-1 section, with a
> reproducibility caveat on the premise's implementation-sensitivity and the crossing-profile
> failure-mode diagnostic as a standalone contribution. Successor direction: PHASE3 spec.

**Status:** resurrection of the transformative-YELLOW atlas card *"Wilson momentum-shell RG curriculum +
Kac–Rice stationary-point-density regularizer → Catastrophic Divergence"* (kill-checked 2026-07-03),
**gated on Phase 1** (the spectral-bias experiments in this repo). Same primitive as Phase 1, opposite
sign: Phase 1 *demands* crossing density (INRs under-oscillate); Phase 2 *forbids excess* (PINNs on
stiff/chaotic PDEs blow up in high frequencies). One mechanism, both directions.

**Do not start Phase 2 experiments until the gates below flip. Write no code against this spec except
the module-boundary note (§Code deltas), which Phase 1 already satisfies.**

---

## Gates (from Phase 1 results)

- **GATE A (hard):** Phase 1 exp1 shows the crossing-density loss *steers* the field's crossing profile
  in the demanded direction on at least one signal class, at stable training. If the loss can't move
  crossing density with the sign we ask for, the sign-flipped version is dead too → mark the atlas card
  RED with that finding and stop.
- **GATE B (soft):** Phase 1 exp2 (non-uniform samples) shows the loss behaves on scattered points.
  Not blocking — Phase 2 has its own scattered-point story (collocation points) — but if scattered-point
  estimates are wildly noisy, raise `n_levels`/batch before starting here.

## One-paragraph idea

PINNs on stiff/chaotic PDEs — canonically the complex Ginzburg–Landau equation (CGLE) — exhibit
**catastrophic divergence**: runaway high-frequency oscillation of the network field that the PDE
residual loss cannot self-correct. Existing fixes act on the **architecture** (adaptive spectral bases —
ASPEN; Fourier features) or the **schedule** (curriculum/seq2seq, causal weighting, frequency marching).
We act on the **loss**: penalize the network field's *expected level-crossing density* wherever it
exceeds the **physical crossing budget** of the true solution class — the Kac–Rice/Rice integrand,
evaluated by Monte-Carlo **at the collocation points themselves** (scattered, non-grid — exactly where
FFT-based spectral penalties are awkward), one-sided so legitimate physical high-frequency content is
untouched. This clamps the blow-up mode directly, in the spatial domain, differentiably, for the cost of
one gradient norm the PINN already computes.

## Why the original card narrowed, and what survives (be honest in the paper)

- **DROP the Wilson-RG curriculum framing entirely.** Coarse-to-fine/frequency-marching/seq2seq
  curricula for PINNs are established (Krishnapriyan et al. failure-modes; causal weighting
  [Wang & Perdikaris]; frequency marching). Do not claim that half. It died on kill-check.
- **The problem×equation has a named occupant: ASPEN** (Adaptive Spectral Physics-Enabled Network for
  Ginzburg-Landau Dynamics, arXiv 2512.03290) — an *architecture-level* adaptive-spectral fix. Our claim
  is **loss-level, architecture-agnostic**: it must be positioned as complementary AND compared
  head-to-head. Also cite Multi-Scale SIREN-PINN for perturbed GL (2601.08104).
- **What survives** (the sliver the kill-check left standing): no prior work penalizes **level-crossing
  density** as an anti-oscillation PINN regularizer. It is spatial-domain, FFT-free, defined pointwise
  at scattered collocation points, one-sided (budget, not match), and physically calibrated via Rice's
  formula. That is the entire novelty claim — make it precisely and no wider.

## Code deltas from Phase 1 (small — this is the point)

Reuse unchanged: `field_and_grad`, `gaussian_delta`, `crossing_density` (`src/kacrice/crossing.py`).

1. **`CrossingBudgetLoss`** (add to `crossing.py`): one-sided variant of `KacRiceLoss`:
   `L = mean_j w_j · relu( c_θ(u_j) − b_j )²`
   - Levels `u_j`: **fixed from the expected amplitude range of the solution class** (CGLE standard
     regime: |A| = O(1)) — NOT `make_levels` on GT values (there is no GT field in a PINN).
   - One-sided is mandatory: early in training the field is smooth (crossings *below* budget); a
     symmetric match would push oscillation *up* — the exact failure we're preventing.
2. **Budget source `b_j`** — three options, **ablate all three** (this is exp5's second axis):
   a. **Reference spectral simulation** (ETDRK4; adapt the KS integrator from `ornstein-dist` — the
      CGLE version is a ~20-line change). Strongest budget, but note honestly: it presumes a solver
      for the equation family (fine for method validation; the paper must say so).
   b. **Rice closed form from the physical spectrum**: mean-level crossing rate ≈ (1/π)·√(λ₂/λ₀) with
      λ_k the spectral moments of the target solution class — a budget from *physics*, no simulation.
      This is the elegant one; if it works, lead with it.
   c. **Scalar cap** (single global budget) — crudest; doubles as an ablation of how much the
      level-resolved structure matters.
3. **`pinn.py`** (new): CGLE residual (1D first: ∂ₜA = A + (1+ib)∂ₓₓA − (1+ic)|A|²A, periodic,
   chaotic regime — **use ASPEN's exact (b, c) setting for comparability**), standard MLP + Fourier-
   feature variants, collocation sampling.

## Experiments (continue repo numbering)

- **exp3_cgle_baseline** — reproduce the documented failure: vanilla PINN on chaotic-regime 1D CGLE
  diverges (ASPEN's premise: "a conventional MLP PINN is fundamentally incapable"). Metrics: relative
  L2 vs reference sim, per-band spectral error, loss-spike trace. **If vanilla does NOT fail in our
  setup, stop — the premise is off; re-scope before spending more.**
- **exp4_budget_rescue** — + `CrossingBudgetLoss` (budget source a). Question: does divergence convert
  to convergence? This is the headline result if yes.
- **exp5_null_control — the experiment that makes it a paper.** Same runs against (i) plain Sobolev/
  gradient-norm damping (`SobolevLoss` already in `losses.py`) and (ii) a global smoothness weight.
  The claim under test: the budget loss clamps *excess* crossings while **preserving the physically
  legitimate high-frequency bands** (CGLE's real spectrum), where uniform damping over-smooths.
  Metric: spectral fidelity in the populated bands vs stabilization. **If indistinguishable from
  Sobolev damping, the machinery adds nothing — report the null honestly and fold Phase 2 into a
  short negative section of the Phase 1 paper.**
- **exp6_baselines** — matched-budget head-to-head: curriculum/seq2seq, causal weighting, Fourier-
  feature PINN, ASPEN (reimplement only if faithful — cite-compare otherwise; SemRF/CABLE precedent:
  a faithful reimplementation strengthened that paper, a sloppy one would have sunk it).
- **exp7_composition** — ASPEN-style architecture + budget loss. Complementarity is the honest win
  condition against the occupant: architecture-fix and loss-fix should stack.
- **stretch** — transfer to KS (integrator already exists in `ornstein-dist`), showing
  equation-generality.

## Kill conditions

1. exp3: vanilla failure not reproducible → premise off, stop.
2. exp5: ≈ Sobolev damping → mechanism adds nothing; publish the null inside Phase 1's paper.
3. Works **only** with budget source (a) (reference sim) → method needs the solution class solved
   already; still publishable but demote the claim from "stabilizer" to "spectral-budget distillation."
4. ASPEN alone saturates the benchmark and composition adds ~0 → we are a redundant fix; report.

## Paper shape (decide after exp4+exp5)

- **Clean rescue + clean null-control separation** → standalone sequel paper aimed at the sci-ML/PINN
  audience ("A physical crossing-density budget stabilizes PINNs on stiff chaotic PDEs"), citing
  Phase 1 as the mechanism paper. Different reviewer pool than Phase 1's INR/vision audience.
- **Modest or mixed** → Act II / extension section of the Phase 1 paper ("one mechanism, both
  directions"), which is still a distinctive shape.

**Effort:** ~1 week on the 3080 after Phase 1 lands (crossing machinery reused; new code = CGLE
residual + reference sim + baselines). **Atlas card:** update `Catastrophic Divergence` on the board
with the outcome either way — it's currently YELLOW with full kill-check receipts; this spec is its
sanctioned resurrection path.
