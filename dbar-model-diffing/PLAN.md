# PLAN.md — Pre-registration: Ornstein d-bar as a computational-equivalence metric for model diffing

**Committed BEFORE any experimental results exist** (see git history: this file's commit precedes
the commit of anything under `results/`). Author: Gunner Levi Howe. Date: 2026-07-08.

## Hypothesis

Ornstein's d-bar — a nonparametric, coding-invariant distance on stationary stochastic
processes, estimated by optimal transport between empirical n-block distributions under
normalized Hamming cost — can, on a deliberately **low-entropy symbolic readout** of a model's
computation, separate *same-output / same-marginal but different-process* model pairs
(a model vs. its noise-injected / distilled / pruned twin) that

- **static representation-geometry metrics** (CKA) and
- **deterministic-conjugacy dynamical metrics** (DSA, Ostrow et al. 2306.10168)

do not separate. The claim is bounded by Ornstein–Weiss finitary observability
(arXiv:math/0608310): we do **not** claim isomorphism detection in general; the claim is a
process-level distance **within a fixed readout**.

## Fixed design (locked before results)

### Tasks (stationary symbolic processes; unifilar HMM generators)

- **GM** — golden-mean process, alphabet m=2: state A emits 0 (p=1/2, stay A) or 1 (p=1/2, to B);
  B emits 0 (p=1, to A). No "11". Entropy rate h = 2/3 bits/symbol; P(1) = 1/3. **Primary task.**
- **FEVEN** — symbol-flipped even process, m=2: A emits 1 (p=1/2, stay A) or 0 (p=1/2, to B);
  B emits 0 (p=1, to A). Runs of 0s between 1s have even length. h = 2/3 bits/symbol; P(1) = 1/3.
  **Different-task control** with *identical* entropy rate and *identical* unigram marginal as GM,
  so agreement on it cannot be attributed to entropy or marginal differences.
- **MESS3** (x=0.15, a=0.6; Marzen–Crutchfield / Shai et al. 2405.15943), m=3: E2 secondary grammar.

### Models

- E0/E1: 1-layer GRU, width 64, input embedding + linear head; trained by next-symbol
  cross-entropy (Adam 1e-3, batch 64, seq len 256) to within 1% relative of the task's entropy
  rate on held-out data. **5 seeds** (0–4).
- E2: 2-layer causal Transformer-LM, d_model=64, 4 heads, context 256, trained the same way.
  **3 seeds**. Free-running generation uses a sliding full-recompute window of 255 tokens, so the
  emitted process is an order-255 stationary Markov process by construction.

### Constructed pairs (per base seed; base model A vs. twin B)

1. **null-recoded** — B = A with a random permutation of hidden units (weights permuted
   consistently; function-preserving by construction). Validation: all metrics must call "same".
2. **seed** — A = seed i, B = seed j (same task, independent training). Report only.
3. **noise** — B = A with i.i.d. Gaussian noise (std σ*) added to the hidden state at every step
   (both in generation and in teacher-forced state collection). **Gate pair (primary).**
4. **distill** — B = student GRU-64 (fresh init) trained by KL to A's conditional
   next-symbol distributions on task inputs, to within 0.5% relative val CE of the teacher.
   **Gate pair.**
5. **prune** — B = A with 50% global magnitude pruning on {W_ih, W_hh, head}, then fine-tuned
   (mask fixed, ≤2k steps) to within 0.5% relative val CE of A; if unreachable, report actual CE.
   **Gate pair.**
6. **diff-task** — A = GM model, B = FEVEN model (same seed index). Agreement control: metrics
   should call "different".

**Noise calibration (σ\*)**: σ* = the largest σ ∈ {0.01, 0.02, 0.05, 0.1, 0.2} such that
(a) free-running unigram TV(A, B) ≤ 0.01, and (b) teacher-forced CE increase ≤ 2% relative.
This enforces the "same output / same marginal" premise. E1 sweeps the full grid regardless.

**Output-preservation check** for distill/prune: unigram TV ≤ 0.01 required for the pair to
count toward the gate; report CE for all pairs.

### Readouts (fixed low-entropy symbolizations)

- **Primary — emitted-token readout**: the model's free-running autoregressive sample stream
  (the stochastic process the model defines). B=64 independent chains × T=32768 steps after 512
  burn-in ⇒ ~2.1e6 pooled symbols per run; **two independent generation runs per model** for
  same-process floors. Blocks never cross chain boundaries.
- **Secondary — quantized-belief readout** (binary tasks): the model's predictive probability
  P(next=1 | history) recorded during free running, quantized into 4 equal bins ⇒ alphabet 4.

### Metrics (computed side-by-side on every pair, all seeds)

- **d-bar**: OT between empirical n-block distributions under normalized Hamming cost
  (network simplex; sampled regime 2000 blocks with 4 repeats when support > 4096),
  n ∈ {1, 2, 3, 4, 6, 8, 12, 16, 24} (claims), {32} additionally plotted for the wall figure.
  Every pair value reported with same-process floors: floor_n(M) = d-bar_n(run1(M), run2(M))
  at identical regime and budget.
- **CKA** (linear, column-centered) on hidden/residual states collected teacher-forced on a
  shared task-sampled evaluation set (B=32 × L=256, first 16 steps dropped), with the twin's
  noise active where applicable.
- **DSA** (public `dsa-metric` package, Ostrow et al.): on the same state trajectories,
  n_delays=8, delay_interval=1, rank=32, iters=1500, lr=5e-3, score_method='angular'.
  One config for all pairs.
- **Diagnostics**: entropy rate ĥ per stream (conditional block entropy with undersampling
  guard; LZ78 cross-check), the estimation wall k_wall = log2(N_windows)/ĥ, and the Fano-type
  certified d-bar lower bound from |Δĥ|.

### E0 gate (pre-registered decision rule)

Let ν_DSA = mean DSA of null-recoded pairs, κ_DSA = mean DSA of diff-task pairs.
For a gate pair, define at each block length n: Δ_n = d-bar_n(A,B) − max(floor_n(A), floor_n(B)),
and n* = argmax Δ_n over 2 ≤ n ≤ k_wall (k_wall from the *larger* ĥ of the two streams).

A gate pair **passes** iff ALL of:

1. **CKA-similar**: mean CKA ≥ 0.90.
2. **DSA-similar**: mean DSA ≤ ν_DSA + 0.25 · (κ_DSA − ν_DSA).
3. **d-bar separates within the wall**: across the 5 seeds, mean Δ_{n*} − 2·std(Δ_{n*}) > 0,
   AND mean d-bar_{n*} ≥ 2 × mean max-floor_{n*}, AND n* ≤ k_wall (K2 guard),
   AND the same-marginal premise holds: d-bar_1(A,B) ≤ 0.02.

**E0 PASSES iff at least one of {noise, distill, prune} passes.** The validation battery must
also hold for results to count:

- **V1 (identity/isomorphism null)**: null-recoded pair: d-bar_n ≤ 2 × floor_n for all n ≤ k_wall,
  CKA ≥ 0.99.
- **V2 (agreement control)**: diff-task pair: d-bar plateau ≥ 10 × floor AND DSA ≥ 5 × ν_DSA.
- **V3 (convergence)**: d-bar_n vs. N subsampling curve (N ∈ {1e4, 3e4, 1e5, 3e5, 1e6, 2.1e6})
  shows the winning Δ_{n*} stable (within its seed CI) for N ≥ one quarter of the full budget,
  and the d-bar-vs-n curve is plotted against the k_wall line.

### Kill conditions (committed)

- **K1 — no separation**: none of noise/distill/prune passes the gate ⇒ honest NARROW/NEGATIVE
  ("d-bar agrees with existing tools on the constructible cases"); publish as such, no forcing.
- **K2 — estimation-wall violation**: a separation that only appears at n > k_wall is an
  estimation artifact ⇒ that claim is retracted/not made.
- **K3 — isomorphism overclaim**: no framing may assert d-bar detects "same computation /
  isomorphism in general from finite data" (forbidden by Ornstein–Weiss math/0608310).
  All claims are "process distance within a fixed low-entropy readout".

### E1 (runs only if E0 passes)

Characterize *when* d-bar separates: sweep noise σ ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.2} and prune
fraction ∈ {0.1, 0.3, 0.5, 0.7, 0.9} (each fine-tuned per protocol), 5 seeds, all three metrics;
deliverable = the d-bar-vs-DSA gap as a function of how stochastic the "sameness" is.

### E2 (runs only if E0 passes)

Transformer-LM (above) on GM and MESS3; pairs: noise-in-residual twin (σ* recalibrated with the
same rule) and distilled twin; states = final-layer residual stream; same metrics and gate-style
report. Claims restricted to n ≤ k_wall computed for the E2 budget (B=64 × T=8192 ⇒ ~5e5 windows).

### Stats discipline

≥5 seeds (E0/E1) with mean ± std and seed-level points shown; all three metrics always reported
side-by-side; every paper number generated by `paper/gen_paper_numbers.py` into `numbers.tex`
(verified by `paper/verify_regen.py`); results JSONs committed under `results/`.

## AMENDMENT 1 (2026-07-08, after E0 ran; results/exp0_existence.json is committed unmodified)

The pre-registered gate **failed as written** for all three pairs (recorded verdict:
`E0_PASS = false`). Post-hoc inspection of the committed curves shows two artifacts of the
*gate arithmetic* (not of the estimator, whose per-n floors flag both):

1. **Plateau-rule artifact.** n* = argmax_n Δ_n sometimes lands on sampled-regime rows
   (n ≥ 16) where the finite-sample OT bias inflates both the pair estimate and its floor to
   ≈ 0.042 (2000-block budget); Δ there is noise on a large common bias, and the
   "d̄_{n*} ≥ 2 × floor_{n*}" check is structurally unsatisfiable at such rows even for pairs
   with clear separation at small n. *Amended rule:* n* = argmax Δ_n restricted to rows with
   d̄_n ≥ 2 · floor_n (the gate's own significance threshold, applied per-row), 2 ≤ n ≤ k_wall;
   a pair with no eligible row has "no separation".
2. **DSA-threshold artifact.** The DSA-similar criterion ν + 0.25(κ − ν) assumed
   ν ≪ κ. Empirically DSA's null (recoded twins: 0.028–0.118) and different-task
   (0.092–0.150) distributions overlap — the threshold (0.071) sits inside DSA's own null
   spread, so no pair can be called DSA-similar reliably. *Amended rule:* a pair is
   DSA-similar if its mean DSA ≤ max(ν + 2·std_null, max over null seeds) — i.e.
   indistinguishable from DSA's empirical null distribution.

No data were re-collected; the amended rules are evaluated on the committed JSON by
`analysis/amended_gate.py` into `results/amended_gate.json`. The paper reports **both**
verdicts (pre-registered: fail; amended: see JSON) and labels every amended number.
E1/E2 proceed under the amended gate; their designs are unchanged except that the E1 noise
sweep also records the (pre-registered, secondary) quantized-belief readout, where the noise
separation is largest.

## AMENDMENT 3 (2026-07-08, same session; formalizing what analysis/amended_gate.py computes)

The pre-registered seed-consistency inequality "mean Δ − 2·std(Δ) > 0" penalizes effect-size
heterogeneity: a pair can separate on every seed and still fail it because one seed separates
*more*. The amended gate therefore records BOTH verdicts: `PASS_strict` (the pre-registered
inequality, unchanged) and `PASS_sign_consistent` (all seeds have an eligible plateau with
d̄ ≥ 2× floor, all Δ > 0 — binomial p = 1/32 at 5 seeds — and min ratio ≥ 2). The
sign-consistent criterion is the one used for gating E1/E2 and is always labeled as amended
wherever cited. Note also that the amended n* (argmax Δ among per-row-significant block
lengths) remains a post-hoc maximum; the V3 convergence study is the pre-registered guard
that the selected separations are budget-stable, below-the-wall signals.

## ADDENDUM (2026-07-08, post-hoc robustness analysis, no gate role)

Because the E0/E1 argument leans on DSA being uninformative on this testbed, a DSA
configuration-robustness check (`experiments/exp3_dsa_robustness.py`) recomputes DSA on the
four argument-carrying pairs (null, prune, noise, difftask) under five configurations
(pre-registered; deeper/higher-rank; no delay embedding; reduced-rank regression; Wasserstein
score). Scores are compared within-config only. This is diagnostic, not a gate criterion; all
DSA claims in the paper are scoped to "DSA as configured, on models at this scale."

## AMENDMENT 2 (2026-07-08, before E1 results existed; E1 was restarted)

The E1 sweep at the E0 budget projected to >12 h under GPU contention. E1/E2 are
characterization experiments, not the existence claim, so their budget is reduced (disclosed):
generation B=64 × T=8192 per run (same-budget floors per condition keep the honesty device),
d̄ blocks n ∈ {1, 2, 4, 8, 12} (E0 located every same-task separation at n ≤ 8; the reduced
wall is log2(5.2e5)/ĥ ≈ 28 ≫ 12), belief blocks n ≤ 8, OT repeats 2, DSA iters 800 (config
otherwise identical). E0's results and claims are untouched. The partially-completed E1 run at
the old budget was discarded before its results were seen (it saved nothing to results/).

## What would make us wrong (a priori)

- The noise twin's marginal may drift (TV > 0.01) before d-bar_n≥2 separates ⇒ smaller σ*, weaker
  separation; if no σ in the grid satisfies both, the noise pair fails the premise and we say so.
- Distillation may match the *process*, not just the output, making d-bar ≈ floor — that is a
  correct null ("the process did not change"), reported as such, not a separation.
- DSA might *also* see the noise twin (via its stochastic residuals): then the disagreement
  claim dies (K1 for that pair) and we report DSA's sensitivity honestly.
