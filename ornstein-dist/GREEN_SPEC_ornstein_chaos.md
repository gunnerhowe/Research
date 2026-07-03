# Green Spec — Ornstein d̄-distance as a dynamical-fidelity metric for chaotic surrogates

**Status:** GREEN, from the method-first inversion, CAUTIOUS / MEDIUM confidence. Novel far-transfer from
ergodic theory. **The whole bet rides on one crux: can you estimate d̄ in practice?** This spec is built to
answer that first, cheaply, before any big investment. Riskier than the Kac–Rice green; more conceptually
original. ~1–2 weekends on the 3080, Lorenz/KS-scale.

---

## One-paragraph idea
Learned surrogates of chaotic systems (neural operators, Neural ODEs, reservoirs) can't be judged by
**pointwise trajectory error** — trajectories diverge exponentially, so MSE is meaningless past a short
horizon. The field's fix is to compare the **invariant measure** (DySLIM, "preserve invariant measures of
chaotic attractors," Wasserstein-on-attractor). But the invariant measure is a **static distribution** —
*where* the system spends time. Two systems can share an invariant measure yet have completely different
**dynamics** (mixing rate, entropy, temporal correlation). Borrow **Ornstein's d̄-distance** from ergodic
isomorphism theory: a metric on stationary processes that measures how close two systems are to being
**dynamically isomorphic** (tied to Kolmogorov–Sinai entropy). It penalizes a surrogate that reproduces the
attractor's *density* but not its *temporal/mixing structure* — a discrimination Wasserstein-on-measure
structurally cannot make.

## Why it's novel (kill-check verdict, condensed)
- The measure-theoretic-eval space is **crowded** (DySLIM 2402.04467; preserve-invariant-measures
  2306.01187; adversarial-OT 2604.21097) — **but all compare the stationary invariant measure.**
- **Ornstein d̄ is a different object**: it metrizes isomorphism of the *dynamics as processes*, not the
  static distribution. Absent from the 3M-abstract index and web as an ML surrogate metric = genuinely
  untapped. The conceptual gap (distributional vs dynamical equivalence) is the whole pitch.

## THE CRUX — estimating d̄ in practice (resolve this FIRST)
Ornstein's d̄ is defined as an infimum over **joinings** of the two processes — famously hard in general.
You do **not** compute it exactly; you estimate a tractable proxy on **symbol sequences**:

1. **Symbolize both systems.** Run the ground-truth system and the surrogate long enough to sample their
   attractors. Map each state to a symbol via a partition:
   - Lorenz-63: sign of x (2-symbol) or a coarse box partition — a near-generating partition is known.
   - KS / higher-dim: data-driven partition (k-means on delay embeddings, or coarse binning). **The
     partition is a confound — see risks.** Use the SAME partition for both systems.
2. **Estimate d̄ between the two symbol processes.** The practical, well-founded estimator: d̄ equals the
   limit of the **optimal-transport distance between the n-block distributions under per-symbol Hamming
   ground cost**. So for block length n = 1,2,4,8…:
   - Build empirical n-gram distributions P_n (truth) and Q_n (surrogate).
   - Compute OT(P_n, Q_n) with ground cost = normalized Hamming distance between n-blocks.
   - d̄ ≈ the value as n grows (it's monotone/convergent; plateau = estimate).
   This is a **standard OT problem** (POT library, Sinkhorn for speed) — fully tractable at Lorenz/KS scale.
3. **Cheap necessary-condition pre-check** (do this in an hour before anything else): d̄ ≥ |h(X) − h(Y)|
   where h is the KS entropy rate. Estimate entropy rates of both symbol streams (Lempel–Ziv / block-entropy
   / Grassberger estimators). If the surrogate's entropy rate already differs from truth, d̄ is provably
   bounded below — instant evidence the metric sees something. If entropy rates match, the OT-on-n-blocks
   estimator earns its keep.

**Kill condition for the crux:** if the OT-on-n-blocks estimator is too noisy to converge at achievable
sample sizes, or is dominated by partition choice, the metric isn't practical — report that and stop.

## THE DECISIVE EXPERIMENT (the "does it add anything?" test — this IS the paper)
Construct a surrogate that is **Wasserstein-matched but dynamically wrong**, and show d̄ catches it while
Wasserstein-on-measure doesn't:
- **Truth:** Lorenz-63 (or KS). Compute its invariant measure + symbol process.
- **Adversarial surrogate:** one that reproduces the invariant measure but scrambles the dynamics. Options:
  (a) a **phase-shuffled / IAAFT surrogate** (preserves the marginal + power spectrum, destroys higher-order
  temporal structure); (b) a neural surrogate **trained only to match the invariant measure** (DySLIM-style)
  but under-fit on dynamics; (c) a system with the **same measure but different entropy** (achievable by
  construction in symbolic systems).
- **Metrics compared:** Wasserstein-on-invariant-measure (small by construction) vs **d̄ (should be large)**
  vs Lyapunov-spectrum-match vs power-spectrum-match.
- **Win = d̄ is large exactly where Wasserstein is small** → d̄ measures a real, orthogonal failure mode. If
  d̄ tracks Wasserstein everywhere (adds nothing), that's an honest null — say so.

## Then (if the decisive test lands): d̄ as a training signal
Add a differentiable surrogate of the d̄ objective (Sinkhorn-OT on soft-symbol / n-block distributions is
differentiable) as an auxiliary loss for a neural surrogate; show it improves dynamical fidelity (entropy
rate, mixing) over a DySLIM (invariant-measure-only) baseline at equal trajectory MSE.

## Baselines it must beat / complement
DySLIM & invariant-measure preservation (2306.01187); Wasserstein-on-attractor; Lyapunov-spectrum matching
(Engelken/Vogt); power-spectrum / autocorrelation matching. The claim is **complementary discrimination**,
not "better everywhere."

## Honest risks / kill conditions
1. **Partition dependence.** d̄ estimates depend on the symbolization; for unknown systems a generating
   partition may not exist. Mitigate: fix one partition for both systems, and report sensitivity across a
   few partitions. If conclusions flip with partition, the metric is too fragile — report it.
2. **Estimator variance.** OT-on-n-blocks needs enough samples for large n; convergence may be slow. The
   entropy-rate lower bound is the cheap fallback signal.
3. **The adversarial surrogate must be constructible.** If you can't build a Wasserstein-matched-but-
   dynamically-wrong surrogate, the metric's advantage is untestable → the whole premise is moot. (IAAFT
   surrogates make this very likely constructible — that's why they're the first choice.)
4. **Impact ceiling:** a better fidelity metric / training signal for chaotic surrogates — substantial in a
   hot area, not transformative.

## Why it's worth the risk
Genuinely original (ergodic isomorphism theory is untapped in ML surrogate eval), with a **clean, cheap,
one-hour go/no-go** (the entropy-rate lower bound) and a **single decisive experiment** (IAAFT adversarial
surrogate) that either shows d̄ sees what Wasserstein can't — or honestly doesn't. This is the green the
**method-first inversion** produced that three hole-first passes never reached; it's the proof the inversion
works.
