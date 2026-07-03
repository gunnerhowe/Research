# Research notes — verified math and design decisions

Date: 2026-07-03. These notes pin down the facts the implementation relies on, correcting the spec
where its shorthand was imprecise. Everything below is either verified against the literature or
derived from first principles (derivations included).

---

## 1. Ornstein's d̄-distance: definition and the n-block characterization

**Definition.** For two stationary finite-alphabet processes X = (X_i), Y = (Y_i) with laws μ, ν,

    d̄(μ, ν) = inf over joinings λ of (μ, ν) of  P_λ(X_0 ≠ Y_0),

where a *joining* is a shift-invariant measure on the product space with marginals μ and ν.
Equivalently (Gray–Neuhoff–Shields 1975, *A generalization of Ornstein's d̄ distance with
applications to information theory*, Ann. Probab. 3(2):315–328): d̄ is the optimal-transport
(Wasserstein-1) distance between the process laws under the (normalized) Hamming metric on
sequence space.

**n-block characterization (the estimator's foundation).** Define

    d̄_n(μ, ν) = OT( P_n, Q_n ; cost = normalized Hamming on n-blocks ),

where P_n, Q_n are the n-block marginal distributions. Then for stationary μ, ν:

    d̄(μ, ν) = lim_{n→∞} d̄_n = sup_n d̄_n ,   and   d̄_n ≤ d̄ for every n.

**Derivation of superadditivity** (this is the standard argument; recorded here because the code's
plateau logic depends on it). Let λ* be an optimal coupling of (P_{n+m}, Q_{n+m}). Its restriction
to coordinates 1..n is a coupling of (P_n, Q_n) (stationarity ⇒ the marginals restrict correctly),
and its restriction to coordinates n+1..n+m is a coupling of (P_m, Q_m). Writing δ_i for the
disagreement probability at coordinate i under λ*:

    (n+m) · d̄_{n+m} = Σ_{i=1}^{n+m} δ_i  =  Σ_{i=1}^{n} δ_i + Σ_{i=n+1}^{n+m} δ_i  ≥  n·d̄_n + m·d̄_m .

So n·d̄_n is superadditive ⇒ (Fekete) lim d̄_n = sup_n d̄_n. Also every joining of the full processes
restricts to a coupling of n-blocks with per-coordinate mismatch = d̄-cost, hence d̄_n ≤ d̄.

**Consequences for the estimator:**
- Along the doubling sequence n = 1, 2, 4, 8, … the true d̄_n is **non-decreasing**
  (superadditivity gives d̄_{2n} ≥ d̄_n directly). Strict monotonicity for every step n → n+1 is
  NOT guaranteed; the spec's "monotone" is correct along doubling, which is exactly the schedule
  the spec proposes. Use doubling.
- Every d̄_n is a **certified lower bound** on d̄ (up to estimation error). So a plateau value is
  conservative: it can only understate the true distance.
- d̄_1 = total-variation-type distance between the 1-symbol marginals (OT with 0/1 cost = TV).
  If a surrogate matches the invariant measure's partition marginal, d̄_1 ≈ 0 and *all* signal
  comes from n ≥ 2 — this is precisely the "static vs dynamical" split, made quantitative per n.

## 2. Entropy is d̄-continuous — the correct inequality (spec correction)

The spec asserts `d̄ ≥ |h(X) − h(Y)|`. That is **not the correct form** (units alone: d̄ ∈ [0,1]
while entropy-rate differences are in bits and can't exceed log₂|A|; the true bound is Fano-type).
The correct statement (Shields, *The Ergodic Theory of Discrete Sample Paths*, GSM 13, §I.9,
"entropy is d̄-continuous"; continuity originally due to work in the Ornstein school / Marton):

    |h(μ) − h(ν)|  ≤  H_b(d̄) + d̄ · log₂(|A| − 1)      [bits/symbol]

where H_b is the binary entropy function. **Derivation** (Fano + optimal joining): take a joining
with per-coordinate mismatch rates δ_i averaging δ = d̄. Then
H(X^n) ≤ H(Y^n) + H(X^n | Y^n) ≤ H(Y^n) + Σ_i H(X_i | Y_i), and Fano's inequality gives
H(X_i | Y_i) ≤ H_b(δ_i) + δ_i log₂(|A|−1) (treating Y_i as an estimate of X_i with error prob
δ_i). Concavity of H_b ⇒ (1/n) Σ_i H_b(δ_i) ≤ H_b(δ). Divide by n, let n → ∞, symmetrize.

**Usable go/no-go lower bound.** Let g(δ) = H_b(δ) + δ·log₂(|A|−1), strictly increasing on
[0, 1 − 1/|A|]. Then

    d̄(μ, ν)  ≥  g⁻¹( |h(μ) − h(ν)| ) .

For a binary alphabet g = H_b and g⁻¹ = H_b⁻¹ on [0, ½]. Note g⁻¹(x) ≤ x, so this is *weaker*
than the spec's claimed bound — but it is the true one, and it preserves the spec's logic
completely: **an entropy-rate gap certifies d̄ > 0 with an explicit quantitative floor.** The code
implements g⁻¹ by bisection.

## 3. Entropy-rate estimators

- **LZ76**: parse the sequence into C(N) phrases per Lempel–Ziv 1976; ĥ = C(N)·log₂(N)/N
  bits/symbol, consistent for stationary ergodic sources. Simple, one number, robust.
- **Conditional block entropy**: ĥ_n = H_n − H_{n−1} (plug-in n-block Shannon entropies).
  Decreases toward h from above as n grows; undersampling bias pulls the plug-in H_n *down* once
  the observed support becomes comparable to sample count, so report the curve and read the
  plateau before the bias knee. Rule of thumb: trust n while (#distinct observed n-blocks)/N ≪ 1.
- These two bracket the answer in practice; agreement between them is the sanity signal.

## 4. The d̄_n estimator (design decisions)

- Empirical n-block distributions from overlapping windows (stride 1 by default), blocks encoded
  as integers (binary: bit-packed uint64, n ≤ 32 fine; m-ary: base-m digits).
- Dedupe blocks into weighted support; OT via POT `ot.emd2` (exact network simplex) with
  Hamming cost matrix computed by XOR + `np.bitwise_count` (binary) or digit-wise mismatch
  (m-ary). Sinkhorn only if support sizes force it.
- If support × support is too large, subsample blocks (random window starts) to a cap per side,
  dedupe, repeat over independent draws for error bars.
- **Noise floor (essential honesty device):** empirical OT between two *independent samples of
  the same process* is strictly positive at finite N (empirical measures don't coincide). Every
  d̄_n estimate is reported next to the same-n truth-vs-truth (and surrogate-vs-surrogate) floor,
  computed from disjoint data halves. Signal = estimate clearly above floor. This directly
  addresses the spec's kill condition "too noisy to converge at achievable sample sizes."

## 5. Analytic validation cases for the estimator

1. **iid Bernoulli(p) vs iid Bernoulli(q):** d̄ = |p − q| and moreover d̄_n = |p − q| for every n
   (product of per-coordinate monotone couplings is optimal; lower bound from d̄_1 marginal TV
   argument per coordinate). Sharp test at all n.
2. **Process vs itself (independent runs):** true d̄ = 0; estimator must return ≈ noise floor.
3. **iid Bernoulli(½) vs period-2 alternating (random phase):** identical 1-marginals
   (½, ½) ⇒ d̄_1 = 0, but the processes are dynamically as different as can be; d̄ > 0.
   The exact value can be computed by exact OT on analytic n-block distributions (both are known
   in closed form) — the numerical pipeline should reproduce it. This is the minimal pedagogical
   instance of "invariant measure blind / dynamics visible."

## 6. IAAFT surrogates (Schreiber & Schmitz 1996, Phys. Rev. Lett. 77:635)

Iterative Amplitude-Adjusted Fourier Transform: alternate (a) replace Fourier amplitudes with the
original's (keep current phases), (b) rank-order remap values to the original's exact amplitude
distribution. Converges in tens of iterations. Result: **exactly** the original marginal
distribution, **approximately** the original power spectrum / autocorrelation, higher-order
temporal structure destroyed. Consequence for our design: applied to the sampled x(t) of Lorenz,
the surrogate is *by construction* (i) Wasserstein-on-marginal-matched (W1 ≈ 0 exactly up to
permutation), (ii) power-spectrum-matched — it simultaneously defeats the invariant-measure metric
AND the spectrum baseline, which is exactly the discrimination test the spec wants.

## 7. Additional adversarial surrogates (all zero-training constructions)

- **Time-rescaled Lorenz (speed ×2):** integrate dx/dt = 2·f(x). *Exactly* the same attractor and
  invariant measure; sampled at the same τ it is statistically Lorenz sampled at 2τ — different
  temporal/mixing structure, entropy rate per symbol ≈ doubled. The spec's option (c)
  ("same measure, different entropy") achieved by construction.
- **Time-reversed Lorenz:** identical invariant measure, identical power spectrum (ACF is even),
  identical entropy rate (reversal preserves h) ⇒ the Fano floor is 0 and only the direct d̄
  estimator can possibly see it (Lorenz is far from reversible). Hard mode — include, report
  honestly whichever way it goes.
- **ρ-changed Lorenz (ρ = 32):** positive control; different attractor, every metric should fire.
- **Independent truth run:** negative control; every metric should read ≈ 0/floor.

## 8. Novelty scan (2026-07-03)

- DySLIM (arXiv 2402.04467): invariant-measure regularization for training; no process-level
  metric.
- "Training neural operators to preserve invariant measures of chaotic attractors"
  (arXiv 2306.01187): optimal transport / contrastive losses on the *invariant measure*.
- "The Dynamic-Probabilistic Consistency Gap in Chaotic Surrogate Modeling" (arXiv 2605.31547):
  closest in *spirit* (finite-horizon probabilistic objectives can degrade dynamics) — but the
  proposed fix is a Kalman-filter training framework; no d̄, no symbolic dynamics, no
  OT-on-block-distributions, no entropy-rate metric.
- "Data-Driven Performance Measures using Global Properties of Attractors…" (arXiv 2506.09546):
  correlation-integral / pdf-based measures — again static-attractor objects.
- No hit anywhere for Ornstein d̄ / ergodic-isomorphism metrics in ML surrogate evaluation.
  **The conceptual gap the spec identifies is still open.**

## 9. Symbolization choices for Lorenz-63

- Primary: stroboscopic sampling at interval τ, symbol = sign(x). Sign of x = which lobe of the
  attractor; closely tied to the classical Lorenz template symbolic dynamics. τ default 0.1
  (lobe residence time is ~0.7–1.5 time units, so blocks of n = 16–32 span several lobe
  decisions). τ ∈ {0.05, 0.1, 0.25} in sensitivity checks.
- Alternatives for partition-sensitivity: 4-bin quantile partition on x (quantiles computed on
  TRUTH and frozen, applied identically to surrogates — the spec's "same partition" requirement),
  and a coarse box partition on (x, z).
- Expected truth entropy rate at τ = 0.1: bounded by h_KS·τ ≈ 0.906 nats × 0.1 ≈ 0.13 bits/symbol
  (h_KS(Lorenz-63) = Σ positive Lyapunov exponents ≈ λ₁ ≈ 0.906 nats/time by Pesin). Useful as an
  order-of-magnitude check on the estimators.

## Sources

- Gray, Neuhoff, Shields, *A Generalization of Ornstein's d̄ Distance with Applications to
  Information Theory*, Ann. Probab. 3(2), 1975. https://projecteuclid.org/euclid.aop/1176996402
- Shields, *The Ergodic Theory of Discrete Sample Paths*, AMS GSM 13, 1996 — §I.9 (d̄ theory,
  entropy d̄-continuity). https://bookstore.ams.org/gsm-13
- El Gamal & Gray, *Katalin Marton's Lasting Legacy* (entropy/d̄ continuity context).
  https://ee.stanford.edu/~gray/Kati_Marton.pdf
- Schreiber & Schmitz, *Improved surrogate data for nonlinearity tests*, PRL 77:635, 1996.
- DySLIM: https://arxiv.org/abs/2402.04467 · Invariant-measure training: https://arxiv.org/abs/2306.01187
- DPC gap / KAFFEE: https://arxiv.org/abs/2605.31547 · Attractor measures: https://arxiv.org/abs/2506.09546
