# Green Spec — Kac–Rice level-crossing density as a spatial-domain high-frequency loss for INRs

**Status:** GREEN, headline, MEDIUM confidence. Survived the full gauntlet (local 3M-abstract index +
4 web angles + my own knowledge). Repo-ready. Tractable on one RTX 3080, ~weekend to first signal.

---

## One-paragraph idea
Coordinate-MLP implicit neural representations (INRs: SIREN, NeRF, neural SDFs) suffer **spectral
bias** — they fit low frequencies fast and high-frequency detail slowly or never. Existing fixes act
in the *frequency domain* (Focal Frequency Loss), on *activations* (SIREN, FINER), or via *coarse-to-
fine curricula* (FreeNeRF). Borrow the **Rice / Kac–Rice formula** from random-field theory: the
expected **level-crossing density** of a field is a direct, closed-form proxy for its high-frequency
content (Rice: crossing rate ∝ √(second spectral moment)). Turn it into a **differentiable auxiliary
loss** that pushes the INR's output field toward a target crossing density — a **spatial-domain,
FFT-free, mesh-free** high-frequency drive that works on **irregular / non-grid domains** where
frequency-domain losses are awkward or undefined.

## Why it's novel (the kill-check verdict, condensed)
- **No prior art** found for a level-crossing-density loss on INRs, across 3M arXiv abstracts +
  targeted web search. The field mitigates INR spectral bias with activations / positional encoding /
  Fourier losses / Sobolev(H¹) losses / curricula — **never** level-crossing density.
- Rice-formula level crossings appear only in **signal processing, reliability, and spiking-neuron**
  literature — never as an INR training objective. The cross-domain gap is the whole point.
- **Honest caveats baked in:**
  1. **Drop the "RG curriculum" framing.** Coarse-to-fine frequency curricula already exist (FreeNeRF,
     spectral gating). Do **not** claim that part. Lead with the level-crossing-density loss.
  2. The *crossing-rate ↔ frequency* link is classic (zero-crossing rate is a textbook frequency
     feature). The novelty is the **differentiable-loss instantiation for INRs**, and specifically its
     advantage on **non-grid domains**. Position there — not as "we discovered crossings measure
     frequency."
  3. On a regular grid this may only **match** a Fourier/Sobolev loss. **The win must come from the
     irregular-domain / mesh-free case.** If it only ties on grids, that's a negative result — say so.

## The mechanism (concrete)
Let f_θ(x) be the INR output field over coordinates x (∈ ℝ¹ for signals, ℝ² for images, ℝ³ for
SDF/NeRF). The INR is differentiable in x, so **∇_x f_θ is free via autograd** — this is the key
enabler (no finite differences, works at arbitrary sampled points).

**Rice formula (1D, level u):** expected up-crossing rate
  μ(u) = ∫ |f'| · p(f = u, f') df'   →   for a stationary Gaussian field, μ(u) ∝ (λ₂/λ₀)^½ e^(−u²/2λ₀),
where λ_k are spectral moments. The crossing density is monotone in √λ₂ = RMS of the derivative =
high-frequency energy.

**Differentiable estimator (what you implement):** over a batch of sampled coords {xᵢ},
  crossing_density(u) ≈ (1/N) Σᵢ δ_ε(f_θ(xᵢ) − u) · |∇_x f_θ(xᵢ)|
using a **smoothed Dirac** δ_ε(z) = (1/√(2πε)) e^(−z²/2ε) (or a soft-bump), so the whole thing is
differentiable in θ. This is the **Kac–Rice integrand** evaluated by Monte-Carlo at the sampled points
— it needs **no grid and no FFT**, which is exactly why it beats Fourier losses off-grid.

**Loss:** L_aux = Σ_u ( crossing_density_θ(u) − target(u) )², with target(u) either (a) the empirical
crossing density of the ground-truth signal at supervised points, or (b) a schedule that ramps up the
demanded high-frequency crossing density. Total loss = recon_MSE + β · L_aux.

## Baselines it must beat (or honestly tie/lose to)
1. **Focal Frequency Loss** (Jiang et al., 2021) — the frequency-domain competitor.
2. **Sobolev / H¹ gradient-matching loss** (HANO-style) — the closest spatial-domain competitor.
3. **SIREN** and **FINER** — activation-based fixes (architecture baseline).
4. **FreeNeRF** free-frequency curriculum — for the NeRF setting.

## Experiments (in order of decisiveness)
1. **1D/2D signal & image fitting** (the standard spectral-bias benchmark). Metric: high-frequency
   PSNR / per-band spectral error vs iterations. *Expectation: ties Fourier loss on grids — this is the
   sanity check, not the win.*
2. **THE DECISIVE ONE — irregular / non-grid domain.** Neural SDF from a **point cloud** (no grid), or
   image fitting on **non-uniform samples**. Here FFT-based losses need resampling/interpolation and
   degrade; the crossing-density loss is defined pointwise and shouldn't care. Metric: reconstructed
   high-frequency detail vs Focal-Frequency (which must resample). **This is where a real win lives.**
3. **NeRF few-shot** (FreeNeRF setting) as a stretch, if 1–2 look good.

## Honest risks / kill conditions
- If it only **ties** Fourier/Sobolev losses on grids **and** shows no advantage off-grid → negative
  result, shelve it. Report honestly.
- Smoothed-Dirac bandwidth ε is a nuisance hyperparameter; too large blurs the crossing signal, too
  small makes gradients spiky. Budget a sweep.
- The Gaussian-field assumption behind the clean Rice formula won't hold exactly — but you use the
  *empirical* crossing density as target, so this only affects the theory framing, not the method.

## Why it's a good bet despite MEDIUM confidence
Genuinely novel mechanism on a real, crowded, high-interest problem, with a **specific structural
advantage** (mesh-free/FFT-free) that isn't just "another regularizer." Cheap to falsify (experiment 2
is a weekend). Even a null is publishable-adjacent as "when do spatial-domain crossing losses help INRs."
This is the best card the untapped-cross-domain-atom-bank machinery produced.
