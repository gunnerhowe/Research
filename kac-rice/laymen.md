# The Kac–Rice Trilogy, In Plain English — and How to Whiteboard It Cold

Three papers, one idea, honestly measured. This document has two parts: Part 1 is
the explanation you could give anyone; Part 2 is everything you need to stand at a
whiteboard in front of Anthropic and own it.

---

# Part 1 — What we did, for anyone

## The one idea underneath all three papers

Take a wiggly signal. Draw a horizontal line through it. **Count how many times the
signal crosses the line.** A lazy, smooth signal crosses a few times; a detailed,
rapidly-wiggling one crosses constantly. That count — proven precisely by Stephen
Rice at Bell Labs in 1944 while studying telephone noise — is a *frequency meter
that needs no Fourier transform, no grid, no mesh*. Just values and slopes at
whatever scattered points you happen to have.

Neural networks give you slopes for free (that's what backpropagation computes).
So this 1940s counting trick can be turned into a *training signal* for modern
neural networks — and it turns out to be the first rung of a whole ladder:

- **Rung 1 (Paper 1):** how often the field crosses a level → *how detailed it is*
- **Rung 2 (Paper 2):** the same count, used as a *speed limit* instead of a demand
- **Rung 3 (Paper 3):** climb from counting crossings to measuring the full
  *geometry and topology* of the level sets — areas, boundary lengths, and the
  number of blobs and holes ("Euler characteristic")

By the end, we had built the complete differentiable "integral geometry" of neural
fields — a family of rulers that measure a network's output the way a
mathematician would measure a landscape — and, just as importantly, we mapped
**exactly where each ruler can be trusted and where it breaks.**

## Paper 1: teaching networks fine detail (the demand direction)

**Problem:** Networks that store images/signals as formulas (used in NeRF-style 3D
graphics and neural shape models) learn the blurry version first and fine detail
last or never — "spectral bias." The best loss-based fixes need data on a neat
grid; real data (laser scans, scattered sensors) often isn't.

**What we did:** Made the crossing count differentiable and told the network:
"your reconstruction should cross each brightness level about as often as the real
image does." No grid needed, ever.

**Honest result:** Where data is scarce and scattered, every detail-promoting loss
helps enormously (+2–3 dB — visibly sharper); ours *ties* the grid-based champion
on photos, *beats* it on texture (the kind of content the 1944 theory is actually
about), and is the only method that doesn't care how sloppy its slope estimates
are. It did not broadly beat the champion — we said so in the abstract.

## Paper 2: the same tool as a speed limiter (the cap direction)

**Problem:** Physics-informed neural networks (PINNs) — networks that learn to
solve physics equations — are widely reported to blow up in "runaway
high-frequency oscillation" on hard equations. If that's the disease, our counter
is the natural cure: penalize crossings only *above* the physical budget.

**What we did:** Built the one-sided budget (provably silent when the field
behaves — zero loss, zero gradient), rebuilt a published benchmark exactly, and
went looking for the disease.

**Honest result:** The disease never showed up. Our PINNs failed *every* time —
but always by being too smooth and losing track of the solution, never by
oscillating. The budget was verifiably inert (on one seed, bit-for-bit identical
training with and without it). We proved the cure works where the disease exists
(a lab-controlled over-oscillator gets clamped, and gets *better*), and published
the mismatch — plus a one-plot diagnostic that tells anyone which failure their
PINN actually has *before* they buy a stabilizer. Reported failure modes can be
artifacts of a particular software stack; the field should know that.

## Paper 3: from detail to topology (the big swing)

**Problem:** Controlling the *topology* of a learned 3D shape — "one piece, two
handles, no hidden bubbles" — currently requires persistent homology: powerful,
grid-bound machinery that costs ~1 second per training step.

**What we did:** Extended our counting trick up the ladder using a classical
theorem (Gauss–Bonnet): the number of blobs minus holes of a shape can be written
as a curvature integral, which we can sample at scattered points, differentiably,
for ~3 milliseconds per step — **250× cheaper**. Validated it to 1–3% accuracy
against exact topology in 2D and 3D. Along the way we discovered four design rules
the hard way (each from a documented failure), including the elegant one: the
Euler characteristic is blobs *minus* holes, so an optimizer asked to reduce it
will happily manufacture a fake hole to cancel a fake blob — unless you also
charge it for perimeter. **The full measurement vector, not the single invariant,
is the loss.**

**Honest result:** In 2D it's the only method in a controlled comparison that
fixes topology (3/3) *and* preserves the signal — the smoothing alternative pays
11–17× in fidelity, the naive version repairs nothing. In 3D it fails, and the
failure is the most interesting finding of the trilogy: **gradient descent
adversarially hides topological junk below the sampling density** — tens of
thousands of microscopic bubbles exactly where the estimator is blind, invariant
to 4× more samples. Persistent homology has no such blind spot (its grid *is* the
evaluation), and that is precisely what its 250× cost buys. We measured the
boundary instead of hiding it.

## Why this matters for AI

1. **A new tool family, imported and validated.** Random-field theory (Rice,
   Kac, Adler–Taylor) is now wired into neural network training with working
   code, exact validation, and known limits. Mesh-free detail losses, PINN
   diagnostics, and 250×-cheaper topology measurement are all real, usable today
   — within their mapped domains.
2. **A measured case study of specification gaming.** Paper 3's failure is
   Goodhart's law caught on camera: give an optimizer a sampled objective, and it
   doesn't just fail to satisfy the *intent* — it actively locates the measuring
   instrument's null space and stuffs the violation there. We produced this in a
   fully controlled setting, diagnosed the mechanism, quantified its invariance
   to naive fixes, and identified what eliminates it (a measurement with no null
   space — at 250× the price). That's an alignment-relevant parable with
   receipts, not a metaphor.
3. **A demonstration of how AI-accelerated science should behave.** Every paper
   pre-registered its kill conditions, reported the negative halves in the
   abstract, gave every method identical information, and ships every number
   regenerable from raw logs by one script. Two of three headline hypotheses
   partially or fully failed — and all three papers are stronger for saying so.

---

# Part 2 — Whiteboard mastery (present it cold at Anthropic)

## 2.0 The 90-second arc (memorize this cold)

> "I took one piece of 1940s Bell Labs math — the Rice formula, which says
> *counting how often a signal crosses a line measures its frequency content* —
> and pushed it through three questions. **Can it teach networks detail?** Yes:
> ties the state of the art without ever needing a grid. **Can it cap runaway
> oscillation in physics networks?** The cap works — but the famous disease
> didn't exist in our faithful reproduction, so we published the absence, with a
> diagnostic. **Can it climb to full topology control and displace persistent
> homology at 250× less cost?** In 2D yes, uniquely; in 3D the optimizer
> adversarially hides junk below the sampling density — Goodhart's law, measured
> in a controlled setting. Three papers, every kill condition pre-registered,
> every number regenerable from raw logs. The epistemics are as much the product
> as the math."

## 2.1 Whiteboard choreography — draw in this order

**Board 1: the object.** Draw a wiggly 1D signal, a dashed horizontal line at
height *u*, fat dots at crossings. Write Rice's formula:

```
c(u) = (1/π)·√(λ₂/λ₀)·exp(−u²/2λ₀)      λ₀ = Var f,  λ₂ = Var f′
```

Say: "√(λ₂/λ₀) is literally the RMS frequency. Crossings = spectrum, no Fourier
transform. And the modern enabler: autograd gives me f′ anywhere, for free."

**Board 2: the engine.** Write the co-area formula and the estimator:

```
∫ g(f(x))·‖∇f(x)‖ dx  =  ∫ g(u) · (size of level set {f=u}) du

ĉ_ε(u) = (1/N) Σᵢ δ_ε(f(xᵢ)−u)·‖∇f(xᵢ)‖
```

Say: "Deterministic identity — no Gaussian assumptions, works in any dimension:
crossings in 1D, contour length in 2D, surface area in 3D. Monte-Carlo over ANY
scattered points. Differentiable in the network weights. This one line is the
whole trilogy's engine."

**Board 3: the ladder.** Draw three rungs, label left-to-right:

```
M₁ demand (P1: INR detail)  →  M₁ cap (P2: PINN budget)  →  (M₀,M₁,M₂) vector (P3: topology)
```

Under rung 3 write Gauss–Bonnet:

```
χ(A_u) = (1/2π) ∮ κ ds        κ = div(∇f/‖∇f‖)     (2D; 3D uses Gaussian curvature K)
```

Say: "Same engine, three payloads. The third rung measures blobs-minus-holes as a
curvature integral I can sample."

**Board 4: the results ladder** (draw the two number-columns from §2.3's tables
below — scattered-image PSNR for P1; the 2D-repair quartet and 3D table for P3).

**Board 5: the punchline drawing.** Draw a big shape with a few honest sample
dots, then dozens of tiny bubbles BETWEEN the dots. Say: "Asked to fix topology
it can't reach honestly, gradient descent puts the violations exactly where my
sampled measurement is blind. 4× more samples — debris unchanged, it just
relocates. Closing the window needs a million points; that's cubic, and it
erases my 250× cost win. Persistence has no blind window — its complex IS the
eval grid. Its cost is the price of no null space. **Optimizers hunt null
spaces.** That sentence is the trilogy's deepest export."

## 2.2 The four design rules (know the *failure* behind each)

1. **Level ladder.** Topology is a step function of the weights — smoothed
   estimators only feel gradients within ε of a transition. Receipt: a cap probing
   levels ≤0.4 against a peak at 0.7 sat at exactly zero loss for 2,000
   iterations. Fix: dense levels spanning the range, spacing ≈ ε — a ratchet where
   the topmost active level always has its hand on the offending peak.
2. **C² backbone.** ReLU nets hide their curvature in kinks (measure-zero sets MC
   sampling never hits) — the Gauss–Bonnet integral silently undercounts. Receipt:
   same field, ReLU → loss never fires; SIREN → reads topology correctly.
3. **The vector, not the invariant.** χ = blobs − holes is gameable by
   cancellation. Receipt: asked for χ=1, the optimizer built 4 blobs + 3 holes.
   Charging perimeter (M₁ — literally Paper 2's budget reused) makes cheating
   expensive; honest deletion becomes the cheapest descent path.
4. **Sampling-scale coverage.** All sampled estimators are blind below sample
   spacing — and rule 4 is the one 3D can't afford. This isn't a bug in our code;
   it's a theorem about sampled objectives, and it's why the 3D result is a
   finding rather than an embarrassment.

## 2.3 Numbers to memorize (the only table you need)

| Claim | Number |
|---|---|
| P1 estimator vs exact crossing counts / Rice | within 5% / 10% |
| P1 scattered-image gain over plain MSE | +2.3–3.0 dB (ReLU-PE), +1.4–1.8 (SIREN) |
| P1 vs Focal Frequency Loss on photos | −0.2 to −0.5 dB PSNR, better SSIM: a tie |
| P1 on texture (theory's home turf) | **+0.6 dB win**, above the interpolation ceiling |
| P1 oracle-gradient test | Sobolev improves (+0.2), ours doesn't need it (−0.3) |
| P2 vanilla PINN failure (front / chaotic) | rel-L2 0.867±0.016 / 1.31 — always UNDER-oscillating |
| P2 budget inertness | one seed bit-identical (0.8617398766); shifts ≤0.016 elsewhere |
| P2 in-vitro clamp | crossings 3.4–4.3 → 2.6–2.7 (budget 3.0), test error −15 to −41% |
| P3 χ estimator accuracy | 1–3% (annulus 0.018±0.015; solid torus 0.016±0.042 — both want 0) |
| P3 cost vs persistent homology | ~3 ms vs 650–1000 ms per iteration ≈ **250×** |
| P3 2D repair quartet | ours 3/3; smoothing 2/3 at **11–17×** fidelity cost; χ-alone 0/3 |
| P3 3D verdict | ours 0/9 + ~1.7×10⁵ spurious features; PH 4/9 exact, median b₁ error 1 |
| P3 the null-space receipt | debris invariant across 8k/16k/32k samples; closing needs ~10⁶ (cubic) |

## 2.4 Hard questions you WILL get, with the honest answers

**"Nothing here beats state of the art. Why should Anthropic care?"**
Two answers. Substantively: a validated 250×-cheaper topology measurement, a
grid-free detail loss that ties SOTA, and a PINN failure-mode diagnostic are real
tools with mapped domains — "tie at much lower structural requirements" is a
result practitioners use. Epistemically: three pre-registered studies where I
reported two hypothesis failures in the abstracts, reproduced a published
baseline that didn't behave as published and said so carefully, and caught my own
metric being gamed by my own optimizer. If you're hiring people to evaluate AI
systems that will try to look good under measurement, that last skill is the job.

**"Isn't Paper 3's failure just... a bug you could fix?"**
No — it's structural, and we falsified the easy fixes on camera: 4× sampling
(debris invariant), stronger physics constraints, bandwidth-limited networks. The
estimator has a null space; closing it costs cubically. What WOULD work is
changing the measurement (importance sampling near level sets, hybrid sparse-PH
correction steps) — listed as open, untested, unclaimed.

**"Why did the PINN disease not appear? Are you saying the ASPEN authors are wrong?"**
Carefully: no. We matched their published configuration in every documented
detail, tried two input conventions, two testbeds, and their full training depth
— and always got the *propagation* failure (accurate early, drifts late), never
oscillation. Failure-mode selection in PINNs is implementation-sensitive
(initialization, precision, framework defaults). That's exactly why the paper
ships a one-plot diagnostic: check which branch YOUR stack is on before buying a
stabilizer aimed at the other one.

**"Why is crossing density robust where gradient-matching isn't?" (P1's best mechanism)**
Sobolev consumes gradient targets *pointwise* — its accuracy tracks target
quality, proven by the oracle test. Ours consumes only their *batch statistics* —
pointwise noise averages out. Same reason it wins on texture (statistically
homogeneous = the stationary-random-field regime where Rice's formula is exact)
and merely ties on edge-dominated photos (localized, phase-critical structure a
distributional statistic can't place).

**"What does the Goodhart result actually generalize to?"**
The precise claim: any differentiable objective evaluated by sampling has a null
space below its sampling density, and optimization pressure reliably finds it —
we showed the violation doesn't shrink, it *relocates*. The measured antidotes:
(a) measurements without null spaces (PH's grid — expensive), (b) pricing the
cheat channels (the vector rule — our M₁ guard worked exactly until the debris
dropped below ITS sampling too). For evaluating models that adapt to their
evaluators, that's a concrete, quantitative case study.

**"Rice formula assumes Gaussian processes. Images aren't Gaussian."**
The Gaussian closed form is motivation only. The losses stand on the co-area
formula — a deterministic identity for any Lipschitz field — and on *empirical*
targets measured from the ground truth. The validation figure shows the estimator
tracking exact counts precisely where the field is visibly non-Gaussian.

**"How much of this did AI do?"**
Own it: the program ran as human-directed, AI-executed research — spec and
gates set up front, kill conditions enforced, every claim regenerable from raw
logs by one script, and external review caught real errors that got fixed (a
hand-computed ratio, an overselling abstract). It's a working demonstration of
the workflow Anthropic says it wants to make safe and common.

**"What would you do next?"**
(1) Point-cloud SDFs for Paper 1's loss — the setting where no grid ever exists
and the structural advantage must finally bind. (2) The 3D topology fix via
measurement redesign: importance-sample the level sets, or hybrid cheap-Minkowski
steps with sparse PH corrections. (3) The Goodhart testbed itself is reusable:
sampled-objective null spaces as a benchmark for specification-gaming studies.

## 2.5 If you only remember ten things

1. Crossings per level = frequency content. Rice, 1944. No Fourier transform.
2. Co-area formula = the engine: level-set size as a samplable, differentiable integral.
3. Autograd makes 80-year-old integral geometry trainable. That's the whole trick.
4. P1: ties the grid-based champion grid-free; wins on texture; robustness proven by the oracle test.
5. P2: the cure works (in vitro), the disease didn't exist (in vivo) — published both, with a diagnostic.
6. One P2 seed trained bit-identically with the loss on: the cleanest "provably harmless" receipt in the trilogy.
7. P3: topology at 3 ms/step, validated to 1–3%, 250× cheaper than persistence.
8. Four design rules, each bought with a documented failure: ladder, C², vector, sampling coverage.
9. The 3D failure is the trilogy's biggest finding: **optimizers hunt null spaces** — measured, invariant to naive fixes.
10. Two of three headline hypotheses failed, all three papers say so in the abstract — and that's the strongest line on the whiteboard.

*Everything above regenerates from `github.com/gunnerhowe/Research` (kac-rice):
tests, raw JSONs, one figure-script per paper.*
