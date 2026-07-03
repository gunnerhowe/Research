# The Kac–Rice Project, In Plain English

## Part 1 — What we did, for anyone

### The problem, with no jargon

There's a kind of neural network that stores a picture (or a sound, or a 3D shape)
as a *formula* instead of a file. You give it a location — "what color is the pixel
at (0.3, 0.7)?" — and it answers. These are called **implicit neural
representations (INRs)**, and they're used in things like NeRF (the "turn photos
into a 3D scene" technology) and neural 3D shape models.

They have one famous flaw: **they learn the blurry version first and the fine
detail last — or never.** Train one on a photo and you get the shapes and shading
quickly, but the grass texture, the hair, the sharp edges take forever to appear.
Researchers call this *spectral bias*. It's arguably THE central annoyance of this
whole subfield.

### The old fixes, and their blind spot

The best existing fixes either redesign the network itself, or add a "detail
police" term to the training objective: take the network's output and the real
image, run both through a **Fourier transform** (the math that splits a signal
into bass/treble frequencies), and punish the network for missing treble.

Here's the blind spot: a Fourier transform needs the data on a **neat, regular
grid** — like a full spreadsheet with every cell filled. But lots of real data
isn't like that. A laser scan of a statue is a cloud of scattered points. Sensor
readings come from wherever the sensors happen to sit. If you only know 12% of the
pixels, scattered randomly and unevenly, you must first *guess* the full grid by
smearing your scattered samples — and that smearing destroys exactly the fine
detail you were trying to protect.

### Our idea: count crossings instead

We dug up a piece of 1940s mathematics from Bell Labs telephone engineering:
**Rice's formula**. The intuition is simple enough to explain to a kid:

> Draw a horizontal line through a wiggly signal. **Count how many times the
> signal crosses that line.** A lazy, smooth signal crosses it a few times. A
> detailed, rapidly-wiggling signal crosses it constantly. *Crossing counts are a
> frequency meter that needs no Fourier transform.*

(Think of a guitar string: the higher the pitch, the more often it whips past its
resting line each second. Rice proved the precise version of this in 1944 for
telephone noise.)

The beautiful part: to count crossings, you don't need a grid. You just need to
check the signal's value and slope at whatever scattered points you happen to
have. Neural networks give you slopes for free (that's what backpropagation
computes). So we turned "how often does the field cross each level" into a
**training penalty**: the network is told, "your reconstruction should cross the
brightness level 0.5 about as often as the real image does — and level 0.3, and
0.7, and so on." If its answer is too blurry, it crosses every level too rarely,
and it gets pushed to add detail.

We're the first to use this as a training objective for these networks — we
checked the literature carefully.

### What we found (the honest version)

We built it, mathematically verified the counting machinery is exact, and ran a
fair tournament against the best existing training-objective fixes, giving every
method identical information.

- **When data is scarce and scattered, every "detail police" method is a huge
  win** — ours included. Adding any of them took reconstructions from a smeary
  mess to visibly coherent (+2–3 dB, which is a big, visible jump).
- **On ordinary photos, ours ties the Fourier method. It does not beat it.** We
  set out hoping it would, and we say plainly in the paper that it didn't.
- **On texture — content that's statistically "the same all over," like fabric,
  grass, or noise patterns — ours beats the Fourier method.** That's satisfying,
  because texture is exactly the kind of signal the 1940s theory was built for.
- **Ours is the low-maintenance one.** It's the only method in the tournament
  that never needs a grid at any step, and — this was the neatest experiment — it
  performs the same whether you feed it sloppy slope estimates or perfect ones.
  The competing slope-matching method gets better with perfect slopes, meaning it
  *depends* on information you usually can't get. Ours only uses slope
  *statistics*, so individual errors wash out.
- **When data is plentiful and on a grid, none of these methods help** — and the
  slope-based ones (ours included) actually hurt. Detail-police objectives are
  tools for scarcity, not abundance.

### Why this matters

1. **For scattered-data AI**: 3D shape networks trained on raw laser scans,
   irregular sensor networks, adaptively-sampled scientific data — settings where
   the Fourier option literally doesn't exist without a lossy preprocessing step.
   We now know a grid-free detail objective works there, and how well.
2. **For the science itself**: we imported a whole toolbox (the "geometry of level
   sets" of random fields — crossing counts are just its first tool) into neural
   network training, and published exact, reproducible measurements of where it
   helps, where it ties, and where it hurts. Negative and boundary results, stated
   plainly, save the field wasted effort — and the positive results (texture,
   robustness, composability with better architectures) point at where to dig next.
3. **A model of honest reporting**: the paper's claims were pre-registered in
   spirit ("here's the kill condition"), the tournament information was strictly
   equalized, and every number in the paper regenerates from raw logs in this
   repo with one script.

---

## Part 2 — The whiteboard version (for presenting without notes)

This section is a script. Draw what it says to draw, in order. Formulas are kept
to the four that matter. Anticipated hard questions at the end.

### 2.1 Frame the problem (2 minutes)

Draw a wiggly 1D signal (a few big slow humps with fast ripples on top). Next to
it, draw a smooth curve through just the humps.

> "An INR is a network `f_θ(x) → value`: feed in a coordinate, get the signal.
> Spectral bias: trained on the true signal [point at wiggly], gradient descent
> finds the smooth version [point at smooth] fast, and takes forever to add the
> ripples. Every fix either changes the architecture (SIREN, Fourier features) or
> adds a loss term that *measures missing detail* and pushes on it."

Draw a grid of dots, then a scattered splatter of dots.

> "The standard measuring stick is the FFT — which needs *this* [grid]. Real data
> is often *this* [splatter]: point clouds, sensors, partial observations. To use
> an FFT loss here you first interpolate the splatter onto a grid, which smears
> away exactly the high frequencies you wanted to police. That's the gap we aim at."

### 2.2 The object: crossing density (3 minutes)

Redraw the wiggly signal. Draw a horizontal dashed line at height `u` and put fat
dots wherever the signal pierces it.

> "Pick a level u. Count crossings per unit length: call it c(u). Claim: c(u) is a
> frequency meter."

Write **Rice's formula (1944)** — for a stationary Gaussian process:

```
c(u) = (1/π) · √(λ₂/λ₀) · exp(−u² / 2λ₀)
```

> "λ₀ is the signal's variance; λ₂ is the *derivative's* variance. In spectral
> terms λ₀ = ∫S(ω)dω and λ₂ = ∫ω²S(ω)dω, so √(λ₂/λ₀) is literally the RMS
> frequency of the signal. More treble → more crossings, at every level,
> in closed form. No Fourier transform was harmed in making this measurement."

Key preemption — say this before anyone asks:

> "Zero-crossing rate as a frequency feature is textbook signal processing
> (Kedem '86). What's new is using it as a *differentiable training objective*
> for neural fields, and measuring whether it does the job frequency losses do —
> off the grid."

### 2.3 The deterministic backbone: co-area (2 minutes)

> "Rice needs 'stationary Gaussian'. Our images aren't. So the loss actually
> stands on a *deterministic* identity — the co-area formula. No probability
> anywhere:"

```
∫ g(f(x)) · ‖∇f(x)‖ dx  =  ∫ g(u) · (size of the level set {f = u}) du
```

Draw a 2D contour plot (like a topographic map).

> "In 2D, 'size of level set' = total *length of the contour line* at height u.
> The formula says: integrate any function of the field's value, weighted by
> gradient magnitude, and you're secretly integrating over contour lines. Choose
> g = a narrow Gaussian bump centered at u, and the left side becomes something I
> can estimate by averaging over ANY set of sample points — my scattered splatter
> included."

### 2.4 The estimator and the loss (3 minutes) — the core slide

Write the estimator:

```
ĉ_ε(u) = (1/N) Σᵢ  δ_ε( f_θ(xᵢ) − u ) · ‖∇ₓ f_θ(xᵢ)‖
```

Walk it left to right:

> "Average over my training points, wherever they are. δ_ε is a Gaussian bump of
> width ε — 'is the field near level u here?'. Times the gradient norm — that's
> the co-area weighting, and autograd gives it to me *exactly*, at any point, no
> finite differences. Everything is differentiable in θ, so this isn't just a
> measurement, it's a loss."

Write the loss:

```
L_KR = (1/L) Σⱼ  [ ĉ_ε(uⱼ) − c_gt(uⱼ) ]²  /  [ c_gt(uⱼ) + mean(c_gt) ]²
total = MSE + β · L_KR
```

Three design decisions, one line each:

> "Levels uⱼ sit at quantiles of the ground-truth values, so every level has data.
> The target profile is computed *on the same batch of points* — estimator and
> target share their Monte-Carlo noise, so it partially cancels. And it's
> normalized because crossing densities scale linearly with frequency content, so
> raw squared error scales *quadratically* — unnormalized, one β can't serve two
> different signals. We learned that the hard way; it's in the paper."

The one conceptual sentence to land:

> "Note what this loss is: L numbers per batch — a *distributional* statistic. It
> says 'you need this much contour length at each gray level', never *where*. The
> MSE term places it. That's both its weakness and its superpower."

### 2.5 The experiments — draw the ladder (4 minutes)

Draw a vertical PSNR ladder for the scattered natural-image test (the "blobs"
sampling, 12.5% of pixels, clumped):

```
21.8  ← plain interpolation of the samples ("the ceiling")
21.6  ← + FFL (Fourier loss on interpolated grid)
21.6  ← + Sobolev (match gradients pointwise)
21.1  ← + Kac–Rice (ours)
18.9  ← MSE only
```

> "Three honest observations. One: every auxiliary loss is a big win over plain
> MSE — this regime is where detail objectives matter. Two: we hoped to beat the
> Fourier loss off-grid; we *tied* it — within half a dB, slightly better SSIM,
> slightly worse PSNR. Say it plainly: the headline hypothesis did not confirm.
> Three: everything crowds under the interpolation ceiling — at this sample
> budget, no loss extracts more than the samples' information content."

Now the two twists. Draw a texture patch (hatch marks):

> "Twist one: on a *statistically homogeneous* texture, we win — 27.2 vs 26.6
> against FFL, and we edge past the ceiling. That's the regime Rice's theory is
> actually *about* — stationary fields. Edges are non-stationary; texture is the
> theory's home turf, and the data agrees with the theory about where its own
> claim applies. I find that genuinely satisfying."

> "Twist two — my favorite experiment: everyone's gradient targets come from the
> smeared interpolant. Feed Sobolev *perfect* oracle gradients instead: it
> improves, 21.6 → 21.8. It was bottlenecked on target quality. Feed ours perfect
> gradients: nothing — 21.1 → 20.9. Ours only consumes gradient *statistics*, so
> the sloppiness was already washing out. It's the only method here that doesn't
> care how good your derivative estimates are. On scattered real-world data,
> derivative estimates are never good."

And the boundary result (draw a dense grid, all pixels known):

> "On a full grid with dense supervision, MSE alone wins, and gradient-target
> losses — ours *and* Sobolev — actively hurt, about −10 dB, because
> finite-difference targets conflict with exact pixel reconstruction at high
> fidelity. Detail objectives are medicine for scarcity, and we mapped the
> dosage boundary."

### 2.6 The close (30 seconds)

> "So: a 1940s telephone-noise formula becomes a modern training objective. It's
> validated to within 5% of exact crossing counts, it's grid-free, FFT-free,
> robust to target noise, ties the state of the art on photos, beats it on
> texture, composes with better architectures, and we published exactly where it
> fails. Crossing counts are the *first* statistic from the level-set-geometry
> toolbox — critical points, excursion-set topology, Euler characteristics are
> sitting right behind it, and point-cloud SDFs are the obvious next target,
> because there a grid never existed in the first place."

### 2.7 Hard questions you will get, and the answers

**"Why not just always use Sobolev? It won or tied everywhere off-grid."**
Fair — on PSNR it did. But its accuracy tracks gradient-target quality (the
oracle experiment proves it), it carries the same finite-difference bias that
cost −10 dB on grids, and on SIREN, ours edged it (21.59 vs 21.50). When your
gradient estimates are trustworthy, use Sobolev; when they're garbage — point
clouds, sparse sensors — the distributional loss doesn't care. That's the
decision rule the paper supports.

**"Isn't L=16 numbers per batch a laughably weak learning signal?"**
Yes, deliberately. Weak = robust; that's one tradeoff, not two facts. It's an
auxiliary drive, not a standalone objective — MSE supplies location, we supply
'how much detail should exist'. And empirically 16 numbers per batch moved
reconstructions by +2.3 dB, so 'weak' is doing a lot of work.

**"Couldn't the network cheat — add fake wiggles anywhere to inflate crossings?"**
The MSE anchor punishes wiggles in wrong places; the crossing target is per-level,
so surplus at one gray level doesn't pay for deficit at another; and the
oracle-vs-estimated result shows the failure mode runs the other way — demand
*more* crossings than MSE can place (sharper targets) and quality drops slightly.

**"How sensitive is the bandwidth ε?" (the KDE question)**
Wide plateau: anything ≥ 0.15·σ of the signal values works identically; only
starving the kernel (0.05σ) hurts, because most batch points stop contributing
gradient. L saturates at 16 levels; β tolerates a 20× range because of the
relative normalization. All swept in the paper.

**"Gaussianity is fake for images. Doesn't that sink the theory?"**
The Gaussian Rice formula is *motivation*; the loss stands on the co-area
formula, which is deterministic and assumption-free, plus an *empirical* target —
we match the ground truth's measured crossing profile, never the Gaussian
prediction. The validation figure shows the estimator tracking exact counts even
where the field is visibly non-Gaussian.

**"Cost?"**
2.1× an MSE iteration (double backprop through the gradient norm) — same as
Sobolev, benchmark script in the repo. FFL is cheaper per iteration but needs
the grid.

**"Why does it lose on natural images? Give me the mechanism."**
Natural images are edge-dominated: their detail is *localized*, phase-critical
structure. Our statistic pools over space, so it can't tell the network where the
edge mass goes — FFL's full spectrum and Sobolev's pointwise targets carry more
constraint per batch on that content. On statistically homogeneous content the
pooling loses nothing — and we win there. Content-dependence isn't an excuse;
it's the theory's own prediction about its domain of validity.

**"What would change your mind / what's next?"**
Point-cloud SDFs (no grid ever exists — the structural advantage should finally
bind), scheduled crossing targets that inject prior spectral knowledge instead of
re-encoding sample information (that's how you might break the interpolation
ceiling), and richer level-set statistics than crossing counts.

### 2.8 Cheat sheet — the five numbers to remember

| Fact | Number |
|---|---|
| Estimator accuracy vs exact counts | within 5% |
| Scattered image: aux-loss gain over MSE-only | +2.3 to +3.0 dB (PE-MLP) |
| Ours vs FFL on natural image (scattered) | −0.2 to −0.5 dB PSNR, +SSIM: a tie |
| Ours vs FFL on texture | **+0.6 dB (we win)** |
| Oracle gradients: Sobolev vs ours | Sobolev +0.2 dB, ours ±0 (robustness) |
