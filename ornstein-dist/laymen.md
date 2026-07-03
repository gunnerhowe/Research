# What We Did, In Plain English

*A no-math explanation of this project and the paper that came out of it.*

---

## The problem, starting from zero

Scientists increasingly use AI to imitate complicated physical systems — weather,
climate, turbulent fluids, and so on. Instead of running a slow, expensive
physics simulation, you train an AI "stand-in" (the technical word is
**surrogate**) that behaves like the real thing but runs much faster.

Here's the catch: these systems are **chaotic**. Chaos means tiny differences
snowball. Two weather simulations that start almost identically will look
completely different a couple of weeks in — that's the famous "butterfly
effect," and it's not a flaw, it's just what these systems do.

That creates a weird grading problem. You can't grade an AI stand-in by asking
"did it predict the exact same future as the real system?" — because even a
*perfect* copy of the real system would fail that test. Perfection doesn't look
like matching; matching is impossible past a short horizon.

## How people grade these AIs today — and the hole in it

So the field settled on a sensible-sounding fix: don't compare the exact
predictions, compare the **overall habits**. Run both systems for a long time
and check: do they visit the same places, in the same proportions? Think of it
like a long-exposure photograph — leave the camera shutter open for hours, and
you get a glowing picture of everywhere the system spends its time.

If the AI's long-exposure photo matches the real one, the AI is declared good.

Here's our starting observation: **a photo is not a movie.** A long-exposure
photo of a dancer tells you where on the stage she spent her time. It tells you
*nothing* about the dance — how fast she moved, in what order, with what rhythm.
Two completely different dances can leave the *exact same* photo.

The same is true for these physical systems. And that means the field's
standard report card has a blind spot: an AI can ace the photo test while
getting the actual *motion* — the dance — completely wrong.

## What we did

### 1. We found the right measuring tool in an unexpected place

Fifty years ago, mathematicians studying a very abstract question ("when are
two random processes secretly the same process?") invented a way of measuring
the distance between two *movies* rather than two photos. It's called
**Ornstein's d-bar distance**. It has beautiful theory behind it, but nobody
had ever used it to grade AI models of chaotic systems. We took it off the
shelf, dusted it off, and turned it into a practical tool.

The intuition: imagine playing the two movies side by side, and you're allowed
to line them up as cleverly as you possibly can. The d-bar distance asks:
*even with the best possible alignment, what fraction of the time do they
still disagree?* If two systems dance the same dance, you can sync them almost
perfectly. If the dances are truly different, no amount of clever syncing
helps — and the tool tells you so.

### 2. We made it actually computable

The pure math version is impossible to compute directly. Our recipe:

- **Simplify the signal.** Instead of tracking exact numbers, record a simple
  stream of letters — for the famous "butterfly" system we studied, literally
  just which wing of the butterfly the system is on: left, left, right, left...
- **Compare the phrases.** Chop both letter streams into short phrases (2
  letters, 4, 8, 16...) and ask how differently the two systems "speak" —
  do they use the same phrases with the same frequencies?
- **Build in a lie detector.** Any measurement on finite data has noise. So we
  always run the same comparison on *two recordings of the identical system* —
  which should show zero difference — and report that alongside. If our
  "difference" isn't clearly bigger than that baseline, we don't count it.
  Our tool is honest about its own limits by construction.
- **Add an independent second opinion.** A classic theorem lets us certify a
  minimum distance using a completely different calculation (based on how
  "surprising" each system's behavior is, moment to moment). When both methods
  agree, you can trust the answer.

### 3. We built counterfeits designed to fool the photo test — and caught them

This was the heart of the project. We manufactured fake systems that are
*mathematically guaranteed* to pass the photo test:

- **The remix.** A doctored version of the real signal that keeps the exact
  same distribution of values and the same rhythm content, but scrambles the
  actual dynamics. (A standard trick from signal processing.)
- **The fast-forward.** The real system played at 2x speed. Same photo,
  *exactly* — every place gets visited in the same proportions — but obviously
  a different dance.

The photo test scored both counterfeits as **flawless**. Our test flagged both
at **enormous confidence** — the measured difference was up to fifty times
larger than the noise baseline. And on genuine copies of the real system, our
test correctly reported "no difference." We repeated everything eight times
with fresh data to make sure none of it was luck, and it held on a second,
much more complex system (a model of flame-front turbulence) too.

### 4. We tested it on a real AI, not just counterfeits

We trained a family of small AI models to imitate the butterfly system. Most
learned it well — and importantly, our test *agreed* they were good (a smoke
alarm that goes off during every shower is useless). Then we deliberately
handicapped one model in a subtle, realistic way. The result was the whole
point of the project in one example: the handicapped AI **still passed the
photo test**, but its dance was wrong — and our test caught it clearly.

### 5. We were honest about what it can't do

One counterfeit slipped past: the real system **played backwards**. Playing a
movie in reverse keeps the photo, the rhythm content, *and* the
surprise-per-second identical — it's the deepest possible fake — and at the
resolution we used, our tool couldn't see it either (we explain exactly why in
the paper, and what it would take to catch it). Real science reports its
misses. This is ours, stated plainly, along with the tool's other limits.

## What we accomplished, in one sentence

**We gave the field a practical, honest way to check whether an AI imitation
of a chaotic system gets the *motion* right, not just the *map* — and showed
that the standard test misses exactly this, while ours catches it.**

## Why this matters

- **Better report cards for science AI.** AI emulators of weather, climate, and
  fluids are being deployed now. Decisions get made on their output. A model
  that hangs out in all the right places but moves wrongly will get event
  *timing*, *persistence*, and *variability* wrong — think how long a heat wave
  lasts, how quickly storms follow each other. Today's standard evaluation
  cannot see that failure. Ours can.
- **Trust through honesty.** Every number our tool reports comes with a
  built-in noise baseline and, where possible, an independent certificate.
  You know not just the answer but how much to believe it.
- **Cheap.** No supercomputer needed — everything in the paper runs on an
  ordinary desktop CPU in minutes, and the method works with surprisingly
  little data.
- **A path to better AI, not just better grading.** The natural next step is
  to use this measurement *during training*, so models are directly rewarded
  for getting the dynamics right — not just the long-exposure photo. That
  could produce a next generation of science AIs that are faithful in a
  deeper sense than today's.
- **A bridge between fields.** A profound 50-year-old body of pure mathematics
  turns out to answer a very current AI question. We suspect there's more
  where that came from.

## Where everything lives

- The full technical story: `paper/main.pdf`
- The verified math notes: `docs/RESEARCH_NOTES.md`
- The code and every raw result: this repository (see `README.md` to reproduce
  everything with a few commands)

---
---

# Part II — Whiteboarding It Cold

*Everything you need to stand at a whiteboard, with nothing in front of you, and
teach this project to a sharp interviewer who will push back. The goal is not to
memorize a script — it's to understand the spine well enough that every detail can
be re-derived on the spot. If you can draw the four pictures and write the one
definition, the rest falls out.*

## The mental model to internalize first

The entire project is one sentence: **"The invariant measure is a photo; the dynamics
are the movie; we imported a 50-year-old metric that compares movies, made it
computable, and showed it catches fakes the photo-test declares perfect."**

If you get flustered, return to that sentence and rebuild outward. Every result is a
consequence of it.

## Three versions, escalate on demand

Interviewers probe at different depths. Have all three ready and let them pull you
deeper.

**30 seconds (the hook).**
"You can't grade an AI imitation of a chaotic system by comparing predicted
trajectories — chaos makes even a perfect copy diverge. So the field grades them on
long-run statistics: does the AI visit the same states as the real system? But that's
a static snapshot — where the system spends time, not how it moves. Two systems can
have identical snapshots and completely different dynamics. I took Ornstein's
d-bar distance from ergodic theory — a metric that measures whether two systems are
the *same process*, not the same distribution — turned it into a practical estimator,
and showed it catches surrogates that fool every standard metric."

**2 minutes (the arc).** The 30-second version, plus: *how* you make it computable
(symbolize the trajectory into letters, compare the statistics of short blocks with
optimal transport, and the math guarantees these block-comparisons climb to the true
d-bar); the *honesty machinery* (every number sits against a noise floor, and an
independent entropy calculation certifies a lower bound); and the *decisive result*
(hand-built fakes that match the invariant measure exactly — a time-rescaled system,
and a phase-scrambled "IAAFT" surrogate — are invisible to the standard metrics but
lit up by d-bar, while a genuine copy reads zero).

**10 minutes (the full walkthrough).** Draw the four pictures below in order. That IS
the talk.

## The one definition you must be able to write

> **d̄(μ, ν)  =  inf { ℙ_λ(X₀ ≠ Y₀)  :  λ ∈ J(μ, ν) }**

(Read: d-bar of μ and ν equals the smallest value of "probability that X₀ ≠ Y₀,"
taken over every joining λ of the two processes.)

Say it in words while you write it: *"Take the two processes. A **joining** is any way
of running them side by side in lockstep that preserves each one's own statistics.
Over all such alignments, d-bar is the smallest achievable fraction of time they
disagree. If they're the same dance, you can sync them almost perfectly and d-bar is
near zero. If the dances truly differ, no alignment helps."*

Then write the computable face and the three facts that make it work:

> **d̄ₙ(μ, ν)  =  OT( Pₙ, Qₙ ; Hamming cost )**

where Pₙ, Qₙ are the distributions of length-n blocks of symbols, and OT is optimal
transport. The three facts (you should be able to justify each in a sentence):

1. **d̄ₙ ≤ d̄ for every n** — each block-length estimate is a *certified lower bound*.
   (A joining of the full processes restricts to a coupling of n-blocks, so the block
   problem can only be easier.) → *You never overstate the distance.*
2. **d̄ₙ climbs to d̄ along doubling n = 1, 2, 4, 8, …** — because n·d̄ₙ is superadditive
   (Fekete's lemma). → *A rising-then-flat curve is the theoretically predicted
   signature; the plateau is your estimate.*
3. **d̄₁ = the distance between the single-symbol distributions = the invariant-measure
   comparison itself.** → *So everything at n ≥ 2 is pure dynamics. The curve literally
   separates "photo" (n = 1) from "movie" (n ≥ 2).*

That third fact is the whole thesis made quantitative. Land it.

## The four pictures to draw

**Picture 1 — Why trajectory error fails.** Two nearly-identical starting points; two
curves that track briefly then peel apart exponentially. Label the gap "meaningless
after a few Lyapunov times." Punchline: *a perfect copy fails this test, so matching
can't be the goal.*

**Picture 2 — Photo vs movie.** The Lorenz butterfly (two lobes). Write the
symbolization: `sign(x)` = which lobe = a stream like `L L R L R R L…`. Next to it draw
the histogram of where the system sits. Say: *"the histogram is the invariant measure —
the photo. The letter-stream's temporal structure is the movie. The standard metrics
only see the histogram."*

**Picture 3 — The d-bar curve.** X-axis n = 1, 2, 4, 8, 16, 32 (log). Y-axis d̄ₙ.
Draw a shaded grey band along the bottom = the **noise floor** (same-vs-same). Draw the
real-vs-fake curve starting near zero at n = 1, rising, then flattening well above the
band. Annotate: "d̄₁ ≈ 0: the fake matches the measure. Signal appears only at n ≥ 2:
the dynamics." This one picture contains the method, the result, and the
honesty device at once.

**Picture 4 — The decisive table.** Rows = surrogates. Columns = {Wasserstein-on-measure,
power spectrum, **d-bar**}. Fill with ✓ (caught) / ✗ (missed):

| surrogate | W-measure | spectrum | **d-bar** |
|---|---|---|---|
| independent copy (neg. control) | ✗ | ✗ | ✗ (correctly reads ~0) |
| IAAFT (marginal + spectrum matched) | ✗ | ✗ | **✓** |
| speed×2 (identical invariant measure) | ✗ | ✓ | **✓** |
| ρ=32 (different attractor, pos. control) | ✓ | ✓ | ✓ |

The story of the paper is the two middle rows: **d-bar is the only column that lights
up where the fake was built to pass.**

## Numbers to have cold

Get these *right* — a wrong multiplier at the whiteboard reads as not understanding your
own result. (These are the corrected, error-barred, 8-seed figures.)

- **Lorenz IAAFT: d-bar separation ≈ 0.10, about 28× the noise floor** (8 seeds).
  Its Wasserstein-on-marginal, power spectrum, and autocorrelation all equal the
  negative control — it defeats every standard metric at once.
- **Lorenz speed×2: ≈ 22× the floor**, while its state-space Wasserstein is *at/below*
  the same-vs-same floor (i.e. the invariant-measure metric certifies it as perfect).
- **Negative control (independent copy): ~0** (well inside the floor). **Positive
  control (ρ=32): caught by everything.**
- **Works on little data:** stable estimate down to **10⁴ samples** (~6× floor there).
- **Kuramoto–Sivashinsky (spatiotemporal chaos): speed×2 ≈ 20× floor**, replicating the
  Lorenz story on a PDE.
- **The learned-model result: a linear-readout ESN** with excellent one-step error and
  near-floor invariant-measure distance, but wrong dynamics — **d-bar catches it at 2–4×
  its floor, while every healthy model sits within ±1 floor of zero.**
- Orders of magnitude for credibility: truth symbolic entropy ≈ **0.22 bits/symbol**
  (Lorenz), Lyapunov λ₁ ≈ **0.9 nats/time** (matches the literature).

If you only memorize one: **"IAAFT is invisible to the standard metrics but 28× the
noise floor under d-bar."**

## The certificate (your second, independent line of evidence)

Be ready for "how do I know the transport number isn't just noise?" Answer: there's a
*completely separate* calculation that agrees. Entropy rate is continuous with respect
to d-bar, with an explicit modulus (a Fano-type inequality):

> **|Hₙ(X) − Hₙ(Y)| / n  ≤  g(d̄),   where  g(δ) = H_b(δ) + δ·log₂(|A| − 1)**

(Hₙ = entropy of the length-n blocks; H_b = the binary entropy function; |A| = alphabet
size.) Invert g and you get a **certified lower bound on d-bar from nothing but a
symbol-counting entropy gap** — no transport at all. It's a one-hour go/no-go you run
before the expensive part. (Footnote you can drop to show depth: the naive "d̄ ≥ |Δh|"
you might guess is *dimensionally wrong* — d-bar is a fraction in [0, 1],
entropy is in bits; you must invert the Fano modulus. Catching that was one of the real
corrections in the work.)

## Why each experiment exists (the design logic)

Interviewers love "why did you build it that way." Each surrogate kills a specific
baseline *by construction*, with zero training:

- **IAAFT** (phase-scramble keeping the exact value distribution and spectrum) → kills
  the spectrum baseline *and* the marginal baseline simultaneously.
- **speed×2** (integrate dx/dt = 2·f(x)) → *exactly* the same attractor and invariant
  measure, so it kills the measure baseline that speed-scrambling can't.
- You need *both* because each defeats a different standard metric; together they show
  d-bar catches what neither baseline can.
- **Two controls** (independent copy = should read zero; ρ=32 = should read large) prove
  the metric isn't just "always fires" or "always quiet."
- **8 seeds** so every number has an error bar. **KS** to show it's not Lorenz-specific.
  **ESN** to show it fires on a *real trained model*, not just hand-built fakes.

## The honesty layer — lead with this, it's your strongest card

At an interview (especially Anthropic), the credibility move is that you can state
exactly what the method *cannot* do, and that honesty is engineered into the tool:

- **Every number is reported against a noise floor** (the same comparison run on two
  independent copies of the same system, which is not exactly zero at finite data).
  Signal counts only if it clears the floor. This turns "is it just noise?" from a
  worry into a published quantity.
- **Detections are certified lower bounds; non-detections are explicitly inconclusive.**
  You never claim more than the math guarantees.
- **The honest null:** the time-reversed system — same measure, same spectrum, *same
  entropy* — is invisible to the binary-partition estimator. You can even explain *why*
  from first principles: for any stationary binary process the pair-statistics are
  reversal-symmetric (P(01) = P(10) by counting transitions), so reversal only shows up
  at 3-blocks and higher, below the floor at this resolution. And you name the baseline
  that *does* catch it (a delay-embedded Wasserstein — the strongest competitor, which
  you report honestly rather than hide).
- **Estimation limits:** cite Ornstein–Weiss — universal consistent estimation of d-bar
  is only possible for a restricted class of processes, so you *don't* claim a universal
  estimator; you claim certified lower bounds, which is exactly the right, defensible
  shape for an evaluation metric.

Saying all of this unprompted is the single highest-signal thing you can do.

## Hard questions and crisp answers

- **"Isn't this just comparing n-gram statistics?"** — The *computation* is n-gram
  optimal transport, yes. What makes it more: ergodic theory guarantees the limit is a
  true metric on the *process* (isomorphism-complete within the relevant class), it
  comes with certified lower bounds, and an independent entropy certificate agrees. The
  n-gram view is the computable face of a principled object, not a heuristic.
- **"Why not just delay-embed and run Wasserstein?"** — That's actually my strongest
  baseline and it does catch some cases. But it mixes static and temporal information at
  one fixed lag, gives no block-length decomposition (which timescale is wrong?), no
  certified bound, and it doesn't read a clean zero on a genuine copy. d-bar is
  complementary and better-behaved, not a strict superset — and I say so in the paper.
- **"Partition dependence — you're symbolizing, isn't that arbitrary?"** — Real concern,
  and the honest framing is: d-bar on symbols is a *lower bound* on the true dynamical
  discrepancy, so a coarse partition can only hide differences, never invent them. I fix
  one partition on the ground truth, apply it identically to both systems, and show the
  conclusions are stable across three partitions and three timescales — and that finer
  partitions detect *more*, never flip the ranking.
- **"Who trains a 2×-speed surrogate? That example seems artificial."** — It's a
  proof-of-concept that the invariant measure *literally cannot* see a real dynamical
  difference — a clean existence proof of the blind spot. The ESN linear-readout result
  is the same failure arising in an actual trained model.
- **"How is this different from just matching Lyapunov exponents?"** — Lyapunov is a
  single number, it presumes determinism (on IAAFT the estimator degenerates — I show
  the fit falls apart), and matching it is necessary but nowhere near sufficient for two
  processes to be equivalent. d-bar is a full metric on the process.
- **"Is d-bar actually a metric?"** — Yes: symmetric, satisfies the triangle inequality,
  and zero iff the process laws coincide. That's Ornstein's original result; it's why
  the object is trustworthy as an evaluation distance.
- **"What's the compute?"** — Everything is CPU, minutes. Exact optimal transport
  (network simplex) on deduplicated blocks; a full curve is seconds to tens of seconds.

## What NOT to overclaim (say the ceiling out loud)

Stating your own limitations is a maturity signal, not a weakness:

- This is **a better evaluation metric and a candidate training signal — substantial in
  a hot area, not a paradigm shift.** The paper says exactly that. Don't inflate it.
- It's **complementary discrimination**, not "beats everything everywhere." The
  delay-embedded baseline catches things d-bar (at binary resolution) misses.
- The training-signal application (differentiable d-bar via Sinkhorn) is **proposed and
  motivated, not yet built** — be clear it's the next step, not a result.
- The demonstrations are **Lorenz and KS scale**, not full climate/turbulence models yet.

Owning these makes every claim you *do* make more believable.

## The bridge to Anthropic (if they ask "why do we care?")

Draw the analogy, carefully and without overreaching: *the core problem — you cannot
evaluate a system by exact-output matching when its outputs are effectively
unpredictable, so you evaluate distributional and behavioral properties instead, and
you had better know when your metric has a blind spot* — is exactly the shape of
evaluating generative and language models. A metric that (a) targets the property you
actually care about rather than a convenient proxy, (b) carries a built-in noise floor
so you know when to believe it, and (c) reports certified bounds and honest nulls, is a
philosophy of evaluation, not just a chaos result. That mindset — rigorous, calibrated,
honest about what it doesn't measure — is the transferable thing, and it's the thing
worth signaling.

## Your 90-second closing (rehearse this)

"So the contribution is threefold. Conceptually: the field evaluates chaotic surrogates
on a static object — the invariant measure — and I showed that misses an entire class of
dynamical failure, using ergodic theory that had never been pointed at this problem.
Practically: I built an estimator that's computable, cheap, works on ten-thousand
samples, and — crucially — reports every number against a noise floor with an
independent entropy certificate, so you know exactly how much to trust it. And honestly:
I mapped its blind spot, explained it from first principles, named the baseline that
beats it there, and corrected two of my own errors along the way. It's a better,
more honest ruler for a problem people are actively building models for — and the
natural next step is to make it a training signal, not just a grade."

Then stop talking. Let them ask.
