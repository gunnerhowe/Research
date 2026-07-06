# The idea in plain terms: making a chip's noise *remember* for you

## The problem

Teach a neural network task A, then task B, and it forgets A. This is
"catastrophic forgetting." The standard fix is to **anchor** the weights that
mattered for A: after learning A, note which weights were important and add a
spring that pulls them back if task B tries to move them.

Now put this network on **analog neuromorphic hardware** — a chip that does the
arithmetic with physics (currents, charges) instead of digital logic. These chips
are fast and astonishingly energy-efficient, but they are **noisy**: every weight
jitters a little, all the time, because of heat and electrical crosstalk. Normally
this noise is a nuisance you calibrate away — an "accuracy tax."

**Our question:** what if you could *aim* that noise so it does the remembering for
you — turn the tax into a dividend?

## The trick: condition on *not forgetting*

Here is the move. Think of an important weight as a particle sitting at its "task-A
value," free to wobble. Draw two lines on either side of it — a **barrier** — and
say: "the memory is safe as long as the particle never crosses these lines."

There is a classical piece of mathematics (a **Doob *h*-transform**) that answers:
"if I *insist* this randomly-wobbling particle never crosses the barrier, how must
it move?" The answer is that it picks up an extra inward push — a restoring force
that (i) points back toward the memory, (ii) becomes infinitely strong right at the
barrier, and, the crucial part, (iii) **grows with the amount of noise**.

That third point is the whole paper. On a noisy chip, more noise doesn't just mean
more wobble — if you steer it through this conditioning, more noise means a
**stronger memory-preserving push**. The noise powers the consolidation.

## The prediction we bet the project on

If the noise both *hurts* (random wobble knocks the weight around) and *helps*
(stronger steering holds it in place), and the helping part grows faster with noise
than the hurting part — then there should be a **sweet spot**. Retention (how much
of the old tasks you keep) should go **up** as you add noise, reach a peak, then
come back down. An **inverted-U**.

Ordinary anchoring can't do this. Add noise to a plain spring-anchor and you only
get more wobble — retention just goes **down**. So the inverted-U is a fingerprint:
if it appears, the *conditioning* is doing something real; if retention only ever
falls with noise, our idea is just anchoring with extra steps, and there's no paper.

We wrote that down as a **go/no-go test before running it** (so we couldn't fool
ourselves), and it **passed**: on a standard forgetting benchmark, adding the right
amount of noise raised retention by about **11 percentage points**, peaking at a
noise level well above zero — while the plain-anchor methods (the ones our idea
shares its "spring" with) all just got worse with noise.

## What is genuinely new, and what isn't (we're strict about this)

- **Not new:** the spring itself — pulling important weights back toward their
  stored value. That exact rule already exists (it's called OUA, MESU, EWC). We say
  so up front and take no credit for it.
- **New (and the entire contribution):** (a) framing consolidation as *conditioning
  a diffusion on not crossing a barrier* — a tool that, as far as we can find, has
  only ever been used in image generation and physics simulations, never for
  synapses — and (b) the falsifiable inverted-U it predicts. Either one alone is not
  enough; it's the pair that's the paper.

## The honest catch

The dream version of this result is measured on a **real** BrainScaleS-2 chip, using
its *actual* physical noise, with the electricity meter running — because then the
efficiency argument becomes concrete: the noise is free physics, whereas a normal
computer has to *spend energy generating* random numbers to imitate it.

We could not run the real chip (no hardware access in this environment). So we did
the next best, honest thing: we built a **faithful software model** of that chip's
noise — colored (correlated in time), signal-dependent, with fixed per-device
quirks, and 6-bit weights — and checked the inverted-U survives it. **It does.** But
we label this an *emulation* everywhere, we report **no** measured chip numbers or
energy, and we list the on-silicon run as the one experiment still owed. If the real
noise turns out to have the wrong texture to consolidate, that result would fail —
and we say exactly that.

## Why it could matter

If it holds on silicon, the usual trade-off flips. Today, noisier analog hardware is
"worse." Under this mechanism, a noisier chip has a *stronger* free consolidation
force — the hardware gets *better* at not-forgetting as it gets noisier, and a
conventional processor has to burn energy to fake what the chip gets from physics.
