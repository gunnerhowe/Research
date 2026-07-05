# Rice's Formula for Event-Driven Networks, In Plain English — and How to Whiteboard It Cold

Paper #4 of the Kac–Rice program. One idea, honestly measured, with two positive
results and one clean negative. This document has two parts: **Part 1** is the
explanation you could give anyone at a dinner table; **Part 2** is everything you
need to stand at a whiteboard in front of Anthropic and own it — including the
objections they'll throw and exactly how to answer them.

---

# Part 1 — What we did, for anyone

## The problem, in one picture

Most neural networks are wasteful in a specific way: every time step, every neuron
recomputes its output from scratch — even if its value barely changed. A whole
family of "**event-driven**" or "**delta**" networks fixes this. The rule is
simple: **a neuron only does work when its output changes by more than a set amount
(a threshold).** If a neuron is sitting still, it costs nothing. If it jumps, it
pays for one update. On the right hardware (phones, hearing aids, always-on "hey
device" chips, neuromorphic processors) this saves real battery, because most
signals — audio, video, sensor streams — hold still most of the time.

There's a catch that has been there since the idea was invented in 2016, and nobody
had solved it: **you can't predict how much energy you'll save until you build the
thing and measure it.** Every paper in this field tunes the threshold by trial and
error — try a value, run the whole network, count the events, repeat — and reports
the savings only *after* the fact. There was no theory. No formula. No way to say
"set the threshold here and you'll get that many events" without running it.

## The one idea that fixes it

Here is the observation the whole paper turns on, and it's almost embarrassingly
simple once you see it:

> **A neuron "changing by more than a threshold" is the exact same event as a
> wiggly line crossing a horizontal level.**

Draw a neuron's value over time as a wiggly curve. Draw a horizontal line. Every
time the curve crosses that line, that's a threshold-crossing — an *event*, a unit
of energy spent. So **counting energy in a delta network is literally counting how
often a signal crosses a line.**

And counting line-crossings is a *solved problem from 1944.* Stephen Rice, at Bell
Labs, studying telephone static, proved a formula that tells you exactly how often a
random signal crosses any level, using just two numbers about the signal (how big it
swings, and how fast it wiggles). This is the same 1944 formula our whole research
program is built on — we'd already turned it into a training tool for other kinds of
networks in papers #1–#3. Here it finally lands in the one place where crossings
*are the cost, by definition.*

So the pitch is: **the century-old math of line-crossings is exactly the missing
energy model for modern event-driven AI.** That's the thesis.

## We placed three bets. Two won. One lost — cleanly, and usefully.

### Bet 1 — "Can we PREDICT the energy without running the network?" → **Won.**

We trained three normal networks (a keyword spotter for voice commands, a
digit-reader that sees an image row-by-row, and a small text-predictor), then asked:
using only the statistics of their neuron activity, can we predict how many events a
delta version would fire — *without* building the delta version and measuring?

Yes. A calibration-based predictor nailed it to **within ~2–3%** on all three
networks. The energy story you previously had to *measure* is now something you can
*compute*. (Rice's Gaussian formula works too, but only when neuron activity is
"bell-curve-shaped"; we mapped exactly where it breaks — heavy-tailed neurons throw
it off by up to 75% — and showed the calibration predictor doesn't care.)

### Bet 2 — "Can we USE that prediction to skip the trial-and-error tuning?" → **Won.**

The old way to set a delta network's thresholds is a brute-force search: try dozens
of configurations, run and measure each. We instead *invert* our prediction — given
an energy budget, solve for the thresholds that hit it, in one shot. Result: **our
one-calculation method matched the quality of a 48-try brute-force search, at about
2% of its tuning cost**, and hit the requested energy budget within 4%. For a field
that currently burns compute tuning these knobs, that's a straightforward practical
win.

### Bet 3 — "Can we TRAIN the network to be even more efficient?" → **Lost — cleanly.**

This was the home-run swing. The idea: instead of just tuning thresholds *after*
training, add our crossing-counter into the training itself as a "budget" penalty,
so the network *learns* to be temporally sparse. We built it (with a mathematically
guaranteed "do nothing when already under budget" property), trained it on 8 random
seeds, and compared honestly.

**It did not beat simply thresholding a normally-trained network.** And neither did
the three obvious alternatives (an L1 penalty, an activity penalty, or just
fine-tuning longer). The scores were a statistical tie (90.4% vs 90.5%; the
difference was noise, p ≈ 0.31).

But here's what makes the loss *valuable* rather than embarrassing — **we found out
exactly why, and it's a trap other researchers are probably falling into.** The
training does make the network "smoother" (its jumps shrink ~4×). But event rate
depends on the threshold *measured in units of the jump size* — so when you shrink
the jumps, you also shrink the natural threshold, and the two cancel. The training
moves the network *along* its existing efficiency curve, not *above* it.

The kicker: if you compare the trained network to the original **at the same fixed
threshold** — which is the natural, default comparison — it looks like a **36% energy
win.** That number is a mirage. Re-tune the threshold to each network fairly (which
is what real deployment does) and the win vanishes. **Prior "training-for-efficiency"
results in this field use exactly that fixed-threshold comparison.** So our negative
isn't just "our idea didn't work" — it's a measurement-methodology warning to the
whole subfield, with the mechanism worked out.

## What we accomplished, and why it matters

**Two new tools the field didn't have:**
1. **An energy model.** For the first time you can *predict* an event-driven
   network's energy from activation statistics, analytically, before building it.
   For hardware designers this means power and bandwidth envelopes at *design time* —
   before silicon, before even a simulation.
2. **A tuning shortcut.** Thresholds by one calculation instead of a sweep.

**One honest negative that saves other people time and warns them of a trap:**
3. Training-for-temporal-sparsity, in our tests, doesn't beat the simple post-hoc
   baseline — and the "wins" reported elsewhere may be an artifact of comparing at a
   frozen threshold. Independently, the newest work in this space (on large language
   models) has already gone *training-free* — the field is converging on the baseline
   our results endorse.

**Why AI should care.** Event-driven / neuromorphic computing is one of the few paths
to genuinely low-power AI at the edge — the "always listening" and "always watching"
devices that can't afford a GPU. Giving that field a predictive energy model, and a
way to allocate its energy budget by math instead of search, makes it easier to
design and deploy. And the negative result is a small parable in how to *evaluate*
efficiency claims without fooling yourself.

---

# Part 2 — Whiteboard mastery (present it cold at Anthropic)

## 2.0 The 90-second arc (memorize this cold)

> "Event-driven neural nets save power by only computing when a neuron's output
> changes past a threshold. That threshold-crossing is *literally* a level crossing
> of the activation over time — so the energy is governed by 1944 Bell-Labs math,
> Rice's formula, which is the engine of my whole research program. I asked three
> questions. **One: can I predict a delta network's energy from activation
> statistics, without running it?** Yes — to two to three percent on three real
> networks. **Two: can I invert that to allocate thresholds analytically instead of
> sweeping?** Yes — I match a 48-configuration search at two percent of its tuning
> cost. **Three: can I train the network to be *more* efficient with a differentiable
> crossing budget?** No — and this is the interesting part. It ties post-hoc
> thresholding, because event rate is scale-free: it depends on threshold-over-jump-
> size, so smoothing the network rescales its dynamics without restructuring them. A
> fixed-threshold comparison would show a bogus 36% win — which is exactly how prior
> training-for-sparsity work reports its gains. Two positive contributions, one
> negative that's a measurement warning to its own subfield, eight seeds, every
> number regenerable from raw logs by one script."

That's it. Everything below is depth for the follow-ups.

## 2.1 Whiteboard choreography — draw in this order

**Board 1: the identity (the whole paper in one drawing).**
Draw a wiggly activation trace `a(t)` over time. Draw a horizontal dashed line at
height `θ` above a "last transmitted value" dot. Circle each place the curve pushes
past the line.

Say: *"Delta network fires an event exactly when the activation moves θ from where it
last reported. That's a level crossing. Counting energy = counting crossings."*
Then write Rice's formula:

```
μ(u) = (1/π)·√(λ₂/λ₀)·exp(−u²/2λ₀)        λ₀ = variance,  λ₂ = variance of the slope
```

Say: *"√(λ₂/λ₀) is the RMS frequency — how fast it wiggles. Two statistics of the
trace give me the crossing rate. No Fourier transform, and autograd gives me the
slope for free."*

**Board 2: the estimator (why it's differentiable).**
Write the smoothed, exact-for-sampled-data version:

```
ĉ(u) = (1/T) Σ_t | Φ((a_{t+1}−u)/ε) − Φ((a_t−u)/ε) |          Φ = Gaussian CDF
```

Say: *"For a sampled trace this is exact as ε→0 — it counts sign changes — but it's
smooth, so it's differentiable in the network weights. That's what lets me put it
inside training. Same estimator family as papers 1–3 of the program."*

**Board 3: the three bets.**
Draw three columns: **PREDICT → ALLOCATE → TRAIN**. Fill checkmarks as you go: ✅ ✅ ❌.

- PREDICT: *"Calibration profile predicts held-out crossing rates to 2–3%. Rice's
  Gaussian version works on bell-curve neurons, breaks on heavy-tailed ones — I
  correlate its error with kurtosis, Spearman 0.4 to 0.8. The naive i.i.d. baseline
  is 5–17× off, so temporal structure is the whole game."*
- ALLOCATE: *"Invert the rate curve for a budget → thresholds in one shot. Matches a
  48-config search at ~2% tuning cost, hits budget within 4%."*
- TRAIN: *"Differentiable budget penalty, provably silent under budget. 8 seeds.
  Ties post-hoc thresholding. So do L1, activity penalty, plain fine-tune."*

**Board 4: the punchline drawing (the mechanism).**
Draw two identical-shaped bumpy curves, one labeled "base," one "budget-trained"
drawn at half the vertical scale. Under them write:

```
event rate  ≈  γ(θ/σ_δ) · TV / (θ·Δt)          TV ∝ σ_δ    ⟹   θ and σ_δ cancel
```

Say: *"Training shrinks the jump scale σ_δ by ~4×. But event rate depends on θ over
σ_δ. Shrink the denominator, the natural threshold shrinks too — the network moves
ALONG its own efficiency curve, not above it. Scale-freeness. That's the whole
negative in one line."*

Then draw the trap: two vertical dashed thresholds at the *same absolute height*,
one cutting the tall curve, one cutting the short curve much lower.

Say: *"Compare at the SAME fixed threshold and the smoothed network looks 36% more
efficient. That's a mirage — you re-tune per model in deployment and it's gone. Prior
training-for-sparsity papers report the mirage."*

## 2.2 The numbers to memorize (the only table you need)

| Claim | Number |
|---|---|
| **GATE V** — Rice vs exact counts (known spectra) | **≤ 1.9%** |
| GATE V — discrete-Gaussian on sampled OU / multisine | ≤ 1.0% / ≤ 0.18% |
| **GATE P** — calibration predictor, median error (SC2 / psMNIST / text) | **1.8–3.1%** (worst stream 8.1%) |
| Rice (Gaussian) — near-Gaussian → heavy-tailed streams | 8% → 75% error; kurtosis-correlated |
| i.i.d. baseline (no temporal structure) | **5.6–17.3× off** |
| Event-rate prediction (open-loop sim), useful regime | ≤ 3.0% (SC2), ≤ 4.2% (psMNIST) |
| **ALLOCATE** — analytic vs 48-config search | **matches front at ~2% tuning cost** (48× cheaper) |
| Allocation hits requested budget | within 4% |
| **TRAIN (negative)** — budget vs post-hoc, SC2, 8 seeds | **90.4% vs 90.5%**, paired diff −0.1%, Wilcoxon p = 0.31 |
| Mechanism — jump-scale shrink under budget | σ_δ ↓ **3.9×** |
| **The trap** — fixed-threshold apparent saving / true cost | **36% "win"** / vanishes on re-tune (2.5% acc cost) |
| Modeled energy (SC2, 20%-events operating point) | **5.2×** less (30 vs 158 µJ/decision), Horowitz-modeled |
| Training overhead (estimator, tiny GRU) | 23 ms/layer, 12× on a ~4 ms step; fixed cost, MC-reducible |
| Base models | SC2 91.7%, psMNIST 98.7%, text 2.23 bpc |

## 2.3 The objections they'll throw — and your answers

**"Isn't this just delta networks, which already exist?"**
> "Delta networks are the *mechanism* I measure. What didn't exist is a *predictive*
> model of their energy. Every prior paper tunes thresholds by search and reports
> events after measuring. I predict events from statistics before running, and
> allocate thresholds by inverting that prediction. The mechanism is theirs; the
> analytic handle is new."

**"Prior work already trains for temporal sparsity and reports gains — you're wrong."**
> "That's the objection my negative is aimed at. Those gains are read off at a *fixed*
> inference threshold. My scale-freeness result says a fixed-threshold comparison
> credits any smoothing with an apparent saving — I measure 36% — that disappears when
> you re-tune the threshold per model, which is what deployment does. I'm not saying
> their measurement is fake; I'm saying it's the wrong comparison, and I show why with
> the mechanism. The newest LLM work in this space went training-free, which is the
> baseline I endorse."

**"A negative result isn't a contribution."**
> "The paper has two positives — the energy model and the allocation shortcut. The
> negative is a *third* contribution because it comes with a mechanism (scale-freeness)
> and a reusable warning (measure at re-tuned thresholds). It's pre-registered — I
> declared the kill condition before running — so it's a real test, not a rationalized
> failure. In this program, the negatives are load-bearing."

**"Rice's formula assumes Gaussian, stationary signals. Real activations are neither."**
> "Correct, and I don't hide behind it. The Gaussian formula is one of three
> predictors, and I show exactly where it breaks — error rises with kurtosis, Spearman
> 0.4–0.8. The *load-bearing* predictor is the empirical crossing profile, which
> assumes no Gaussianity at all and is the same object I put in the training loss. Rice
> is the interpretable special case; the empirical estimator is the workhorse."

**"You only tested tiny models."**
> "Deliberately, on one RTX 3080, because the claims are about *rates and statistics*,
> which are scale-free in model size — the estimator is O(N·L) over trace points, not
> parameters. I state the scale limitation explicitly and don't claim past it. The
> prediction and allocation machinery is exactly what transfers; the training negative
> I claim only at tested scale."

**"Why should Anthropic care about edge/neuromorphic efficiency?"**
> "Two reasons. Narrowly: event-driven compute is a real path to low-power always-on
> AI, and I gave it a design-time energy model. Broadly — and this is the part I'd
> actually pitch here — it's a clean case study in *not being fooled by your own
> metric.* The fixed-threshold comparison looks like a 36% win and isn't. Catching
> that is an evaluation-integrity problem, and evaluation integrity is the thing that
> matters most when you're measuring capabilities you care about being honest about."

## 2.4 The one sentence to leave them with

> **"The century-old math of line-crossings turned out to be the missing energy model
> for event-driven AI — I could predict the energy and allocate it by calculation
> instead of search; and when training-for-efficiency didn't beat the simple baseline,
> the reason was a scale-freeness that also explains why the field's usual
> fixed-threshold comparison overstates its wins. Two tools and a warning, every
> number regenerable from raw logs."**

Its predecessors in this program each carried one deep export — Paper 3's was
"optimizers hunt null spaces." This one's is: **"an efficiency metric measured at a
frozen operating point rewards moving the operating point, not the frontier."** If
they remember one idea, make it that one — it generalizes far past delta networks.
