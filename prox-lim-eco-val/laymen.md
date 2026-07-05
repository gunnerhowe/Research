# Event-Timing Priors, In Plain English

*(Section 1 is for anyone. Section 2 is the whiteboard-interview deep dive.)*

---

## Section 1: The Simple Version

### The problem, in one sentence

AI weather-and-climate simulators are **graded** on how well they handle extreme
events (heat waves, storms, floods), but nobody ever **teaches** them *when*
extreme events are supposed to happen.

### Why that matters

Imagine a flight simulator that gets the average flight right but gets
turbulence wrong — not how *strong* turbulence is, but its *rhythm*: how often
it hits, whether it comes in bursts, how long the calm stretches last. For the
people who actually use these simulators, the rhythm is often the whole point:

- **Return periods** ("a 100-year flood") are what building codes and insurance
  pricing are built on. If your simulator says 60 years when nature says 100,
  real decisions get made wrong.
- **Clustering** matters because two heat waves back-to-back is much worse than
  two heat waves a decade apart — same count, totally different damage.

Modern AI emulators (Google's GenCast, NeuralGCM, AI2's ACE2) are spectacular
at ordinary forecasting, and every one of their papers *evaluates* extremes.
But their training objectives only say "predict the next state well" or, in the
fanciest cases, "get the overall statistics right." None of those objectives
can even *see* the rhythm of events — you could shuffle time like a deck of
cards and those objectives wouldn't notice.

### The key real-world observation

Here's the asymmetry we exploit: even when the raw weather data is noisy and
messy, humanity keeps **beautifully curated lists of when extreme events
happened** — hurricane track archives, earthquake catalogs, flood records.
These catalogs are tiny (just timestamps!) but very clean, and they're
maintained separately from the noisy gridded data that models train on.

So: can a tiny, clean list of *when things happened* be used to fix a huge
simulator trained on noisy data?

### What we built

1. **A tiny "rhythm expert."** We train a very small neural network (a
   *temporal point process*, or TPP — a standard tool for modeling event
   timings, used for earthquakes and hospital visits) on nothing but the
   catalog: the timestamps of past extreme events. It learns the rhythm —
   "after an event, expect quiet for a while, then rising chance" — whatever
   the true pattern is. It has ~40,000 parameters. The simulator has millions
   of data points. The expert sees only a few thousand numbers.

2. **Use the expert as a coach.** Periodically during training, we let the
   simulator free-run for a while, automatically mark where its extreme events
   happen, and ask the expert: "does this rhythm look right?" The answer flows
   back into the simulator as a learning signal (this requires some math to
   make "did an event happen" differentiable — we use a soft version of the
   event detector).

3. **The trap, and the fix (our favorite part).** The obvious approach —
   "make your events more likely under the expert" — fails
   catastrophically, and it fails in an interesting way. For rare events, the
   single most likely story is *"nothing happened."* So the simulator learns
   to just... stop producing extreme events. In our test, it suppressed
   **99.75%** of them. It gamed the coach.
   The fix: train a *second* tiny expert that continuously tracks the
   simulator's own current rhythm, and only reward the **difference** —
   "how much more does the true expert like your rhythm than an expert fit to
   *you* likes it?" That difference is zero exactly when the simulator's
   rhythm matches reality, so there is nothing left to game. (Readers from
   the generative-modeling world will recognize this as the point-process
   version of a trick used to distill image diffusion models.)

### What we accomplished

On a standard chaotic physics testbed (a toy atmosphere called Lorenz-96),
trained under realistic conditions (coarse time steps, noisy observations),
across 8 repeated experiments:

- The simulator's event **rate** was wrong by ~11%; after coaching it's wrong
  by ~0.1%.
- The **spacing** between events (including its subtle "quasi-periodic" beat)
  becomes statistically indistinguishable from reality by our strongest test.
- **Return periods** — the "100-year event" numbers — get ~2× more accurate.
- Ordinary forecast skill is essentially untouched (~0.5% cost — about 170×
  smaller than the penalty you'd pay for using a non-generative model).

And the controls prove *why* it works:

- Give the coach the right event *amount* but a scrambled rhythm, and the
  simulator gets **worse**, not better. So it really is the rhythm being
  transferred, not just "pay attention to extremes."
- The best existing method (matching overall statistics) fixes the averages
  but **not** return periods or burstiness — exactly the gap our theory
  predicted.

We also found — and honestly report — where the method **breaks**: on a second
system whose internal dynamics were too corrupted, the coach can't teach the
rhythm (the simulator physically can't produce it), so the simulator hides
events instead. The failure has a clear signature (event rate crashing during
training) that you can monitor for free, and we analyze the mechanism.

### Why this matters beyond weather

1. **A new, general recipe:** "Fit a tiny model to the *timing statistics* of
   rare events; use its likelihood-ratio as a training signal for a big
   generative model." Nothing in the recipe is weather-specific. Anywhere you
   have (a) a generative simulator and (b) an event catalog — seismology,
   epidemiology, power-grid failures, financial shocks — this applies.

2. **A cautionary tale about optimizing likelihoods of rare events** that
   generalizes way past physics: any time you fine-tune a generative model
   against a learned judge, "make the judge happy" can quietly become "avoid
   doing anything the judge scores." Our collapse-and-fix is a crisp, fully
   measurable case study of that failure mode — and of the fix (judge the
   *difference* against a tracking baseline, not the raw score).

3. **Honest boundary-mapping.** We didn't stop at the success story; we mapped
   where the idea stops working and why. That map (and the cheap warning
   signal) is arguably as useful to practitioners as the method itself.

---

## Section 2: Whiteboard It Cold — The Interview Deep Dive

*Everything below is what you need in your head to present this with a marker
in your hand. Structure: the 90-second pitch → the three drawings → the math →
the numbers → the story arc → hard Q&A → the 5-minute script.*

### 2.1 The 90-second elevator pitch (memorize this)

> "Generative emulators of chaotic systems — think GenCast-style weather
> models — are evaluated on extremes but never trained on their *timing*. All
> existing training signals are invariant to permuting time: they can't
> distinguish quasi-periodic extremes from bursty ones. But event *timing* is
> what return periods and compound risk are made of, and clean event catalogs
> exist even where state data is noisy.
>
> So I fit a small neural temporal point process to catalog event times and
> used its likelihood on soft, differentiable events extracted from surrogate
> rollouts as a fine-tuning signal. The naive version reward-hacks: for rare
> events the empty sequence is near the density mode, so the surrogate
> suppresses 99.75% of events. I fixed it with a likelihood-*ratio* against a
> co-trained self-TPP — the point-process analog of variational score
> distillation — whose gradient vanishes exactly at distribution match.
>
> Across 8 paired seeds on Lorenz-96: event rate restored from +11% error to
> +0.1%, return-period error halved, and the surrogate's event stream becomes
> statistically indistinguishable from truth under the catalog model — at a
> 0.5% cost in CRPS. A rate-matched Poisson control with the identical loss
> *damages* the model, proving the temporal structure is what transfers. And
> on a second system I mapped the boundary: when rollout dynamics are too
> corrupted, timing-only priors collapse rather than repair — with a
> mechanism analysis and a free training-time canary."

### 2.2 The three whiteboard drawings

**Drawing 1 — the pipeline (draw left to right):**

```
 chaotic system (2-scale L96)          CATALOG (clean event times only)
        │  + observation noise                     │
        ▼                                          ▼
 noisy trajectories ──► generative surrogate   tiny TPP  p_GT   (frozen)
                        (flow matching,            ▲
                        next-state, CNN)           │ fit by MLE
                              │                    │
              short rollout (100 steps)      self-TPP p_self ◄─ refit online on
                              │                    │            surrogate's own
                    soft event extraction          │            (hard) events
                    w_t = σ(·)σ(·)  ∈ [0,1]        │
                              ▼                    ▼
               L_aux = −E[ log p_GT(w) − log p_self(w) ]
                              │
              gradient back through rollout (truncated windows)
                              ▼
                    θ ← θ − η ∇(L_FM + λ·L_aux)
```

Say while drawing: "Two data streams — noisy states train the base model with
flow matching; the clean catalog only ever touches the TPP. Fine-tuning adds
the ratio term; the FM loss stays as an anchor on ordinary skill."

**Drawing 2 — the money figure (hazard curve, draw freehand):**

Axes: x = "time since last event", y = "event hazard (instantaneous risk)".
Draw four curves:
- **Truth**: oscillating (peaks at the recurrence period and its harmonics) —
  the system's waves make extremes come back rhythmically.
- **Base model**: same oscillation but over-amplified (noise → too many
  crossings).
- **TPP-aux**: sits on top of truth. ✓
- **Poisson-aux**: a flat line. It literally erased the rhythm — the
  content-specificity dissociation in one picture.

**Drawing 3 — the 2×2 that summarizes both systems:**

```
                     │ dynamics mostly OK      │ dynamics deeply broken
                     │ (L96: waves right,      │ (KS: spectrum off by e³,
                     │  amplitudes noisy)      │  marginals fine!)
─────────────────────┼─────────────────────────┼─────────────────────────
 structured TPP prior│ FIXES everything        │ COLLAPSES rate (0.31×)
                     │ (rate 1.00, RP 2×)      │ ← the boundary
─────────────────────┼─────────────────────────┼─────────────────────────
 Poisson (rate-only) │ HURTS (rate was already │ HELPS on return periods
 prior               │ ~right; forces wrong    │ (rate WAS what's broken;
                     │ structure)              │ flat hazard = permissive)
```

Punchline: "A prior helps exactly to the extent its content matches what's
broken. Structured priors whose structure the dynamics can't realize do
damage instead."

### 2.3 The math you must be able to write

**TPP likelihood (grid form).** GRU reads the event stream, emits per-step
intensity λ_t (events/time, from history *before* t):

```
L(w; λ) = Σ_t [ w_t · log λ_t  −  λ_t · Δt ]
```

Two properties to say out loud:
1. It's the discretization of the continuous point-process log-likelihood
   (Σ log λ(t_i) − ∫λ dt), exact as Δt→0.
2. It's **linear in w_t** — so soft (relaxed) event indicators drop in
   natively. This is what makes the whole thing end-to-end differentiable.

**Soft events.** s = observable (e.g., −x for deep troughs), u = threshold:

```
w_t = σ((s_t − u)/τ) · σ((u − s_{t−1})/τ)     (an upcrossing, softened)
```

Calibrate τ so soft rate ≈ hard rate (ours: +4.9% bias, 80% of soft mass on
true event steps).

**Why naive likelihood collapses (be able to derive this).** We'd like the
surrogate's event law P to match the catalog law P_GT. Naive objective:
maximize E_{w~P}[log p_GT(w)]. But this functional is maximized by putting all
mass on the *mode* of p_GT — and for rare events the (near-)empty sequence has
enormous density (density ≠ typicality). So gradient descent discovers "have
no events." Empirically: rate ratio **0.0025** (99.75% suppressed) while the
"likelihood" got *better* (−0.055 vs. the truth's own −0.0994 — rollouts more
likely than reality!).

**The ratio fix.** Maintain self-TPP p_ψ ≈ MLE fit to current P; optimize

```
L_aux = − E_P[ log p_GT(w) − log p_ψ(w) ]      (both TPPs frozen in this step)
```

If p_ψ tracks P well, E_P[log p_GT − log p_ψ] ≈ −KL(P ‖ P_GT): non-positive,
zero **iff** P = P_GT. The self-term cancels the uniform "fewer events are
likelier" pressure. Fixed point = distribution match, not mode.
Name-drop: this is **variational score distillation / Diff-Instruct**
transplanted to point processes. And the RLHF framing: reward = judge score,
collapse = reward hacking, self-TPP = the KL-to-current-policy penalty.

**Two-timescale condition (the KS lesson).** The brake only works if p_ψ
tracks P *faster than P moves*. On clustered priors (very low baseline
intensity between bursts), each spurious event is punished so hard that P
sprints toward sparsity, outruns the self-TPP, and lands in a collapsed local
equilibrium where p_ψ ≈ P so the gradient ≈ 0. We showed this from the
training traces (llg/lls curves) and that no hyperparameter setting we tried
(λ ∈ {1,3,10}, self-steps ∈ {5,25}, 3× longer, gentler lr, milder noise)
escapes it when the dynamics can't realize the target structure.

**Gradient path through chaos.** Full BPTT through a chaotic rollout explodes
like e^{λT} (λ = Lyapunov exponent). We truncate: gradients flow at most W=25
steps. Trick that made it fast: run the rollout gradient-free recording
states+noises, then **re-forward all 25-step windows in parallel as one
batch** — bitwise identical states (deterministic sampler given noise), same
gradient semantics as truncated BPTT, ~8× less sequential depth. Plus CUDA
Graphs on the no-grad rollout (kernel-launch-bound on Windows: ~5 ms per tiny
CNN eval eager!). Net: 30.7 s/iter → 1.57 s/iter.

### 2.4 Numbers to memorize (L96 = 8 paired seeds; KS = 3)

| Thing | Number |
|---|---|
| Base event-rate error | **+10.7%** → TPP-aux: **+0.1%** (p=8×10⁻⁵) |
| IET distribution KS stat | 0.068 → **0.026** (p=3×10⁻⁵) |
| Return-period log-error | 0.113 → **0.052** (p=0.002) |
| Catalog-TPP LL of rollouts | −0.1020 → **−0.0988** vs truth's own **−0.0994** ("at the ceiling") |
| CRPS cost | **+0.5%** (~170× smaller than the deterministic model's gap) |
| Poisson control | rate → 0.82, IET W1 ×2.2 worse — **actively harmful** |
| Naive-likelihood ablation | rate ratio **0.0025** — total collapse |
| Shuffled-interval control | ≈ full TPP (events are near-renewal → IET *shape* is the payload) |
| Jiang control (invariant stats) | best marginals; return periods **unimproved** (p=0.73) |
| KS: base → TPP-aux rate | 2.50× → **0.31×** (collapse; the boundary) |
| KS: Poisson control | best return periods (0.16) — the mirror image |
| Divergences, all conditions | **0** |
| Model sizes | surrogate ~0.66M params; TPP ~40k; catalog = timestamps only |
| Compute | one RTX 3080; base 9 min; fine-tune 20 min; full grid ~8 h |

Statistical design: two-phase training — one base checkpoint per seed, every
condition fine-tunes *that same checkpoint* with an identical budget → paired
tests. Seed 0 was the dev seed (all tuning), seeds 1–8 confirmatory, never
touched for selection. Wilcoxon floor at n=8 is 0.0078 — hit on every
headline metric.

### 2.5 The story arc (this is interview gold — the pivots)

Tell it as "the plan survived contact with reality five times":

1. **The planned event definition was empirically Poisson.** Spec said "site
   energy above 98th percentile." Measured it: Fano ≈ 1, IET CV ≈ 1.0, zero
   serial correlation — nothing for a TPP to learn, and the controls couldn't
   dissociate. *Scanned observables × thresholds first*, found deep troughs
   are strongly quasi-periodic (CV 0.72) and aggregate events are clustered
   (CV 1.15). Lesson: validate that your signal exists before building the
   machine to exploit it.

2. **The baseline was too good — four times in a row.** Full data, 2
   trajectories, coarse steps, 2-step sampler: all near-perfect event timing.
   Root cause: circular-CNN translation equivariance × 40 sites = enormous
   effective sample size; the flow map is easy. Lesson: architecture-matched
   testbeds don't produce realistic deficits; you must *design* the deficit
   regime (we used observation noise, which is both realistic and the exact
   setting of the closest prior work).

3. **Caught mode collapse on the whiteboard before wasting GPU.** Thinking
   through the gradient of the naive likelihood revealed the empty-sequence
   mode issue *a priori*; designed the ratio objective before the first run,
   then kept the naive version as an ablation — which validated the theory
   spectacularly (0.0025!).

4. **The dissociation moved.** In the noise regime, the invariant-measure
   control fixed more timing than expected (noise deficits are partly
   marginal-reducible). Rather than shop for a friendlier regime, we
   *repositioned the central claim* onto the Poisson control (identical loss,
   rate info; only structure differs) — which is the cleaner test of "timing
   structure as a training target" anyway, and it landed with p≈10⁻⁵.

5. **KS failed — and we shipped the failure.** Every timing-only variant
   collapsed on KS. Diagnosed the mechanism from logged likelihood components,
   established it wasn't estimator choice or tuning, showed what *does* work
   there (increment statistics), showed naive composition doesn't simply add,
   and wrote it as the paper's boundary + future work. A method paper you can
   trust tells you where it breaks.

Also worth mentioning: the negative transfer result (site-level prior does
not fix spatial-max events — the prior repairs the process it was fit to,
it's a targeted prior not a generic regularizer), and the bibliography was
adversarially verified against arXiv before shipping.

### 2.6 Hard questions you will get, with answers

**"Why not fix extremes at inference time — guidance, importance sampling?"**
That line exists (Manshausen et al. 2026 — likelihoods of extremes from a
trained emulator). It changes *samples you draw on purpose*; it doesn't change
the model's long-rollout climatology, which is what return periods are
computed from and what every downstream consumer inherits. Training bakes the
correction in once, for all uses.

**"Isn't this just distribution matching? Why a TPP and not a rate/count
loss?"** The Poisson control *is* the rate-only version — identical loss
machinery, correct rate, no structure — and it makes the surrogate *worse*.
The information that repairs the model is the conditional temporal law
(hazard shape / IET distribution), which a TPP is the minimal model of.

**"Why does the naive likelihood collapse when MLE doesn't?"** Direction of
optimization. MLE optimizes the *model* against fixed data — fine. Here the
*data distribution* (surrogate rollouts) is optimized against a fixed model,
and E_P[log q] is maximized at q's mode, not at P=q. You need the entropy-like
self-term; that's exactly why VSD/Diff-Instruct exist for image diffusion and
why RLHF uses a KL penalty to the reference policy.

**"How is this different from Jiang et al. / DySLIM / spectral losses?"**
Those objectives are functionals of the time-marginal or time-averaged law —
invariant under permuting time. Two processes with identical invariant
measures can have opposite event rhythms (quasi-periodic vs. bursty). We
constrain the conditional intensity of the event process itself; empirically,
the invariant-measure control (given equal clean-reference privilege) leaves
return periods and burstiness unrepaired while ours lands on the ceiling.

**"Why flow matching? Why not a 'real' diffusion model?"** Same family
(continuous-time generative; rectified-flow instantiation). Chosen because the
few-step deterministic sampler makes rollouts differentiable via
reparameterized noise — which the aux loss needs. Nothing in the method cares
which family; DDPO-style score-function gradients are the fallback if your
sampler isn't differentiable (we didn't need them, and we verified the KS
failure isn't an estimator artifact).

**"Would this scale to a real weather model?"** The recipe scales in
principle: catalogs exist (IBTrACS, earthquake catalogs, flood records);
events per grid cell/region are exchangeable streams like our 40 sites; the
TPP stays tiny; fine-tuning cost is trivial next to pretraining. Open
problems, honestly: event definitions in high-dimensional fields (tracked
objects vs. threshold crossings), catalog noise/censoring entering the prior,
and the KS lesson — you need the base model's dynamics to be sound enough to
*realize* the target rhythm, which for a GenCast-class model is plausibly true
(their dynamics are excellent; their timing is unexamined).

**"What's the Anthropic-relevant takeaway?"** This is a controlled,
fully-measurable micro-study of aligning a generative model to a learned
objective: reward hacking appears (mode collapse), the fix is a
tracking-baseline / KL-style regularizer with a provable fixed point, there's
a two-timescale stability condition (the judge must adapt faster than the
policy moves), failure has a cheap online canary (event rate), and the
evaluation is paired, pre-registered-style (dev seed vs. confirmatory seeds),
with adversarial controls. Swap "surrogate" for "policy" and "TPP" for
"reward model" and every lesson transfers.

**"Why should I believe the stats?"** Paired design (same base checkpoint per
seed, matched budgets), tuning quarantined to a dev seed, 8 confirmatory
seeds, both parametric and nonparametric tests (Wilcoxon at its floor on all
headline metrics), zero divergent rollouts, negative results reported
(transfer, KS, CRPS cost at p≈0.05 stated plainly).

### 2.7 The 5-minute whiteboard script

- **0:00–0:45** — Problem + asymmetry. Draw a squiggly time series, mark
  events, say "graded on these, never taught their timing; catalogs of
  timestamps exist and are clean even when states are noisy."
- **0:45–2:00** — Drawing 1 (pipeline). Write the TPP likelihood; stress
  linearity in w → soft events → differentiable end-to-end.
- **2:00–3:00** — The trap. Write E_P[log p_GT], say "maximized at the mode;
  for rare events the mode is silence; 99.75% suppression." Write the ratio;
  say "gradient vanishes iff distributions match — VSD for point processes;
  RLHF's KL penalty is the same medicine for the same disease."
- **3:00–4:00** — Drawing 2 (hazard curves). Read the result off the picture:
  base over-oscillates, ours lands on truth, Poisson control flattens —
  structure, not rate. Quote: rate +11%→+0.1%, RP error halved, CRPS +0.5%,
  n=8 paired, everything at the Wilcoxon floor.
- **4:00–5:00** — Drawing 3 (the 2×2). "Second system inverts it — that's the
  boundary, we shipped it with the mechanism and a free canary. A prior helps
  exactly insofar as its content matches what's broken. Open problem:
  dynamics-aware timing priors."

**Closing line:** *"The rhythm of rare events is a training signal nobody was
using. We showed how to use it, proved it's the rhythm itself that transfers,
showed exactly how the naive version reward-hacks and how to fix it, and
mapped where the whole idea stops working. That's the full life cycle of a
method: motivation, mechanism, controls, and boundary."*
