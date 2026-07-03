# What this project did, in plain English

## The weird thing we studied: "grokking"

When you train a small AI on a simple math task (adding numbers with a wraparound, like
clock arithmetic), something strange happens. The AI memorizes all its practice problems
almost immediately — it gets 100% on anything it has seen before. But show it a problem it
*hasn't* seen, and it fails completely. It's like a student who memorized the answer key
without understanding anything.

Then, if you just... keep training it on the same problems, over and over, for a absurdly
long time — nothing changing, scores flat — it suddenly "gets it." Almost overnight, it goes
from failing every new problem to acing all of them. It stops memorizing and genuinely
understands the math. Researchers call this delayed lightbulb moment **grokking**, and the
mysterious dead time before it is the **grokking delay**.

Why should anyone care about a toy math task? Because this is one of the only places where
we can watch, in complete detail, an AI transition from *memorizing* to *understanding*.
Big AI models like ChatGPT almost certainly undergo transitions like this too, but they're
far too large to watch closely. This little task is the lab mouse of AI understanding.

## The question we asked

Everyone agrees the delay can be shortened — people have found tricks that speed it up.
But all the existing tricks are generic: they fiddle with the training process (like
turning knobs on the engine) without telling the AI anything about *what the right answer
looks like*.

So the open question was: **is the delay actually just the time the AI needs to organize
its internal "mental map" the right way?** If so, then *directly helping it organize that
map* should shrink the delay — and helping it organize the map the *wrong* way should not.
Nobody had actually run that experiment. We did.

## What we actually did

Think of the AI's internal representation as a big pinboard where it places every math
problem. We added a gentle "nudge" during training that says: *problems with the same
answer should be pinned close together.*

The clever part is the comparison. We trained many copies of the same AI, identical in
every way except for one thing:

1. **The real hint**: "group problems that share the same answer" — the true structure of
   the task.
2. **The fake hint**: "group problems according to this random list" — a hint with the exact
   same strength and format, but pointing at meaningless groupings.
3. **No hint at all** (the normal case, for reference).
4. **A trick-check**: our hint had a side effect on an internal setting of the network (its
   "weight size") that's known to affect grokking speed on its own. So we ran extra copies
   where we reproduced *only that side effect*, with no hint — to make sure any improvement
   wasn't just the side effect in disguise.
5. **The best existing speed-up trick** (called Grokfast), for comparison.

Then we ran 80 training runs on a single gaming graphics card and timed everything.

## What happened

- **The real hint worked.** In most runs it made the AI grok — sometimes dramatically
  faster than normal. Our best case understood the task almost **3x sooner** than it
  would have on its own.
- **The fake hint never worked. Not once, in 20 tries.** Same nudge, same strength, wrong
  content — and the AI never reached understanding within our time limit. The odds of that
  split happening by chance are about one in ten million.
- **The trick-check also never worked** (0 for 15). So the real hint's success wasn't the
  side effect in disguise. In fact, the side effect alone was *harmful* — which makes it
  more impressive that the right hint overcame it.
- **Honest fine print:** the real hint isn't a free lunch. Its benefit depended on the
  dose, and in a minority of runs it backfired and *slowed things down*. It's a powerful
  lever, not a magic button.

We also watched the AI's "pinboard" throughout training, and the timing lined up
perfectly in all 80 runs: understanding never arrived before the internal map got
organized, and whenever the map finished organizing, understanding followed. The delay
*is* the map-making time.

## Why this matters

**We turned a "probably" into a "demonstrated."** Researchers suspected the grokking delay
was really about internal organization time. We showed it causally: give the AI the right
organizational blueprint and understanding can arrive much sooner; give it a wrong
blueprint of identical strength and understanding never arrives. It's the *content* of the
guidance that matters, not the push itself.

**What this could mean for AI more broadly** (with the usual caution that we studied one
small task, and bigger systems may differ):

- **Training efficiency.** Big models cost millions of dollars in compute, partly because
  we train them far past the point of memorization and wait for deeper generalization to
  emerge. If that waiting is fundamentally organization time, then guiding internal
  organization directly — rather than hoping it emerges — could make training meaningfully
  cheaper and faster.
- **Steering what models learn.** Our fake-hint result cuts both ways: pushing a model
  toward the wrong internal structure reliably *blocked* understanding. That means internal
  structure is a steering wheel, not just a speedometer — something you could in principle
  use to encourage the kinds of understanding you want.
- **Reading a model's mind to predict its behavior.** The internal measurements predicted
  the moment of understanding in every single run. That's encouraging for AI safety
  research, which wants to detect what a model has *actually* learned (understanding vs.
  memorization) before that difference shows up in its behavior.

## The one-sentence version

> An AI's long "confused phase" between memorizing and understanding is really the time it
> takes to organize its internal map — and we proved it by handing different AIs a correct
> map, a fake map, and no map, and watching only the correct-map ones understand sooner.

---
---

# Part 2: The whiteboard mastery guide

Everything below is what you need to present this cold — the pitch scripts, the exact
diagrams to draw, every number, the causal logic, the mechanism, and the questions you'll
get asked with answers. Practice until you can do the 5-minute version without notes.

## 2.1 The pitch at three lengths

**One-liner (10 seconds):**
"I ran the first causal test of whether the grokking delay is representation-formation
time — injected the task's true structure into hidden representations via a contrastive
auxiliary loss, against a shuffled-structure control and a weight-norm-matched control,
and showed the structural *content* alone determines whether and when generalization
arrives."

**90-second version:**
"Grokking is delayed generalization: a small transformer memorizes modular addition in ~200
epochs, then sits at chance test accuracy for ~20,000 epochs before abruptly generalizing.
The 2026 literature can accelerate this, but every known lever is structure-agnostic —
gradient filtering, weight-norm clamping, geometric penalties. Meanwhile observational work
says the delay is really the time for task-structured representations to form. Nobody had
intervened on that directly. I added a SupCon auxiliary loss on the hidden representations
where positives share (a+b) mod p — the task's true equivalence structure — and compared
against a shuffled-structure control that is identical in every respect except the content
of the partition: same loss, same strength, same class sizes, same geometry. True structure:
22 of 30 runs grok, up to 2.75x faster than baseline. Shuffled: 0 of 20, Fisher exact
p ≈ 1e-7. And because the aux loss inflates the weight norm — which by the June 2026 delay
law should itself change grokking time — I replayed each intervention run's exact norm
trajectory onto plain cross-entropy: 0 of 15 generalize; they collapse into logit
saturation. So it's not the loss, not the norm, not the optimizer: it's the structure.
Representation probes confirm the timing story in all 80 runs — no run generalizes before
its class-clustering and Fourier metrics move, and every run whose metrics complete the
rise generalizes."

**5-minute version:** the 90-second version + draw Diagrams 1–3 below + the numbers table
+ one mechanism sentence: "the aux loss has three effects — norm inflation (bad), logit
anti-saturation (protective), and structure seeding (the causal one); grokking speed is a
race between structure seeding and norm-driven saturation, which also explains the
bimodality across seeds."

## 2.2 What to draw (four diagrams, in order)

**Diagram 1 — the phenomenon.** Axes: accuracy vs log-epochs. Two curves: train accuracy
shoots to 100% at ~200 epochs; test accuracy flat at ~1% until ~20k, then a sharp S-curve
to 100%. Label the gap "the grokking delay (~19,700 epochs)". This takes 15 seconds and
anchors everything.

**Diagram 2 — the intervention.** Draw the model as a pipeline:

```
[a, b, =] -> Embed(+pos) -> 1x TransformerBlock (attn + MLP, NO LayerNorm)
                -> h (final-position residual, 128-d)
                    -> Unembed -> logits -> CE loss
                    -> Proj W_p (128->64) -> L2 normalize -> z -> SupCon loss (x lambda)
```

Then draw two pinboards side by side: one where points colored by true label are pulled
into 97 clusters ("true structure"), one where the same points are pulled into 97 clusters
of *randomly assigned* membership ("shuffled structure — same sizes, same pull, wrong
content").

**Diagram 3 — the causal logic table.** This is the heart of the talk. Draw it as a table:

| Condition | What it isolates | Result |
|---|---|---|
| Baseline | reference | 10/10 grok, median ~20k |
| SupCon-true | the intervention | 22/30 grok, up to 2.75x faster |
| SupCon-shuffled | "is it just the aux loss/geometry?" | 0/20 — NO |
| Norm-matched replay | "is it just the weight-norm side effect?" | 0/15 — NO |
| Grokfast | "how does it compare to optimizer-side SOTA?" | 5/5, 1.2–2.3x, reliable but bounded |

Say explicitly: "each control is a named alternative explanation, and each row kills one."

**Diagram 4 — the mechanism.** Two-axis sketch over training time: weight norm rises
(50 -> ~110-150) under SupCon; two arrows compete — "structure formation (class clusters ->
Fourier circuit)" pulling toward grok, "logit saturation at high norm" pulling toward
freeze. Norm-matched = saturation arrow only (dies at logit scale ~10^4, confidence >0.99).
Shuffled = anti-saturation but no structure arrow (soft logits ~0.65 confidence, flat
Fourier). True = both arrows; whoever wins the race decides fast-grok vs stall — which is
the observed bimodality.

## 2.3 The exact setup (memorize this spec)

- **Task:** (a + b) mod p, p = 97. Tokens [a, b, =], vocab p+1. All p² = 9,409 pairs;
  30% train (2,822), 70% test. Standard grokking regime.
- **Model:** 1-layer decoder-only transformer, d_model = 128, 4 heads (d_head = 32),
  d_mlp = 512 with ReLU, learned positional embeddings, **no biases, no LayerNorm**.
  Readout = linear unembed of the final-position residual stream h.
- **Why no LayerNorm (they WILL ask):** the weight-norm delay law paper showed LayerNorm
  decouples weight scale from function — clamping the norm of an LN network does ~nothing.
  Our norm-matched control only means something in a network where norm matters. Also
  matches Nanda et al.'s analysis architecture.
- **Optimizer:** full-batch AdamW, lr 1e-3, betas (0.9, 0.98), weight decay 1.0. The
  wd = 1.0 is what makes baseline grokking happen at all (circuit-efficiency/cleanup story).
- **Aux loss:** SupCon, L_out variant, temperature tau = 0.1, computed over the full train
  batch on z = normalize(W_p h), W_p: 128->64 linear, no bias. Strengths lambda in
  {0.1, 0.3, 1.0}. Formula (know it):
  L = sum_i (-1/|P(i)|) * sum_{q in P(i)} log[ exp(z_i.z_q/tau) / sum_{a != i} exp(z_i.z_a/tau) ]
  where P(i) = positives of anchor i. True: P(i) = same label. Shuffled: P(i) = same
  pseudo-label from a fixed random permutation of the label vector (drawn once at init —
  preserves exact class-size distribution).
- **Norm-matched control:** record total weight norm per epoch (base params only) from
  each SupCon-true run; rerun plain CE from identical init/split, and after every optimizer
  step globally rescale all base parameters so the total norm equals the recorded value.
  Optimizer moments preserved. This is the delay-law paper's matched-counterfactual clamp.
- **Grokfast:** EMA gradient filter — ema = alpha*ema + (1-alpha)*g; g_used = g + lamb*ema,
  alpha = 0.98, lamb = 2.0.
- **Thresholds:** t_fit = first epoch train acc >= 99%; t_gen = first epoch test acc >= 95%
  sustained 3 consecutive evals (every 50 epochs); delay = t_gen - t_fit. Budget 50,000
  epochs; censored runs conservatively scored at budget.
- **Probes (every 50 epochs):** (1) embedding Fourier concentration — rFFT over the token
  axis of the number-token embedding, power per frequency, report top-8 fraction and Gini;
  (2) class clustering of 1,024 held-out reps — Fisher ratio (between/within-class
  variance) and cosine gap (mean within-class minus between-class cosine sim);
  (3) linear CKA(t, final); (4) total weight norm + logit scale (mean ||logits||) +
  mean max softmax confidence.
- **Scale:** 80 runs, 10 seeds, one RTX 3080, ~4 GPU-hours total, ~80 epochs/sec/run.

## 2.4 The numbers (memorize the bold ones)

| Condition | n | Grokked | Median t_gen | Median paired ratio | Notes |
|---|---|---|---|---|---|
| Baseline | 10 | **10/10** | **19,950** | — | range 5,500–37,550 (~7x seed spread!) |
| Grokfast | 5 | 5/5 | 16,150 | 0.77 | ratios 0.43–0.85 |
| True λ=0.1 | 10 | 8/10 | 28,725 | 1.17 | 2 stalls |
| True λ=0.3 | 10 | 6/10 | 33,325 | 1.22 | 4 stalls |
| True λ=1.0 | 10 | **8/10** | 24,400 | **0.80** | 6/10 faster (6 of the 8 that grokked) |
| Shuffled (pooled) | 20 | **0/20** | >50k | ≥2.2 | **Fisher p = 1.3e-7** |
| Norm-matched (pooled) | 15 | **0/15** | >50k | ≥2.2 | **Fisher p = 1.9e-6** |

Headline singles: fastest grok **2,000 epochs** (λ=1.0, seed 6; same-seed baseline 5,500 →
**2.75x**); seed 9 λ=1.0: 5,550 vs 10,350 (1.87x); seed 3 λ=1.0: 9,800 vs 20,000 (2.04x).
Dose-monotone on grokking seeds (seed 6: 5,100 → 2,850 → 2,000 across λ). Baseline t_fit
is always ~200 epochs; delay ≈ t_gen.

Mechanism numbers: SupCon inflates total weight norm from ~50 (baseline plateau) to
**105–150**; baseline relaxes to ~31 at grok. Norm-matched controls: logit scale explodes
to ~10^4 (baseline ~500), softmax confidence >0.99, test CE >2,000, Fourier top-8 stuck
at ~0.32 (grokked runs reach ~0.97). Contrastive runs (both kinds) hold confidence at
~0.6–0.7 — the anti-saturation channel.

The timing invariant (verified programmatically, 80/80 both directions): every run that
grokked first crossed cosine gap > 0.05 or Fourier top-8 > 0.45 at an earlier epoch, and
no run reached Fourier top-8 > 0.8 without grokking.

## 2.5 The causal logic, spelled out

The claim: *the structural content of a representational prior causally controls the
grokking delay.* Each objection and its dedicated control:

1. **"Any auxiliary loss changes dynamics — it's not the structure."** → Shuffled control:
   identical in form, strength, positive-set sizes, and normalized-projection geometry.
   0/20. The only difference is which examples count as "same," so the effect is the
   content. This is the paper's contribution in one sentence.
2. **"You changed the weight norm, and the norm sets the timescale (delay law)."** →
   Norm-matched replay: 0/15, including the trajectory of the 2x-accelerated run. The norm
   signature of acceleration confers no acceleration.
3. **"It's an optimizer effect."** → Grokfast comparison shows what optimizer-side
   acceleration looks like (uniform, bounded); ours looks different (bimodal, larger peak).
4. **"Seed luck."** → All conditions share seeds: same init, same data split, per-seed
   paired ratios, Wilcoxon on pairs, Fisher on pooled grok fractions.
5. **"You gave the model extra supervision."** → No *new label information* — the same
   labels the CE loss already sees; only a relational *format* is demanded. (Say exactly
   "no new label information"; relational supervision technically differs from per-example
   supervision, and the shuffled control handles the substance of that objection.)

## 2.6 The mechanism story (for the "why does this work?" follow-up)

Baseline grokking (per Nanda et al.): the network first memorizes (attention lookup ≈ a
hash table); under weight decay, a more parameter-efficient Fourier circuit slowly forms
in parallel — embeddings develop cos/sin(ω·a) features at a handful of key frequencies,
attention+MLP combine them so same-(a+b) inputs produce the same logits — and once the
circuit works, weight decay cleans up the memorization and test accuracy jumps.

What SupCon-true adds, three channels:

1. **Structure seeding (the causal channel).** SupCon demands that same-sum inputs collide
   in representation space. The cheapest circuit that makes same-sum inputs collide *is*
   the Fourier circuit (cos/sin of ω(a+b) is constant on each equivalence class). So the
   aux loss creates gradient pressure directly toward the generalizing solution's
   representation geometry, long before CE alone would find it. Evidence: cosine-gap rises
   thousands of epochs pre-grok; Fourier concentration rises earlier than any other
   condition.
2. **Norm inflation (the harmful side-channel).** SupCon is scale-invariant in z (it's
   computed on normalized vectors), so its gradients through W_p into h are largely
   tangential; tangential gradients + Adam's per-parameter normalization grow parameter
   norm despite weight decay — equilibrium norm lands at 105–150 vs ~50. High norm means
   high logit scale means saturation risk (the delay-law/logit-scale mechanism).
3. **Anti-saturation (the protective side-channel).** The contrastive term keeps
   representations moving and logits soft (confidence ~0.65 vs ~0.99 for norm-matched CE),
   preventing the frozen-saturation death that kills the norm-matched control. Shuffled
   also has this — which is why shuffled runs slowly creep upward (some reach 30–47% at
   budget) instead of dying at 1% — but without channel 1 they never complete the
   transition in budget.

Bimodality = the race between channels 1 and 2: if class clusters consolidate before
saturation locks in (visible as early test-accuracy traction by epoch ~1k), the run groks
early; if not, it inherits the slow high-norm regime. Stall counts by dose: 2/10, 4/10,
2/10 — roughly dose-independent, while speedup among grokking seeds is dose-monotone.

## 2.7 Questions you will get, with answers

**Q: Why modular addition and a 1-layer transformer? Isn't this a toy?**
A: Deliberately. It's the one setting where the generalizing solution is fully
reverse-engineered (Nanda's Fourier circuit), so "the right structure" is a measurable
object, not a metaphor — you can't run a true-vs-shuffled-structure experiment if you
don't know the true structure. Toy scale is what makes the causal design affordable:
80 controlled runs, 10 shared seeds, on one consumer GPU.

**Q: The shuffled partition is also learnable by memorization. Is it really matched?**
A: It matches loss form, strength, class sizes, and geometry — it controls for
optimization pressure, not partition learnability, and we say so in limitations. Note the
model does partially fit it (SupCon loss decreases comparably), and at λ=1.0 the wrong
structure fights CE hard enough to delay even memorization to ~10–15k epochs. A
learnability-matched control would need a partition that's wrong-but-equally-compressible;
that's future work and wouldn't change the headline (0/20 vs 22/30).

**Q: Only 10 seeds. Why should I believe the stats?**
A: The primary tests aren't means-with-error-bars: they're categorical outcomes with
overwhelming effect sizes (22/30 vs 0/20 → Fisher exact p ≈ 1e-7) and per-seed paired
contrasts (same init, same split). Where the data is genuinely equivocal — true-vs-baseline
median speed at fixed λ — we report it as non-significant and bimodal rather than claiming
uniform acceleration. Also: baseline t_gen itself spans 5,500–37,550 across seeds, which
is exactly why everything is seed-paired.

**Q: Censoring — stalled runs might grok at 60k and change your medians.**
A: Yes, and we treat them conservatively: censored runs are scored at the 50k budget
(bounding ratios from below), survival curves show budget explicitly, and stalled true-runs
visibly climbing (28% and rising Fourier at budget) are described as delayed, not dead.
The shuffled/norm-matched zeros are not close calls: most sit at 1–10% with flat structure
metrics.

**Q: Doesn't the norm-replay control disturb training itself?**
A: It's the published matched-counterfactual methodology of the delay-law paper: per-step
global rescale after the optimizer step, moments preserved. Their own ρ=1 arm shows the
projection at the natural norm is inert. And our replay isn't at some exotic norm — it's
at the exact norm the successful SupCon runs lived at, which is what makes the comparison
decisive.

**Q: Is this just distillation / supervision smuggling?**
A: No new label information enters — positives are defined by the same train labels CE
already consumes. The manipulation is purely the *format*: "make these representations
similar" vs "make these logits correct." The shuffled arm shows format-with-wrong-content
does nothing (worse than nothing), so the result isn't "more supervision helps."

**Q: Would this scale? Would you do this to a real LLM?**
A: Unknown and I'm careful to say so. The honest scaling story: (1) contrastive
structure-injection needs known equivalence structure, which real tasks rarely expose —
the research direction is self-supervised positives from data invariances; (2) the negative
results likely scale better than the positive one: "wrong representational pressure blocks
generalization" and "internal structure metrics precede capability emergence" are the
transferable claims; (3) there's already a pre-training analogue of grokking in LLMs
(delayed grammatical generalization), which is where I'd test the probe-timing claim first.

**Q: Why does this matter for safety/interpretability?** (Anthropic-specific)
A: Three connections. First, emergence prediction: capability jumps that look sudden in
behavior are gradual in representation space — our 80/80 timing invariant is a clean,
causal demonstration that internal probes lead behavior, which is the premise behind
using interpretability to anticipate emergent capabilities. Second, steering: we show
internal representational structure is a control surface, not just a diagnostic — both
directions (accelerate with right structure, block with wrong structure) worked. Third,
methodology: the paper is basically an exercise in adversarial self-review — every claim
paired with the control that could kill it — which is the epistemic standard
interpretability work needs.

**Q: What's the weakest part of the paper?**
A: Three things, in order: single task family and architecture; the bimodality means the
practical speedup story is "sometimes 2.75x, sometimes worse" — the mechanism paper is
strong but the method paper is conditional; and the supervised nature of the prior. I'd
also flag the stall-mode mechanism as suggestive rather than proven — the race account
fits the trajectories but we didn't intervene on the race itself (e.g., λ annealing).

**Q: What would you do next?**
A: (1) Kill the trap: norm-preserving contrastive gradients (project out the radial
component) or λ annealing — prediction: keeps acceleration, removes stalls. (2)
Self-supervised positives from algebraic invariances (e.g., commutativity: (a,b)~(b,a))
— removes the label-supervision caveat. (3) Replicate on p=113, S5 composition, sparse
parity, and an MLP. (4) The LLM pre-training analogue: track cluster-formation probes
during delayed grammatical generalization.

**Q: How much compute/time did this take?**
A: One RTX 3080, ~4 GPU-hours for all 80 runs, about a day end-to-end including the paper.
Small-scale rigor is a feature: every claim is re-runnable overnight.

## 2.8 Related work — the mental map (name-drop correctly)

- **Power et al. 2022** — discovered grokking (OpenAI, modular arithmetic).
- **Nanda et al. 2023** — reverse-engineered the Fourier circuit; progress measures;
  gradual-formation-then-cleanup. Our architecture and Fourier probes follow this.
- **Liu et al. 2022 (effective theory)** — representation quality gates generalization;
  conceptual ancestor.
- **Liu et al. 2023 (Omnigrok)** — init-scale/data-size mismatch account; norm story precursor.
- **Varma et al. 2023** — circuit efficiency: generalizing circuit wins under weight decay.
- **Grokfast (Lee et al. 2024)** — EMA gradient filter; our optimizer-side baseline.
- **The three 2026 must-reads we position against:**
  - *Radial Suppression* (2606.32000) — geometric penalty on rep norms, 6.3x; explicitly
    can't isolate mechanism → our shuffled control answers their limitation.
  - *Two Speeds* (2605.27078) — representation/readout decomposition; slow-encoder account,
    observational → we're the interventional test.
  - *Weight-Norm Causal Delay Law* (2606.13753) — clamped norm sets timescale exponentially
    (T ∝ e^{7.5ρ}); LayerNorm caveat → dictated our architecture and the replay control.
    Plus *logit-scale mediation* (2606.18465) — the saturation mechanism our norm-matched
    controls exhibit on cue.
- **SupCon (Khosla et al. 2020)** — the loss, repurposed as a structure-injection device.

## 2.9 Whiteboard run-of-show (15-minute version)

1. (2 min) Diagram 1 + "why grokking is the lab mouse of emergence."
2. (2 min) The gap: all known accelerators are structure-agnostic; representation-first
   accounts are observational. State the causal question.
3. (3 min) Diagram 2 + the two pinboards. Emphasize "identical except content."
4. (2 min) Diagram 3 table; deliver the three headline numbers (22/30 & 2.75x; 0/20,
   p=1.3e-7; 0/15).
5. (3 min) Diagram 4 mechanism: three channels, the race, the bimodality as evidence.
6. (1 min) The 80/80 timing invariant — probes lead behavior, causally.
7. (2 min) Limitations + what's next + the safety connection.

Practice tip: the moment most likely to impress is delivering row-by-row of Diagram 3
*as alternative explanations being eliminated*, not as "conditions we ran." Frame every
control as "here's the objection; here's the run that kills it."
