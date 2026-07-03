# What We Did, In Plain English

*A no-jargon explanation of the research in this repository.*

---

## The problem we tackled

AI language models (like the ones behind chatbots) read text as a long line of
word-pieces. To make sense of a sentence, the model needs to know not just
**what** each piece is, but **where** it is — "dog bites man" and "man bites
dog" use the same words, so word order is everything.

The standard way to give a model this sense of order is basically **seat
numbers**: piece #1, piece #2, piece #3, and so on. That works fine — until the
text gets longer than anything the model saw during training. If a model was
trained on texts up to 500 pieces long and you hand it 4,000 pieces, the seat
numbers past 500 are like rows in a theater the model has never seen. It gets
confused, and its performance falls apart. This is a real, well-known weakness:
it's part of why AI models have "context limits" and degrade on very long
documents.

## Our idea

Think about how *people* keep track of "where" things are in a conversation.
You don't remember that an important detail was "the 3,417th word." You
remember it relative to meaningful reference points — "she mentioned that right
after she brought up the contract."

We built that intuition into the model. Our method is called **Semantic
Reference Frames (SemRF)**, and it has two parts:

1. **Landmarks instead of seat numbers.** The model learns a small set of
   "anchor points" — think of them as landmarks in the space of meanings. Every
   word-piece gets described by which landmark it's closest to ("this is a
   punctuation-ish thing," "this is a letter-ish thing") plus a small note about
   how it differs from that landmark. Position and timing are then tracked
   *relative to these landmarks*, never as an absolute number.

2. **Different memory spans for different kinds of things.** Here's the key
   twist: not everything deserves to be remembered equally long. In English
   text, the letter you just read matters a lot for the next letter, but ten
   sentences later it's irrelevant. A section heading or a table divider,
   though, stays relevant for pages. So we let each *kind* of token (each
   landmark group) learn its own "fade-out speed" — how quickly the model's
   attention to it should weaken with distance.

Nobody tells the model which things should fade fast or slow. It figures that
out on its own, from data.

## What we actually did

- Built one identical "model brain" and eight interchangeable "position
  systems" for it — ours plus seven established methods from the research
  literature (including the ones used in today's major AI models).
- Trained around **160 models** on a single consumer gaming GPU, keeping
  everything else perfectly equal so the comparison is fair — same brain, same
  data, same training budget, multiple runs with different random starts so
  results aren't flukes.
- Tested them on two fronts:
  - **Three lab-style memory games**: recalling a fact stored earlier
    (like remembering a phone number from a list), reporting the *most recent*
    value of something that kept changing (like tracking a score), and copying
    scattered items out in order.
  - **Real text**: predicting Wikipedia text character by character, including
    texts up to **8× longer** than anything seen in training.
- Wrote everything up as a formal research paper (in this repo, `paper/main.pdf`),
  with every number in the paper automatically checked against the raw
  experiment files.

## What we found

**1. Our method read real text best — at every length.** On the Wikipedia
test, SemRF made the most accurate predictions of all eight systems at the
normal training length. More importantly, when we stretched the text to 8×
longer, most systems got dramatically *worse* — one older method became about
25× worse, and the method used in many modern AI models tripled its error —
while ours actually kept **getting better** as the text got longer. More
context helped instead of hurting, which is how you'd want an AI to behave.

**2. It was the only one to pass all three memory games.** Every other system
failed at least one. The "most recent value" game was especially telling: the
popular methods either failed it or were unreliable, while ours got it perfect
even on streams 4× longer than training — because "prefer the recent one" is
exactly what a learnable fade-out is good at.

**3. The coolest part: we can see what it learned.** When we opened up the
trained model, the landmarks it invented on its own turned out to be
recognizable categories: lowercase letters clustered together, digits together,
punctuation together, and so on. And the fade-out speeds it chose made real
sense: **letters got fast fade-outs** (spelling is a local affair) while
**punctuation, line breaks, and structural markers got slow fade-outs**
(document structure matters over long distances). In other words, the model
discovered by itself that *structure is long-range and spelling is local* —
which is a rather human insight, and we never told it that.

**4. Honest fine print.** We also found and reported the weaknesses. A recent
competing method (CABLE) beats ours on one specific game — recalling facts
across very long stretches of filler — though it completely fails a different
game and reads real text worse. And our method's learned fade-outs can "forget
too fast" across gaps much longer than anything seen in training. No method won
everything; ours won the most, and we say exactly where it doesn't.

## Why this matters

- **Longer documents, more gracefully.** A major frustration with AI models is
  degraded quality on long inputs — long contracts, books, codebases, chat
  histories. This work adds evidence for a promising direction: stop tying a
  model's sense of position to absolute numbers, and tie it to *meaning*
  instead. Our results show you can do that and get better accuracy at normal
  lengths *and* graceful behavior far beyond them.

- **Time-awareness.** Lots of real tasks are about "the latest": the latest
  price, the latest diagnosis, the latest version of a fact. We showed a clean
  mechanism that makes models reliably prefer recent information — and proved,
  by removing pieces one at a time, that this specific mechanism is what does it.

- **AI you can look inside.** Most tweaks to AI internals are black boxes. This
  one is inspectable: you can literally print out what categories the model
  invented and how long it chooses to remember each. That matters for trust,
  for debugging, and for science.

- **A recipe others can reuse.** The whole thing costs almost nothing —
  under 2% extra model size — and the code here lets anyone rerun every
  experiment, figure, and table with a couple of commands, on one ordinary GPU.
  We also documented pitfalls we hit (like the fact that a model's ability to
  do recall "switches on" suddenly during training, and small design choices
  can keep it from ever switching on) that other researchers can sidestep.

## What this is *not*

To keep expectations honest: these are small research models (about 11 million
parameters — thousands of times smaller than commercial chatbots), trained on
one dataset, on one GPU. We showed the idea works and *why* it works at this
scale. Whether it holds at chatbot scale is exactly the kind of follow-up
question this paper is designed to motivate — and to make easy for others (or
us) to test next.

## One-sentence version

**We taught a language model to track "where and when" relative to meaningful
landmarks instead of seat numbers — and it read text better, handled 8×-longer
inputs gracefully where standard methods collapsed, and, without being told,
discovered that structure deserves a long memory and spelling only a short
one.**

---
---

# Part 2: Whiteboarding It Cold — The Interview Deep-Dive

*Everything above is the elevator version. This part is what you need to stand
at a whiteboard and defend the work to a research engineer: the pitch, what to
draw, the math, the numbers to memorize, every design decision with its "why,"
the war stories, and the hard questions with answers.*

---

## 1. The 60-second pitch (memorize the arc, not the words)

> "Transformers need a positional signal, and every mainstream choice —
> sinusoidal, learned, RoPE, ALiBi, T5 — is ultimately a function of the
> absolute index or the index gap. That conflates *what* a token is with *when*
> it occurs: a comma and a rare entity get the same distance treatment. I built
> SemRF: tokens are softly assigned to a small set of learned semantic anchors,
> and the attention bias is assembled from anchor affinity, residual alignment,
> and — the key term — a temporal decay whose *rate is a learned function of
> the token's semantic frame*. It's constructed to reduce exactly to ALiBi, and
> I initialize it *at* ALiBi, so it starts from a known-good extrapolating
> operating point and learns semantic structure on top. Across 8 positional
> schemes under matched everything, it's best on enwik8 at every context length
> out to 8× the training length, it's the only scheme that solves all three of
> my diagnostics, and — the part I care most about — the trained anchors are
> interpretable: unsupervised, they recover character classes, and the learned
> time constants say 'structure is long-range, orthography is local.'"

If they remember one sentence: **"I made the decay rate of attention a learned
function of what the token *is*, not just where it sits — and the model used
that freedom in an interpretable way."**

---

## 2. The whiteboard flow (3 panels)

**Panel 1 — the problem (30 seconds).** Draw a row of boxes numbered 1…512,
dashes out to 4096. Write `A_ij = q_i·k_j/√d (+ position)`. List the families:
absolute adds `p_i` to the input (dies past trained index); RoPE rotates q,k by
angle ∝ index (relative, but degrades out-of-distribution); ALiBi/T5 add
`b(i−j)` to the logits (extrapolates, but content-blind). Say the punchline:
"every `b` is a function of the index gap only — a comma and a named entity
decay identically. That's what I attack."

**Panel 2 — the mechanism (the core drawing).** Draw K stars ("anchors") in a
2-D meaning space, a token projected in, soft-assignment arrows to the nearest
stars, and the residual vector from the weighted centroid to the token. Write
the math beside it (Section 3), in order: assignment → residual → three bias
terms → gates → ALiBi reduction. Close with: "the bias is computed **once**
from token embeddings and shared across all layers — same plumbing as
ALiBi/T5, so the comparison is apples-to-apples."

**Panel 3 — results (two curves).** (1) bpc vs context length, log-x:
exploding curves for sinusoidal/learned/RoPE; two gently *declining* curves
for ALiBi and SemRF, SemRF strictly below, gap widening; mark 1.4246@512 and
1.340@4096. (2) Recency task: SemRF/ALiBi/CABLE flat at 1.0; learned-absolute
collapsing 1.0 → 0.36 at 4× — "absolute-position saturation in one picture."

---

## 3. The math you must write from memory

Setup: `x_i ∈ R^d` is the token embedding. **No absolute position is ever
added anywhere.**

**Anchor assignment (the frames):**
```
u_i = W_u x_i                       # project to anchor space (d_a dims)
α_i = softmax(u_i Aᵀ / τ)           # A ∈ R^{K×d_a}: K learned anchors
r_i = u_i − α_i A                   # residual offset within the frame
```
Gloss: α_i = "which frame am I in" (soft, sums to 1); r_i = "where exactly,
within that frame."

**Three bias terms:**
```
sem_ij      = α_iᵀ B α_j                    # frame–frame affinity, B ∈ R^{K×K}
res_ij      = (W_q r_i)·(W_k r_j) / √d_r    # within-frame residual alignment
time_ij^(h) = −σ_i^(h) · |i−j|              # frame-conditioned temporal decay
    where σ_i^(h) = Σ_k α_ik · softplus(s_hk),   s ∈ R^{H×K}
```
Gloss for the time term: each head h keeps a slope *per frame*; a token's decay
rate is its membership-weighted mixture. Softplus keeps it non-negative
(always decay, never anti-decay).

**Combination (causal, j ≤ i):**
```
b_ij^(h) = g_sem·sem_ij + g_res·res_ij + g_time·time_ij^(h),   g_• = exp(γ_•)
A_ij^(h) = q_i·k_j/√d_head + b_ij^(h)
```

**The reduction — the design's spine, write it:**
```
g_sem = g_res = 0,  s_hk = s_h ∀k   ⟹   b_ij^(h) = −softplus(s_h)·|i−j| = ALiBi
```
**Initialization = exactly that point**: per-frame slopes set to ALiBi's
published per-head slopes via inverse-softplus; content gates ≈ 0.05;
g_time = 1. Say: "training *starts at ALiBi* and buys semantic structure only
if the data pays for it."

**Cost:** extra params = W_u + A + B + {W_q, W_k} + s ≈ **1.7%** at our scale
(10.92M vs 10.74M). The bias is one B×H×T×T tensor per forward — O(T²) like
any additive bias — reused across layers. ~**1.55×** step time vs RoPE
because arbitrary float masks fall off the fused attention kernel.

---

## 4. Numbers to know cold

**enwik8** (byte-level, d=384 / 6 layers / 6 heads, ≈11M params, train context
512, 10k steps, 2 seeds; bits-per-character, lower = better):

| | 512 (train) | 1024 | 2048 | 4096 (8×) |
|---|---|---|---|---|
| **SemRF** | **1.4246 ± .0006** | **1.379** | **1.354** | **1.340** |
| RoPE | 1.4276 | 2.74 | 4.01 | 4.52 |
| ALiBi | 1.4348 | 1.389 | 1.367 | 1.356 |
| CABLE | 1.4409 | 1.395 | 1.368 | 1.352 |
| T5-bias | 1.5084 | 1.53 | 2.39 | 3.35 |
| Learned | 1.5194 | 2.93 | 3.76 | 4.21 |
| Sinusoidal | 1.5615 | 3.71 | 21.4 | 34.6 |
| NoPE | 1.6306 | 2.21 | 3.31 | 3.99 |

Sound bites: best at *every* length; margin over ALiBi widens
(−0.010 → −0.016); only the three decay methods improve monotonically with
context; sinusoidal's 34.6 is the shock number.

**Diagnostics** (d=256 / 4L / 8H, lr 1e-3, 3 seeds; train-length accuracy):

| | Recall (15k steps) | Recency (10k) | Copy (5k) |
|---|---|---|---|
| **SemRF** | **1.000 (3/3)** | **1.000** | **0.999** |
| CABLE | 1.000 (3/3) | 1.000 | **0.174 — fails** |
| ALiBi | 0.75 (2/3 seeds) | 1.000 | 0.999 |
| RoPE | 0.26 (0/3) | 0.60 (1/3) | 1.000 |
| Learned | 0.26 | 1.000 | 1.000 |

Extrapolation punchlines: **recency @4×** — SemRF/ALiBi/CABLE 1.0, learned
**0.36**, no-time ablation **0.55**. **Copy @4×** — T5 0.95, no-time 0.88,
SemRF 0.73 > ALiBi 0.70, RoPE/learned <0.10. **Recall gaps** — CABLE 1.0 at
every gap (wins), SemRF 0.87@64 → 0.39@160+ (our documented weakness), no-time
can't even train (0.33).

**Interpretability:** 12 of K=32 anchors used by the 98 occurring bytes;
frames = lowercase / digits / uppercase / whitespace / punctuation / the `|`
wiki delimiter (a singleton frame!). Slowest decay: `|`, punctuation, newline.
Fastest: lowercase. Within-head slope spread up to **88%** of the head mean.

**Scale of evidence:** ~160 trained models; every numeric claim machine-
verified against raw result files by a 25-check script.

---

## 5. Every design decision, with its "why"

1. **Additive bias, not input embedding or rotation.** Plugs into any
   backbone, composes with content attention instead of entangling with it,
   and enables the ALiBi-reduction argument. Same injection point as ALiBi/T5
   → fair comparison.
2. **Bias computed once, shared across layers.** Matches ALiBi/T5 plumbing,
   O(1) cost in depth, and keeps interpretability clean: one frame set, one
   slope table. (CABLE recomputes per layer — one of our contrasts.)
3. **ALiBi-point initialization — the load-bearing decision.** See war story 1.
   Principle: *initialize a generalization at its best-known special case.*
4. **Frame-level (not token-level) decay.** K slopes per head, not one per
   token: it's the interpretability handle, a regularizer (frame-mates share a
   time constant), and the honest contrast with CABLE (token-level wins one
   task, fails another, reads text worse).
5. **Span-growing extrapolation with memory load fixed.** Adding recall pairs
   confounds positional generalization with recall *capacity* (Zoology/MQAR).
   So: same 8 pairs, growing filler gap (filler in-distribution at train time
   so OOD-token effects don't masquerade as length effects); same variables,
   longer streams; same data tokens, bigger blank field.
6. **Calibrate tasks so the best baseline solves the training distribution.**
   Otherwise you compare failure modes, not methods (war story 2).
7. **Small-lab statistical honesty.** Mean ± std everywhere; paired t-tests
   labeled indicative at n=3; bistable seeds (ALiBi 2/3, RoPE 1/3) reported as
   data, never averaged into mush.

---

## 6. War stories ("what went wrong?" — these are gold in interviews)

**1 — The init failure.** First sanity run: SemRF scored 0.12 on selective
copy while ALiBi — a strict *special case* of SemRF — solved it. A
generalization losing to its own special case is a training-dynamics bug, not
an expressiveness bug: content-bias noise at init drowned the positional
signal. Fix: initialize at the special case. 0.12 → 0.998 in one change.
Lesson: *expressiveness ⊄ trainability.*

**2 — The phase transition.** Associative recall sat at chance for every
method. Isolation ladder: 1 pair → solved trivially (positionally); 2 pairs →
loss stuck at exactly ln(2) — coin-flipping between the two values in context,
i.e., the failure was *content matching* itself. Sweeps: precision no effect;
d=128 never forms the circuit at any step count; dense supervision helps; and
the decisive probe — same config, 6k steps = 0.26, **15k steps = 1.00**. A
sharp, late phase transition in induction-circuit formation. It bit us again:
widening the training gap jitter from (0,8) to (0,32) pushed every baseline
back past the transition; only one SemRF seed crossed. We kept that as an
appendix finding and ran the headline comparison in the calibrated regime.
Lesson: *near a formation boundary, small task changes move every method;
calibrate before you compare.*

**3 — The novelty challenge.** Mid-project: "Moschella et al. 2023 already did
this — kill it." Verified from sources: that paper anchors *sample embeddings*
for zero-shot latent-space stitching — nothing on position, attention, or
time. It became our credited inspiration, not prior art. But the same search
surfaced CABLE (2025), genuinely adjacent — so instead of just citing it, we
implemented it from the authors' reference code and ran it as the 8th
baseline. It won one diagnostic (gap-invariant recall), failed another at
train length (copy, 0.17), and lost on enwik8 at every length. Lesson: *the
right response to "you've been scooped" is a source check, then an
experiment.*

---

## 7. Hard questions you will get, with answers

**"The RoPE gap at train length is tiny. Is that claim real?"**
Correct to be careful: 1.4246 vs 1.4276 is ~2 std with n=2 seeds. The robust
claims are "≥ every baseline at 512" and "dramatically better beyond 512."
Mechanistically, decay biases act as a learned multi-scale locality prior,
which byte-level LM rewards; SemRF refines that prior per frame — and its
margin over ALiBi *grows* with context, which a fixed slope cannot do.

**"CABLE beats you on gap recall — so why is your structure better?"**
Real trade-off, reported in the paper. CABLE's cumsum distance lets filler
tokens learn ≈0 increment — elegant, wins that task. The same unstructured
freedom gives it no stable ordinal geometry (fails copy at *train* length)
and costs it on real text (worst of the four competitive methods at 512).
Frame-level structure is a regularizer with readable parameters. Want both?
The natural hybrid — frame-conditioned decay over *accumulated content*
rather than raw distance — is my stated future work.

**"Your slopes over-decay at 20–40× training gaps. Bad, no?"**
Yes, quantified: 0.87@64 → 0.39@160+. Learned scale parameters fit the
training scale; ALiBi is scale-free by fiat. Fixes to try: log-distance
parameterization, regularizing slopes toward the ALiBi prior, accumulated-
content distance. Note the flip side: the same learnability is why our recall
circuit forms 3/3 seeds vs ALiBi's 2/3 and why recency works at all.

**"n=2 or 3 seeds. Really?"**
One RTX 3080, ~160 runs, honest budget. Mitigations: population std reported
(enwik8 seed spread ≤0.004 bpc vs effect sizes 0.01–33), paired tests labeled
indicative, bistability reported not suppressed, every claim machine-checked
against raw logs. No qualitative separation hinges on a single seed.

**"Would it scale? What breaks at 7B?"**
Unknown — stated in limitations. Watch-list: (1) BPE instead of bytes — do
frames become POS/entity-like classes? K probably grows; (2) the O(T²) bias
needs a fused kernel — the time term folds into FlashAttention ALiBi-style
(per-token slope × distance), and sem/res are low-rank and could ride an
extra QK component; (3) whether content-conditioned decay still pays once
models are large enough to route around fixed decay. The +1.7% params is
scale-friendly.

**"Inference / KV-cache story?"**
Decoding appends one token: its α, r, σ come from one projection of its own
embedding — O(1) extra state. Its bias against cached keys is σ·distance plus
low-rank content terms — the same incremental pattern as ALiBi. Nothing
depends on absolute index, so no cache re-positioning issues.

**"Isn't this just ALiBi with extra steps? Or CoPE?"**
vs ALiBi: strict generalization *by design* — that's the init point, not an
accident; the delta is content conditioning, and the ablation makes it causal
(recall never forms without the time term: 0.33 vs 1.00). vs CoPE: CoPE
changes *what gets counted* (a gated position counter per query); SemRF keeps
raw distance but changes *how fast it matters* per semantic frame, adds
frame-affinity content terms, and exposes interpretable per-class time
constants, which CoPE doesn't.

**"Which single result would you defend to the death?"**
Recency plus its ablation: SemRF 1.0 at all lengths; learned-absolute
1.0 → 0.36 (saturation demonstrated); remove the time term → train accuracy
survives (0.997) but extrapolation collapses (0.55). Causal attribution, not
correlation.

**"Next steps with real compute?"**
WikiText-103 then a BPE run (do frames become syntax/entity classes?); fused
kernel; the CABLE-hybrid distance; a mechanistic pass — do induction heads
form *earlier* under SemRF? (attention-capture hooks already in the repo);
needle-in-a-haystack and long-doc QA for task-level validation.

---

## 8. Fluency check — terms to use correctly without pausing

- **bpc**: bits per character = NLL/ln 2 per byte. Large-model enwik8 SOTA is
  ≈0.9–1.0; 1.42 is right for an 11M-param class.
- **Induction head**: the two-layer circuit (previous-token head composing
  with a match-and-copy head) implementing [A][B]…[A]→[B]; the thing that
  "clicks" in our phase transition — and the subject of Anthropic's own
  in-context-learning work (Olsson et al.), which we cite. Know that link.
- **MQAR / Zoology**: multi-query associative recall; recall capacity scales
  with model width — the reason our extrapolation grows *span*, not pairs.
- **ALiBi slopes**: geometric per-head sequence (½, ¼, …, with a defined
  scheme for non-power-of-2 head counts) — inherited at init via
  inverse-softplus.
- **Backbone details**: pre-norm decoder-only transformer, GELU MLPs, GPT-2
  scaled residual init (0.02/√(2L)), tied embedding/head weights, AdamW +
  cosine schedule + warmup, bf16 autocast.

## 9. If you get ten extra minutes: run the live demo

`python scripts/anchor_analysis.py` prints the anchor clusters and the
slowest/fastest-decaying frames from the shipped checkpoint in seconds.
Showing an interviewer the model's learned "structure is long-range,
orthography is local" table *live* beats any slide.
