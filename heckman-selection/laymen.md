# What this project did, in plain English

## The problem: your data was chosen for you, and the AI never knew

Almost every AI is trained on data that somebody, or some process, *selected* — and
usually the AI has no idea this happened. A bank only sees whether a loan was repaid for
the loans it *granted*; it never sees what would have happened to the applicants it turned
down. A hospital only records disease outcomes for the patients it *admitted*. A team
running experiments only samples the settings someone thought were worth trying.

This matters because the piece that got left out is not random. The loan officer who
rejected an applicant often had a hunch — some scrap of information not written in any
column of the spreadsheet. So the loans you *do* see are systematically different from the
ones you don't, in a way tied to the very thing you're trying to predict. Statisticians
call this **selection on unobservables**: the invisible reason a data point was collected
is tangled up with its answer.

The standard machine-learning fix — "reweight the data you have to look like the data you
wanted" (importance weighting) — **cannot fix this**. It can correct for *who* you sampled;
it cannot correct for a bias hiding in *why* you sampled them, when that why is invisible.
You can have perfect knowledge of the sampling odds and still get the wrong answer. Worse,
a model trained this way is not just uncertain in the regions it avoided — it is
*confidently wrong* there, which is the dangerous kind.

## The 1979 idea we borrowed

Economists solved this in 1979. James Heckman won a Nobel Prize partly for a trick: instead
of pretending the missing data isn't there, you **model the selection itself**. You write
down two linked equations — one for "was this observed?" and one for "what's the outcome?"
— and let them share a hidden correlation. That correlation is exactly the invisible bias,
and once you estimate it, you can subtract it back out. The machinery has lived in
econometrics textbooks for decades. Almost nobody has plugged it into modern deep learning.
We did — twice.

## Paper A: teaching an AI to know what it doesn't know

We built the Heckman correction into a neural network's **uncertainty estimate** — its
sense of "how sure am I?" In a controlled test where we secretly know the truth, we sampled
training data in a biased way (avoiding certain regions for reasons tied to the answer) and
watched what happened:

- Ordinary confidence estimates (deep ensembles, dropout, Gaussian processes) were
  **overconfident exactly where the data had been avoided** — the worst place to be
  cocky.
- Reweighting with *perfect* knowledge of the sampling odds **did not help** — as the
  theory predicts.
- The Heckman-corrected network **restored honest uncertainty** — *but only when it had an
  "instrument"*: some extra clue that affects whether data was collected but not the answer
  itself (like which sensor took the reading). Without that clue, the fix degrades, and
  we show exactly how much rather than hiding it.

We also checked a real, slightly uncomfortable example: public AI leaderboards. Researchers
report their method on the benchmarks where it shines and quietly skip the others. That's
selection on unobservables too — and when we tried to correct it without an instrument, the
math broke down in the textbook way, which is itself the lesson: *leaderboards need a
"why did they report this" variable before anyone can de-bias them.*

## Paper B: the runs that survived were lucky, not good

Modern AI tuning uses a shortcut called **successive halving**: start hundreds of training
runs, peek early, kill the ones doing worst, keep going with the survivors. It saves
enormous compute. But there's a catch we made precise.

Early scores are noisy. A run can survive an early cut partly because it got *lucky* on a
noisy measurement, not because it's truly better. So the survivors are a mix of genuinely
good runs and lucky ones — and the lucky ones **regress to the mean**: they disappoint
later. Any tool that learns to predict final performance from these survivors inherits the
bias. It's the same Heckman/selection structure, now hiding inside a hyperparameter tuner.

We showed the bias is real and grows exactly as the theory says — with more noise and more
aggressive cutting. Then we were honest about the punchline: **at the noise levels of real
benchmark data (LCBench), the bias is tiny** — successive halving is robust where its
measurements are clean. It only bites in genuinely noisy regimes (small validation sets,
or hard tasks like ImageNet tuning, where we measured meaningful bias). We built the
selection-corrected predictor anyway; it fixes the bias in prediction, and it *rescues*
naive curve-extrapolation tuners from a "winner's curse," but it does **not** beat the
plain, cheap method at the actual tuning decisions on clean benchmarks. We report that gap
squarely instead of overselling.

## The honest thread running through both

The same caveat governs both papers, and we put it up front every time: the Heckman fix is
trustworthy **only when you have that "instrument"** — a variable that nudged whether data
was collected but not the answer. With one, it works. Without one, it leans entirely on an
assumption about the shape of the noise and gets shaky. Econometricians have known this for
forty years; we made sure a machine-learning audience can't miss it. Sometimes the useful
result is a clean "here's when it works," and sometimes it's an honest "here's a real
effect that turns out to be small in practice." This project shipped both.

---

# Section 2 — The whiteboard guide (everything to present it cold)

This section is written so you can walk to a whiteboard with nothing prepared and
reconstruct the entire project — the mechanism, the math, the diagrams, the exact numbers,
and the "why it matters." Read it until you can *derive* the crux rather than recite it.
If you can explain **why importance weighting fails but Heckman works** from first
principles, you own this project.

## 2.0 The pitch ladder (memorize the top two rungs)

**One sentence.** "We took a 1979 Nobel-winning econometrics tool for correcting bias from
non-random data collection, built it into deep neural networks for the first time, and
showed it fixes a failure mode — models being *confidently wrong* exactly where their
training data was avoided — that importance weighting provably cannot fix."

**Thirty seconds.** "ML data is almost always *selected* — you only see loan repayments for
approved loans, only see eval scores where people chose to report. When the reason for
selection correlates with the outcome's *unobserved* part, the standard fix — reweighting
by sampling probability — is structurally helpless. Heckman's 1979 model instead jointly
fits a 'was-it-observed' equation and an 'outcome' equation coupled through their error
correlation, and subtracts the bias out. We instantiated it in deep nets for uncertainty
estimation (Paper A) and inside hyperparameter tuning (Paper B). Headline: with a valid
instrument, oracle-propensity importance weighting gives 43% coverage where it should give
90%; Heckman restores 89%. And we're rigorously honest about when it *doesn't* help."

**Two minutes.** Tell it as the arc in 2.1 → 2.4 below, ending on the honest limitation.
The honesty *is* the pitch at Anthropic.

## 2.1 The core mechanism — the one thing to be able to derive

Draw this model on the board. It is the whole project in five lines.

```
Selection:   s_i = 1[ w_i·γ + u_i > 0 ]           "was unit i observed?"  (a probit)
Outcome:     y_i = f(x_i) + ε_i                    "what is the label?"
Observed:    you see y_i ONLY when s_i = 1
Coupling:    corr(u_i, ε_i) = ρ                    the two errors are correlated
             (u, ε) ~ bivariate normal
```

The entire problem is `ρ ≠ 0`. If ρ = 0, the errors are independent, selection carries no
information about the outcome, and everything is fine (this is "Missing At Random" — the
regime importance weighting handles). If ρ ≠ 0, then **the units you observe have a biased
draw of ε**, because you only kept the ones where `w·γ + u > 0`, and u is correlated with ε.

The math that fixes it — write this too:

```
E[ y | x, w, observed ]  =  f(x)  +  ρσ · λ(w·γ)          where λ(a) = φ(a)/Φ(a)
                                     └────────┘
                                     the bias term (the "inverse Mills ratio")
```

`λ` (the **inverse Mills ratio**) is the mean of a standard normal *truncated* to the region
that survived selection — i.e., "how much upward push did surviving get me, on average."
It's an explicit, estimable function of the selection equation. So:

- **Two-step estimator:** fit the probit → compute λ̂ for each observed point → regress
  `y` on `[x, λ̂]`. The coefficient on λ̂ *is* `ρσ`. Throw the λ̂ term away at prediction
  time and you have the bias-free `f(x)`.
- **Joint MLE:** maximize the full bivariate-normal likelihood over everything at once.

**THE crux — why importance weighting cannot do this (be able to say this cold):**

> Importance weighting corrects *which x's* you have — it reweights the covariate
> distribution so under-sampled regions count more. But the bias here is not about x-coverage.
> It's that **at a fixed x, the y's you kept are a skewed sample** (E[y|x,observed] ≠ E[y|x]),
> because selection filtered on a noise term correlated with y. Reweighting only touches the
> x-marginal; it never reaches the conditional y|x. So even with *oracle* propensities — perfect
> knowledge of the sampling odds — the conditional bias survives. Heckman is the only fix
> because it's the only one that models the y-side error correlation ρ directly.

The one-liner: **"Importance weighting fixes which inputs you have; Heckman fixes which
labels you have at each input. The bias lives in the second, and reweighting can't reach it."**

Our ρ=0.9 experiment is the proof: oracle IW doesn't just fail to help, it makes coverage
*worse* (43% vs the do-nothing baseline's 64%), because upweighting the sparse
selected-against points concentrates training on the units whose noise was most strongly
selected.

## 2.2 The identification caveat — the instrument (this is where you show maturity)

Heckman is only trustworthy with an **exclusion restriction**: a variable `z` that enters
the *selection* equation but not the *outcome* equation (an instrument). Intuition: with an
instrument, selection wiggles for a reason unrelated to the outcome, and that independent
wiggle is what lets you separate "bias from selection" from "real signal." Without one,
identification rides entirely on the bivariate-normal *assumption* being exactly right —
fragile, and every econometrics reviewer knows it.

**In ML, instruments exist and you should name them:** which sensor / site / annotator /
batch collected the data; acquisition-policy randomness (an exploration ε in active
learning); rung-assignment randomness in a hyperparameter scheduler. If you have none of
these, you are in the fragile regime — and our experiments *quantify* how fragile, rather
than pretending otherwise. Leading with this caveat is the honest move and it's the thing
that separates a real result from a hype result.

## 2.3 The three diagrams to draw

**Diagram 1 — the selection picture (Paper A setup).** Draw a scatter of (x, y). Shade the
right side ("selected-against region"). Show the observed points (dense on the left, sparse
on the right) sitting *above* the true curve f(x) in the sparse region — because selection
kept the high-noise survivors there. Draw a tight confidence band from a normal model
hugging those biased points (→ confidently wrong), and a wider, correctly-centered Heckman
band. That picture *is* Paper A.

**Diagram 2 — the coverage-vs-ρ curve (Paper A headline).** X-axis: ρ (0 → 0.9). Y-axis:
"coverage of a 90% interval in the avoided region," dotted line at 0.90. Four curves:
Heckman stays flat on 0.90; deep ensemble sags to ~0.64; oracle importance weighting sags
*worst* to ~0.43; oracle floor at 0.90. The fact that IW is the *lowest* line is the
money shot — it visually kills the standard fix.

**Diagram 3 — survivor bias (Paper B).** Draw several rising learning curves. Put a vertical
"rung" cut early; keep only the top third by their *noisy* value at the cut. Circle a curve
that survived by luck (its noisy point spiked up). Show its true continuation landing
*below* a naive extrapolation. That gap, averaged, is the survivor bias — regression to the
mean, formalized as a Heckman/Tobit structure on the latent curve parameters.

## 2.4 The numbers to memorize (quote 4–6 of these and you're credible)

**Paper A — Heckman-corrected uncertainty.** Selected-against coverage of a nominal-90%
interval at ρ=0.9, *with* instrument:

| Method | Coverage (target 90%) | What it shows |
|---|---|---|
| Oracle importance weighting | **43%** | the standard fix makes it *worse* |
| Deep ensemble (do-nothing) | 64% | overconfident in the avoided region |
| Blind two-head (ablation) | 69% | the extra head alone isn't enough |
| **Heckman (two-step)** | **89%** | ≈ restores calibration |
| Heckman (joint MLE) | 89% | ties two-step; needs a warm-up trick |
| Oracle (knows truth) | 90% | the ceiling |

- Recovered the true correlation: **ρ̂ = 0.90** (truth 0.9). Kill-check paired test **p = 0.008**
  (the minimum possible at 8 seeds — every seed favored Heckman).
- **Without an instrument:** the correction breaks — ρ̂ collapses to **0.14**, mean bias stays
  large (0.58). Subtle honest point: the MLE's *coverage* still looks okay (85%) but only
  because weak identification balloons its intervals — it's uncertain, not correct. We flag
  this rather than exploit it.
- **Real data (California housing, wine):** by region-ECE (calibration error, which can't be
  gamed) Heckman is the best non-oracle method on both (ECE **0.05** / 0.14 vs deep ensemble
  0.20 / 0.22, IW 0.24 / 0.25; p<0.001). Baselines that "win" on raw coverage do it by
  over-widening *everywhere* — the GP covers **98%** in well-sampled regions where it should
  cover 90%. Lesson: coverage misleads; ECE tells the truth.
- **Correctness gate:** our estimators reproduce the published seven-digit Stata output on
  the classic RAND health-insurance dataset to **5×10⁻⁷** (MLE), ρ̂ = 0.7356. This gate ran
  *before* any experiment — you can't claim a novel result with a buggy estimator.
- **Real-world vignette:** public benchmark leaderboards are selective-reporting panels;
  fit Heckman without a reporting instrument and **8 of 13** fits hit the |ρ|=1 boundary —
  the textbook no-instrument pathology, showing up in the wild.

**Paper B — survivor bias in tuning.**

- Bias is **dose-responsive**: grows from 0.001 to **0.054** accuracy units as noise rises,
  and with cut aggressiveness — exactly as the mechanism predicts (two falsifiable
  predictions, both confirmed).
- **The honest null:** at the noise level of real LCBench curves (~0.006), the bias is
  ~0.002 — negligible. Successive halving is *robust where its measurements are clean*. We
  report this as the answer, not a disappointment. It only bites in noisy regimes (small
  val sets; PD1's ImageNet task).
- The corrected surrogate cuts the extrapolation "winner's curse" ~**7×** (top-decile
  over-prediction 0.183 → 0.026) and ties plain last-value SH on LCBench decisions — but is
  *worse* on PD1 (0.0078 vs 0.0017 regret): **the prediction–decision gap**. Better
  predictions, worse decisions, because refitting a prior on small pools adds decision noise.
- A one-parameter "just inflate the variance" hack captures most of the benefit. Nothing
  beats plain last-value SH at matched budget. We say so.
- Population recovery is honest too: the survivor-only correction *overshoots* (0.744 vs
  truth 0.780, from a naive 0.864); the estimator we actually recommend just refits on **all**
  observed prefixes (0.784, near-perfect) — "use the data you already have, don't model the
  selection." The Arellano–Bond dynamic-panel variant we tried was worse than naive and we
  dropped it without ceremony.

## 2.5 Why Anthropic specifically should care (connect it to the mission)

Frame it as: **selection-on-unobservables is everywhere in the frontier-AI data pipeline,
and the safety-relevant failure is being confidently wrong exactly where you have no data.**

- **Preference / RLHF data is selected.** Annotators choose what to label; the policy
  generates what gets rated; raters skip ambiguous cases. The "why this example got a label"
  is tangled with the label. That's selection on unobservables in the reward-model training set.
- **Evaluation is selected (our A-E4 vignette, literally).** Labs report benchmarks where
  they win. SOTA leaderboards are missing-not-at-random panels; aggregate "progress" inherits
  an unidentified selection bias. Knowing you *can't* de-bias these without a reporting
  instrument is itself a useful, sobering result for anyone reasoning about eval trends.
- **Red-teaming / safety data is the extreme case.** You deliberately collect data where the
  model fails — maximal selection on the outcome. Naively fitting on it misestimates the true
  failure rate.
- **Calibrated uncertainty is a safety primitive.** "The model knows what it doesn't know" is
  exactly what you want before deploying in a new regime. Our failure mode — *tight, wrong*
  intervals in the regions training avoided — is the precise thing that makes overconfident
  deployment dangerous. A method that widens intervals for the *right structural reason*
  (not just globally) is a step toward trustworthy abstention.
- **The culture fit is the honesty.** Anthropic prizes calibrated claims. This project has a
  pre-registered gate, explicit kill conditions ("if oracle IW matches Heckman, our premise
  is wrong — report and stop"), a shipped honest *null*, and a limitation stated in the
  abstract of both papers. That's the epistemic style, demonstrated, not asserted.

## 2.6 What makes it real science (the rigor story, in case they probe method)

- **Faithfulness gate first:** reproduce a 45-year-old estimator to 7 digits against Stata
  before trusting your deep version. No result ships on a buggy foundation.
- **Pre-registered gates and kill conditions,** written before running: the synthetic
  generator makes the premise *falsifiable* in week one. If importance weighting had matched
  Heckman, the whole idea was wrong and we'd have said so.
- **Every number is machine-generated** into the papers via a script; a "regenerate and diff"
  check must pass byte-for-byte at submission — no hand-typed numbers, no drift.
- **≥8 seeds on headlines, paired Wilcoxon tests,** honest error bars, and the small-n test
  floor stated explicitly.
- **Two of the three valid outcomes shipped:** a positive-with-caveat (Paper A) and a
  null-with-mechanism (Paper B). Knowing when a fix *isn't* needed is a contribution.
- **Deliberately small compute:** one RTX 3080, ~6 GPU-hours total. The point is the idea and
  the rigor, not scale.

## 2.7 Anticipated hard questions (rehearse these answers)

- **"Why not just reweight / do covariate-shift correction?"** → Covariate shift assumes
  ignorability given observables — it corrects the x-marginal. Our whole point is selection on
  *un*observables, where E[y|x,observed] ≠ E[y|x]; reweighting can't touch that conditional.
  Oracle IW *worsened* coverage (43%) in our test. (This is 2.1 — know it cold.)

- **"The bivariate-normal assumption is unrealistic."** → Correct, and that's exactly why we
  lead with the instrument caveat and show the no-instrument degradation curve. With a valid
  exclusion restriction the estimate is robust; without one it leans on the parametric form
  and we *quantify* the fragility instead of hiding it. On real data we switched the headline
  metric to region-ECE precisely because the Gaussian assumption isn't exact.

- **"Paper B is a null — why is that interesting?"** → A calibrated null is a real result:
  "successive halving is safe at realistic noise, and here's the precise noise threshold and
  mechanism where it stops being safe." Practitioners deserve to know when *not* to add
  machinery. And the mechanism (survival selects on noise) is a clean, previously-unstated
  observation about a method everyone uses.

- **"What's the inverse Mills ratio, intuitively?"** → The average value of the outcome's
  noise term among the units that survived selection — the mean of a truncated normal. It's
  literally "how much did surviving bias me upward," as an estimable function.

- **"How do you know your implementation is correct?"** → It reproduces the published Stata
  reference on the RAND HIE dataset to 5×10⁻⁷ and matches statsmodels' probit to 1e-8. That
  gate was pre-registered and had to pass before any experiment.

- **"Does this scale to real frontier models?"** → Untested at that scale — deliberately. The
  contribution is the mechanism, the honest identification story, and the demonstration that
  the standard fix provably fails. The natural next step (2.8) is a real selected pipeline
  with a genuine instrument.

- **"What's the single most important result?"** → Oracle importance weighting getting *43%*
  coverage where Heckman gets *89%*. It proves the failure is structural, not a tuning issue —
  perfect propensities don't save you.

## 2.8 What I'd do next (shows research taste)

- Apply it where a **real instrument exists**: active-learning with a logged exploration ε, or
  multi-annotator data where annotator-assignment randomness is the exclusion restriction.
- **Reward-model calibration** under preference-selection — does a Heckman-style correction
  change RLHF outcomes in regions the policy under-explores?
- Replace the bivariate-normal coupling with a **learned copula / heteroscedastic** error
  model to relax the fragile assumption while keeping the identification logic.
- A **selective-reporting audit** of a public eval suite *with* a plausible reporting
  instrument (venue page limits, dataset release dates) to actually de-bias a leaderboard.

**If you remember nothing else:** the two equations (2.1), the one-liner about why
reweighting can't reach the y|x bias, the 43%-vs-89% number, and the instrument caveat.
Everything else hangs off those four things.
