# Lit check: the LOSSONLY finding (R-ESN, 2026-07-18)

**The claim being checked**: learned temporal features of the *logged training-loss
scalar alone* (frozen random reservoir + ridge readout) forecast capability emergence
with 1,500–3,000-step median leads, zero misses, and zero false alarms across every
held-out cell including the trap language — i.e., loss *thresholds* are nowcasts but
loss *dynamics* are not. Question: is any part of this already in the literature?

## Verdict

**No prior work combines (a) timing forecasts (not binary will-it-happen), (b) input =
the logged loss series only (no gradients, no activations, no extra instrumentation),
(c) learned temporal features, and (d) false-alarm certification against manufactured
capability-blocked negatives + adversarial trap distribution under frozen one-shot
scoring.** Every component exists separately; the combination and the certification
regime do not. Nearest neighbors, in order of proximity:

## 1. Loss-curve → grokking prediction (nearest input-class neighbor)

- **Notsawo et al. 2023, "Predicting Grokking Long Before it Happens"
  ([arXiv:2306.13253](https://arxiv.org/abs/2306.13253))** — already in our
  references.bib. Detects early loss-curve *oscillations* (Fourier spectral signature)
  that predict *whether* a run will eventually grok. Same input class as LOSSONLY
  (loss curve only). Deltas: binary outcome not timing; hand-designed spectral feature
  not learned temporal features; no lead-time calibration, no arrival intervals, no
  FA-certified negatives, no adversarial trap cell, offline early-window classification
  not an online alarm. **Must be cited as the closest prior art**; our smoothing/residual
  diagnostic (esn_diag.py) directly tests whether LOSSONLY is secretly in this family
  (reading oscillations) or reading slow curve shape.
- **Thilak et al. 2022, "The Slingshot Mechanism"** (cited already) — loss spikes /
  cyclic instabilities co-occur with grokking; qualitative precursor, no forecaster.
  Related: "Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes"
  ([arXiv:2605.06152](https://arxiv.org/html/2605.06152v3)).

## 2. Early-warning signals for grokking from richer instrumentation

- **"Early-Warning Signals of Grokking via Loss-Landscape Geometry"
  ([arXiv:2602.16967](https://arxiv.org/abs/2602.16967), Feb 2026)** — commutator
  defect (curvature from non-commuting gradient updates) "rises well before
  generalization," lead times follow a power law; causal interventions
  (amplifying non-commutativity accelerates grokking). Deltas: requires *gradient*
  information (extra instrumentation/compute), we read only the logged scalar; no
  manufactured-negative FA certification, no frozen one-shot protocol; modular
  arithmetic/SCAN/Dyck grokking, not LM capability emergence. **Cite as nearest EWS
  neighbor; note their power-law lead vs our multiplicative gap law.**
- **Clauw et al., "Information-Theoretic Progress Measures reveal Grokking is an
  Emergent Phase Transition"
  ([overview](https://www.semanticscholar.org/paper/ab03153d0393f88fe506ae18d78c38f0780f1a04))**
  — synergy/O-information/entropy measured in early epochs predict *whether* a model
  will grok. Input = activations/outputs, binary not timing.
- **Developmental interpretability / (refined) Local Learning Coefficient** — Timaeus
  line: LLC detects stage boundaries incl. induction-head formation
  ([Hoogland et al.; refined LLC, arXiv:2410.02984](https://arxiv.org/html/2410.02984v1);
  [timaeus.co](https://timaeus.co/research/2024-10-04-differentiation-and-specialization)).
  Loss-*landscape*-derived (SGLD sampling; gradients + real compute), detection is
  concurrent/retrospective staging — not a calibrated advance forecast with FA rates.
  **Cite prominently**: it is the flagship "loss-geometry sees the transition" program;
  our delta is zero-instrumentation input + certified timing.

## 3. Emergence prediction at the loss *value* level (supports our nowcast framing)

- **Du et al. 2024 (NeurIPS), "Understanding Emergent Abilities of Language Models from
  the Loss Perspective"
  ([paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/5f1eee2509599faeeb3570a887016a64-Paper-Conference.pdf))**
  — downstream ability appears when pretraining loss crosses a critical *value*,
  across scales. This is the value-threshold (nowcast-family) claim our paper already
  makes; LOSSONLY sharpens it: the *value* is a nowcast, the *dynamics* are not.
- **Snell et al. 2024, "Predicting Emergent Capabilities by Finetuning"
  ([arXiv:2411.16035](https://arxiv.org/abs/2411.16035))** — finetuning shifts the
  emergence point; across-scale lever, not within-run temporal forecasting.
- **"Emergent Capabilities Arise Randomly from Learning Sparse Attention Patterns"
  ([arXiv:2606.25010](https://arxiv.org/html/2606.25010), Jun 2026)** — acquisition
  timing is stochastic per run (larger models earlier on average); attributes abruptness
  to sparse-attention learning; does NOT attempt within-run forecasting. Useful foil:
  per-run stochasticity is exactly what a within-run monitor addresses and what
  scale-level laws cannot.

## 4. Learning-curve extrapolation (the classical family — and why it misses this)

- **Domhan et al. 2015; LC-PFN
  ([arXiv:2310.20447](https://arxiv.org/abs/2310.20447)); LC-Net
  ([Klein et al.](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/17-ICLR-LCNet.pdf))**
  — parametric/Bayesian extrapolation of (assumed smooth, saturating) curves for early
  stopping/NAS. Phase transitions violate the smooth-family assumption; none certify
  against capability-blocked negatives. Our delta: transition-specific, nonparametric
  learned features, FA-certified.

## 5. The cross-field template: EWS for tipping points

- **Bury et al. 2021 (PNAS 118:39), "Deep learning for early warning signals of tipping
  points" ([doi](https://www.pnas.org/doi/10.1073/pnas.2106140118))** — a learned
  classifier over raw time series anticipates bifurcations in systems it was not trained
  on (normal-form universality). Same *shape* of idea as a reservoir over training
  telemetry; never applied to NN training trajectories/capability emergence. Also:
  "Deep learning for tipping points: Preprocessing matters"
  ([PNAS 2022](https://www.pnas.org/doi/10.1073/pnas.2207720119)) — a caution directly
  relevant to our z-scoring choices; and classical Scheffer-line EWS (variance/AC1
  rising near transitions), cf.
  [climate EWS](https://royalsocietypublishing.org/rsif/article/20/201/20220562/90332/Universal-early-warning-signals-of-phase).
- **Reservoir computing for critical transitions in dynamical systems**
  ([Chaos 2020](https://pubs.aip.org/aip/cha/article/30/12/123126/1074588/Predicting-critical-transitions-in-multiscale))
  — RC anticipates tipping in multiscale dynamical systems. No application to training
  trajectories found.

## Framing consequences for any write-up

1. Position LOSSONLY as **"EWS-for-training-runs, certified"**: the Bury-style learned
   detector, ported to capability emergence, under our FA/trap/one-shot regime.
2. **Cite Notsawo first** and run the oscillation-vs-shape diagnostic before claiming
   the mechanism is new (if RESID preserves the leads, LOSSONLY is a learned
   generalization of their spectral signature — say so plainly).
3. The threshold-vs-dynamics sentence is the crisp citable claim: value-threshold
   nowcasts (Du et al.) vs temporal-dynamics forecasts (ours).
4. 2602.16967 and the LLC line get a "richer-instrumentation EWS" paragraph; our
   distinguishing axis is zero added instrumentation + certification, not signal
   existence.
5. Honest boundary: all of this is on our small-LM/grokking substrates; no claim past
   what R-ESNb certifies.
