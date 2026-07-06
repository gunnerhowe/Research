# Literature notes — verified citations (live web sweep, 2026-07-06)

A live verification + recency sweep was run (the brief's frozen index missed OUA;
a live sweep is mandatory). Every entry below was checked against primary sources.
Corrections to the brief are flagged. **Bottom line: the (a)+(b) conjunction is
UNOCCUPIED** — no prior work uses Doob's h-transform / first-passage barrier
conditioning as a synaptic rule, and none claims a falsifiable inverted-U between
intrinsic device noise and continual-learning retention.

## The surrendered drift (cite as origin, claim no novelty)

- **OUA** — Garcia Fernandez, Ahmad, van Gerven, *Ornstein-Uhlenbeck Adaptation as
  a Mechanism for Learning in Brains and Machines*. arXiv:2410.13563 (Oct 2024);
  **also** *Entropy* 26(12):1125 (2024), DOI 10.3390/e26121125 (PMC11675197).
  Eq. 5: `dθ = λ(μ−θ)dt + Σ dW`. Names catastrophic forgetting only as FUTURE
  WORK. No Doob/first-passage/barrier. => the exact anchored-diffusion drift, as
  a stochastic-process learning rule.
- **MESU** = "Metaplasticity from Synaptic Uncertainty" — Bonnet, Cottart,
  Hirtzlin, Januel, Dalgaty, Vianello, Querlioz, *Bayesian continual learning and
  forgetting in neural networks*. Nature Communications 16 (2025),
  DOI 10.1038/s41467-025-64601-w; arXiv:2504.13569; PMC12575836. Eq. 11:
  `Δμ = −σ²∂C/∂μ + [σ²/(N σ²_prior)](μ_prior − μ)` — variance-scaled anchor pull,
  verbatim. Device read-noise used only as a Bayesian SAMPLING resource, never as
  a retention optimum. (Brief said "MESU" is the title — it is NOT; corrected.)
- **EWC** — Kirkpatrick et al., *Overcoming catastrophic forgetting in neural
  networks*. PNAS 114(13):3521–3526 (2017), DOI 10.1073/pnas.1611835114. Fisher-
  weighted quadratic anchor; deterministic; no diffusion/conditioning.

## Cite-and-distinguish (the moat)

- **Kolesnikov & Semenova**, *Internal noise in hardware deep and recurrent neural
  networks helps with learning*. arXiv:2504.13778 (Apr 2025). Internal noise ->
  SINGLE-TASK test-noise robustness (FNN/ESN). Not sequential-task retention, no
  inverted-U over cross-task retention. (Title says noise "helps" — distinguish
  carefully: helps test-noise hardness, not retention.)
- **Shaham, Chandra, Kreiman, Sompolinsky**, *Stochastic consolidation of lifelong
  memory*. Sci Rep 12:13107 (2022), DOI 10.1038/s41598-022-16407-9; PMC9339009.
  Stochastic REHEARSAL/replay (beneficial); retrieval noise separately hurts. No
  hardware, no intrinsic-device-noise optimum, no first-passage conditioning.
- **Benna & Fusi**, *Computational principles of synaptic memory consolidation*.
  Nat Neurosci 19:1697–1706 (2016), DOI 10.1038/nn.4401. Complex/cascade "beaker"
  synapses; deterministic multi-timescale flow, no noise optimum. Incumbent to beat.
- **NADO** (nearest neighbor on intrinsic-noise+neuromorphic+SDE) — Manneschi,
  Vidamour, Stenning, et al., *Noise-Aware Training of Neuromorphic Dynamic Device
  Networks*. arXiv:2401.07387; Nat Commun 16 (2025), DOI 10.1038/s41467-025-64232-1.
  Neural-SDE digital twins to train THROUGH device stochasticity, SINGLE-TASK
  temporal. No continual learning, no inverted-U, no Doob.
- **Probabilistic Metaplasticity** — Zohora, Karia, Soures, Kudithipudi,
  *Probabilistic Metaplasticity for Continual Learning with Memristors*.
  arXiv:2403.08718; Sci Rep 14 (2024), DOI 10.1038/s41598-024-78290-w. Modulates
  update PROBABILITY (coin-flip gate), treats device noise as an OBSTACLE; no
  intrinsic-noise retention optimum, no barrier conditioning.

## Doob h-transform corpus (used only in generative modeling — proves the absence)

- Du, Plainer, Brekelmans, Duan, Noé, Gomes, et al., *Doob's Lagrangian: A Sample-
  Efficient Variational Approach to Transition Path Sampling*. arXiv:2410.07974;
  ICLR 2025.
- Nguyen, Do, Kieu, Nguyen, *h-Edit: Effective and Flexible Diffusion-Based Editing
  via Doob's h-Transform*. arXiv:2503.02187; CVPR 2025.
- Deng, Chen, et al., *Reflected Schrödinger Bridge for Constrained Generative
  Modeling*. arXiv:2401.03228; UAI 2024.
- Heng, De Bortoli, Doucet, Thornton, *Simulating Diffusion Bridges with Score
  Matching*. arXiv:2111.07243.
- **Direct searches for "Doob synapse", Doob + plasticity/continual-learning,
  "barrier-conditioned" + consolidation returned ZERO. The technique is absent
  from synaptic/plasticity/CL — this absence is the moat.**

## BrainScaleS-2 (BSS-2)

- **System:** Pehle, Billaudelle, Cramer, Kaiser, Schreiber, Stradmann, Weis,
  Leibfried, Müller, Schemmel, *The BrainScaleS-2 Accelerated Neuromorphic System
  With Hybrid Plasticity*. Front. Neurosci. 16:795876 (2022),
  DOI 10.3389/fnins.2022.795876; arXiv:2201.11063.
- **Intrinsic analog noise characterization:** Weis, Spilger, Billaudelle,
  Stradmann, et al., *Inference with Artificial Neural Networks on Analog
  Neuromorphic Hardware*. arXiv:2006.13177 (2020); ITEM/IoT-Streams 2020, Springer
  CCIS 1325. Distinguishes fixed-pattern (calibration-reducible) vs temporal
  trial-to-trial variability (thermal, crosstalk, analog-storage drift). Best
  anchor for "BSS-2 has intrinsic analog noise."
- **In-the-loop training that self-corrects analog mismatch:** Cramer,
  Billaudelle, Kanya, Leibfried, Grübl, Karasenko, Pehle, Schreiber, Stradmann,
  Weis, Schemmel, Zenke, *Surrogate gradients for analog neuromorphic computing*.
  PNAS 119(4):e2109194119 (2022), DOI 10.1073/pnas.2109194119; PMC8794842;
  arXiv:2006.07239.
- **CORRECTION (UNVERIFIED):** the brief's "node-perturbation pipeline on BSS-2
  (0.90-0.95)" could NOT be verified to a specific paper by the named authors. Do
  NOT cite "node-perturbation on BSS-2." The BSS-2 port section is framed honestly
  against Pehle 2022 / Weis 2020 / Cramer 2022, and the on-silicon run is stated
  as the pre-registered remaining step (K2).

## Recency sweep queries (2025–2026, newest-first)

"noise continual learning retention"; "Doob synapse" / "Doob h-transform continual
learning"; "intrinsic noise consolidation neuromorphic"; "stochastic device
continual learning retention inverted-U"; "analog noise catastrophic forgetting
neuromorphic 2025 2026"; "first-passage conditioning plasticity". No hit occupies
(a) [Doob-as-synaptic-rule], (b) [intrinsic-noise inverted-U for retention], or the
conjunction. Nearest neighbors (NADO, Probabilistic Metaplasticity, MESU) cited and
distinguished above.
