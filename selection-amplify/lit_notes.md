# Literature notes + live recency re-sweep

Live web verification (2026-07-06), newest-first, per the brief's mandatory
recency sweep (the program's frozen index once missed OUA, so a live sweep is
required before pre-registration). Queries run: "MNAR generative augmentation
selection bias 2025 2026", "selection-guided diffusion estimated propensity
out-of-support generation", "Doob h-transform bridge selection bias censored
generation", "reject inference generative credit scoring 2025". The moat --
*guidance signal = estimated selection function, to synthesize whole
never-observed units in the censored complement* -- is unoccupied. The two
load-bearing distinctions (unit-level vs feature-level; selection-propensity
guidance vs reward/classifier/likelihood guidance) survive.

## Reweight / correct only (the baseline-to-beat, B3 -- NOT prior art)

- **Heckman (1979) / IPW / CRM (Swaminathan-Joachims).** Reweight the loss or
  correct inference; never fabricate censored units. B3, the central squeeze.
- **Distribution Regression with Censored Selection (2505.10814).** Relaxes
  Heckman parametrics, still corrects estimates under selection. Baseline.
- **Amortized Bayesian Inference for selection bias (2604.18319).** Embeds
  selection into a simulator but corrects the posterior; emits no censored
  samples, no bridge, no entropy relationship.

## Impute label/entries for observed-feature units (feature/label-level)

- **Deep Generative Models for Reject Inference (1904.11376).** Canonical
  Heckman-selection generative work: imputes the OUTCOME for rejected applicants
  whose FEATURES are observed. Does not generate never-observed feature-space
  units. Unit-level-vs-feature-level distinction is load-bearing; cite + draw it.
- **RMT-Net (2206.00568), SMART/GAIN (PMC12041391).** Reject-aware correction /
  feature-level imputation, not unit-level generation of the complement.
- **MNAR imputation nets -- GNR (2308.08158), Identifiable MNAR generative
  (2110.14708), Missingness Augmentation (2108.02566).** (recency sweep) All
  impute missing ENTRIES / the missingness MASK for partially-observed rows in
  the latent space; none locates a never-observed feature region by an estimated
  selection function and synthesizes whole units there. Feature-level.
- **Reject-inference 2025-26: "Illusion of Improvement" (2606.18479),
  "Confident Inlier Extrapolation" (2510.12967), graph-based RI
  (s10479-025-06621-9).** (recency sweep) Newer reject-inference; still infer
  labels for feature-observed rejects, and 2606.18479 finds reported RI gains
  are often illusory -- supports our honest, decoy-controlled framing.

## Out-of-support / low-density guided diffusion (nearest neighbor, generation side)

- **Self-Guided Generation of Minority Samples (2407.11555).** (recency sweep,
  NEAREST new neighbor) Steers a pretrained diffusion toward low-density
  "minority" regions by minimizing the model's OWN estimated feature
  likelihood. Verified via WebFetch: guidance = self-likelihood, NOT an
  estimated selection propensity; targets minority samples *on the existing
  data manifold*, not never-observed units located by a selector. Adjacent-
  distinguishable; the distinguishing axis is exactly "guidance = estimated
  selection function." Cite + distinguish.
- **Support-robustness under guidance (2605.07220), Stein guidance (2507.05482),
  low-density-region exploration (2606.13347), training-free Doob (2602.16198),
  infinite-dim Doob (2602.06621), h-Edit (2503.02187).** Steer into low-density
  / out-of-support regions but driven by classifiers / rewards / conditions /
  edits, NEVER an estimated selection propensity. Confirmed by the Doob sweep.

## Pattern-match preemption (chief presentational risk)

- **Bias-Corrected Data Synthesis for Imbalanced Learning (2510.26046).**
  Nearest-sounding "bridge + complement," but distribution-matching WITHIN the
  existing minority class with an explicit correction term -- no estimated
  selection model, no Doob bridge, no entropy relationship. Cite to preempt
  reviewer pattern-matching.

## Companion (same author, same program)

- **Heckman-Corrected Epistemic Uncertainty (howe2026, `heckman-selection/`).**
  Instantiates Heckman for deep UQ: reweighting/correction under selection on
  unobservables. It is the intellectual parent of B3 here (reweight, don't
  fabricate). This swing asks the orthogonal question: once you have s_hat, can
  you GENERATE the complement rather than only reweight the observed?

## Verdict

No hit occupies (i) unit-level generation of a never-observed feature region
located by an estimated selection function, or (ii) the method-minus-decoy gap
tracking selection entropy. Nearest neighbors (reject-inference generative;
self-guided minority diffusion; training-free Doob) are cited and distinguished.
