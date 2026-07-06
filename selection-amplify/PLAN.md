# PLAN — Selection-Corrected Data Amplification: Manufacturing the Censored Complement a Curated Corpus Implies

Working plan per [info.txt](info.txt). The claim split, gates, kill conditions,
numeric thresholds, the pre-registered beta grid / N / seeds, and the ONE fixed
operating point below were committed BEFORE running the gated experiments. This
file is committed BEFORE the results commit so the pre-registration is
git-verifiable (a program review flagged that "pre-registered" is not verifiable
when plan and results land in one commit — fixed here). Every deviation is
recorded, dated, in "Deviations & decisions".

## The claim split (hard constraint from the kill-test + hardening)

We claim novelty ONLY in the conjunction:
  (a) **unit-level generation of a never-observed feature region located by an
      ESTIMATED SELECTION FUNCTION** — a Doob-guided bridge whose guidance signal
      is the estimated selection propensity `1 - s_hat(x)` (vs prior work that
      reweights, or imputes labels for units whose features are observed, or
      guides generation by classifiers/rewards/conditions/self-likelihood); and
  (b) the **method-minus-misdirected-guidance gap tracks selection entropy** —
      the advantage of selection-targeting over identically-strong MISDIRECTED
      guidance rises with the corpus's `I(O;X)`.
We SURRENDER "training gain tracks selection entropy" as a theorem (the method's
OWN rise is nearly trivial: more MNAR mechanically creates more test-shift
headroom). We do NOT claim a blowout over generic augmentation. We do NOT let a
SYNTHETIC go/no-go be read as a real-corpus GO.

The honest contribution is a CHARACTERIZATION: *selection-guided generation buys
a bounded, s_hat-reliability-limited COLLAR of advantage beyond 1/s_hat
reweighting, and the method-minus-decoy gap tracks selection entropy.* Absent (a)
it is Heckman/IPW/reject-inference; absent (b) the gain is generic
guidance-induced diversity.

## Status board

- [x] Package skeleton (src/selamp: data, selection, diffusion, bridge, entropy,
      validate, downstream, stats)
- [x] Live literature verification + recency re-sweep (lit_notes.md); (a)+(b)
      UNOCCUPIED. Nearest new neighbor Self-Guided Minority Samples (2407.11555):
      self-likelihood guidance, on-manifold minority — distinguished.
- [x] tests green (28)
- [x] Operating point FIXED by a coarse pre-gate calibration (see Decisions)
- [x] E0 GATE **PASS**: s_hat recovers the selector globally AND on the
      complement (Spearman >=0.97, ECE <=0.015, plug-in I(O;X) tracks analytic r=0.998)
- [x] E1 **PASS** (mechanism): synth lands in the collar-complement,
      method(0.30)>decoy(0.17)>B2(0.13), off-manifold reject <1%
- [x] E2 GO/NO-GO **FAIL (K1+K2)**: the method-minus-DECOY slice gap does NOT
      rise with I(O;X) (Spearman -0.18, p=0.84) and does not exceed the decoy at
      beta=8 (p=0.59); method also trails B3. => honest NEGATIVE.
- [~] E3 perceptual testbed DESCOPED to future work (see Deviations)
- [x] E4 central squeeze (bounded collar) + K4 foreign control PASS (foreign
      gain 0.0) + robustness
- [x] E5 real-corpus pilot (California housing), K5 not triggered, limits only
- [x] Figures + gen_paper_numbers (102 macros) + verify_regen byte-identical
- [x] paper/main.tex; honest abstract (negative-with-characterization);
      "What we do not claim"; Limitations; Reproducibility

## Fixed numerical conventions (pre-registered)

- **Testbed-A (2D).** `two_moons` (scale 2, noise 0.12) is the PRIMARY connected-
  manifold task: the censored tail of each moon is reachable by SHORT
  extrapolation ALONG the moon (the recoverable collar). `eight_gaussians`
  (closed-form density) and `pinwheel` are E0 robustness testbeds.
- **MNAR selector.** `s_beta(x) = sigmoid(-beta * phi_std(x))`, `phi = x_1`
  standardized on a fixed 40k-sample population. beta=0 is a uniform 50% thinning
  (MCAR, I(O;X)=0); beta grows -> censors the high-phi tail (MNAR). phi is
  symmetric so E[s]≈0.5 for all beta (corpus SIZE ~ constant, SHAPE shifts).
- **Censored slice.** A FIXED spatial region `phi_std(x) > 0.5` (~top 30% by
  x_1), beta-independent so the same test points define "the slice" for every
  beta; at beta=0 they are not actually censored so slice gain -> 0.
- **Reference pool D_ref.** A separate, uncensored, UNLABELED same-domain pool
  ~ p(x) (the full-support cover). A controlled affordance NOT available on real
  corpora (E5/K5).
- **beta grid:** {0, 0.5, 1, 2, 4, 8}.  **Seeds:** 5 (0-4) for the headline
  curve; ablations/E1 use 5, E0 uses 5 x 3 testbeds.
- **Sizes:** n_pop = n_ref = n_test = 8000; augmentation budget = 1000 synth per
  class (2000 total), MATCHED across every generative method and B1/B4 budget.
- **Downstream classifier:** sklearn MLP (64,64), alpha 1e-3, max_iter 800,
  random_state=seed. Metric: accuracy on the frozen full test AND the censored
  slice; per-seed gain over the B0 floor.
- **Stats:** paired Wilcoxon signed-rank at matched seed (one-sided as noted),
  matched-pairs rank-biserial effect size, percentile bootstrap CIs, Spearman.

## The fixed operating point (calibrated once; see Decisions)

Guidance scale **gamma = 6**. The collar gates are pre-registered as QUANTILE
RULES that adapt per fit (no per-run tuning inside the sweep):
- soft density gate `tau_log` = 10th pct of gate-KDE log-density over D_ref;
- hard density veto `veto_log` = 2nd pct;
- uncertainty gate `u_max` = 90th pct of epistemic uncertainty (ensemble log-odds
  std) over D_obs;
- proximity gate `d_max` = 6 x median observed-point nearest-neighbour distance;
- LOW-t-only guidance `sigma_guide_max` = 0.6; guidance-to-score norm cap = 1.0;
  gate-KDE bandwidth 0.30; deep ensemble = 5 members.

## Gates (pre-registered)

- **E0 (estimation go/no-go).** For every testbed and beta>0: mean global
  `Spearman(s_hat, s_beta) > 0.7` AND mean complement Spearman (on `s_beta<0.5`)
  `> 0.7` (the K3b region-split fix), mean `ECE < 0.10`, and the plug-in `I(O;X)`
  from s_hat correlates with the analytic value across beta at `> 0.95`. FAIL =>
  fix estimation before spending anything downstream.
- **E1 (bridge).** Method complement-hit-rate (on-manifold AND `s_beta<0.5`) >
  decoy AND > B2 at matched budget (paired over seeds), method off-manifold
  reject-rate `< 0.15`, and guidance confined to the collar (uncertainty/proximity
  gating active, not steering the deep complement).
- **E2 (THE go/no-go curve).** GATE = left-end-zero confirmed (at beta=0 the
  method censored-slice gain is within seed noise of 0) AND the method-minus-DECOY
  censored-slice gap RISES with I(O;X) (K1) AND is >0 at high beta with p<0.05
  (K2). Pass K1 + K2 here or the project stops.
- **E3 (perceptual).** MNIST subpopulation censoring; the method-minus-decoy gap
  on the censored subpopulation must be >0 (paired over seeds) with the
  off-manifold validators satisfied. Confirms the mechanism survives structured
  data.
- **E4.** K4 foreign-structure positive control passes (method does NOT recover a
  disconnected censored mode); the central-squeeze plot shows (method - B3)
  advantage bounded to the collar; s_hat-noise degrades gain gracefully.

## Pre-registered kill conditions (commit BEFORE results)

- **K1.** The censored-region method-minus-misdirected-selector gap shows no
  significant positive trend across the beta sweep: `Spearman(gap, I(O;X))` not
  significantly >0 (one-sided p>=0.05), OR beta=0-vs-beta=8 gap fails paired
  Wilcoxon p<0.05 over 5 seeds. (The method's OWN rise is trivial; the
  discriminating prediction is the rising method-minus-CONTROL gap.)
- **K2 (CORE FALSIFIER).** At beta=8 the method does not exceed the
  permuted/misdirected-selector control on the censored slice (paired Wilcoxon
  p>=0.05 or effect within noise), OR does not exceed the diversity-matched B2.
  Then selection guidance is inert. Beating gamma=0 B2 ALONE is NOT sufficient.
- **K3.** Off-manifold garbage: synthesized complement reject-rate exceeds 0.15 on
  the INDEPENDENT density/NN validator (fit on a fresh full-population pool,
  different estimator + bandwidth than the gate). Even if downstream accuracy
  moves, do not claim success.
- **K3b (region-split identifiability).** On-complement `Spearman(s_hat,s_beta)`
  not significantly >0 while global passes, OR the bridge is steered by
  high-uncertainty s_hat (guidance mass outside the low-uncertainty collar).
- **K4.** Scope over-claim: if the method recovers the genuinely foreign
  (non-recombinable) censored satellite (foreign-region gain not ~0 and
  comparable to the collar gain), that is leakage/artifact — investigate; the
  interpolative-only scope is retracted only with extraordinary evidence.
- **K5 (real-corpus only).** Cover-blind: if the reference pool is co-censored on
  the target region (classifier confidence collapses to the class prior there),
  the real-corpus result is uninterpretable — report ONLY the synthetic result.

## Method skeleton

- **LOCATE (selection.py).** Calibrated 5-member deep ensemble separating D_obs
  (1) from D_ref (0), trained balanced so `c/(1-c) = s(x)/Z`; `s_hat = clip(Z *
  c/(1-c), 0, 1)`, Z = known obs fraction. Epistemic uncertainty = ensemble std
  of the calibrated LOG-ODDS (does not saturate as s_hat->0, unlike std of s_hat;
  grows in the deep complement — the identifiability signal). Differentiable in x.
- **GENERATE (bridge.py).** Class-conditional DDPM base (diffusion.py) trained on
  D_obs. Reverse dynamics get `gamma * g^2 grad_x log h`, `h ~ r(x_hat_0)` by
  Tweedie reconstruction (eps detached, the "~free" approximation — a biased
  surrogate, stated honestly). Reward `r = (1-s_hat) * softgate(p_hat) * masks`.
  Guards: (i) density gate in the reward; (ii) LOW-t-only guidance; (iii) HARD
  density veto; (iv) guidance-to-score norm clip <=1; (v) independent validation
  (validate.py). The sharp control is the identical bridge with s_hat precomposed
  with a 90-degree rotation (decoy="rotate") at matched gamma.
- **CHARACTERIZE (entropy.py).** `I(O;X) = Hb(E[s]) - E[Hb(s)]` (bits, headline);
  `D_comp = KL(p(x|O=0)||p(x))` (nats, complement driver). The discriminating
  quantity is the method-minus-decoy slice gap vs I(O;X), never the method's own
  curve. The provable left end: gain -> 0 as I(O;X) -> 0.

## Deviations & decisions (dated)

- 2026-07-06 (uncertainty measure, BEFORE any gated run). The first draft used
  the ensemble std of s_hat as the epistemic gate; a diagnostic showed it
  SATURATES to ~0 in the deep complement (all members confidently agree s_hat≈0),
  the wrong direction. Replaced, before calibration, with the ensemble std of the
  calibrated LOG-ODDS, which grows monotonically into the complement (core 0.076
  -> collar 0.125 -> deep 1.02 at beta=4) and correctly flags where s_hat is
  off-support extrapolation. The proximity gate independently confirms (excludes
  91% of the deep complement, 0% of the collar).
- 2026-07-06 (OPERATING POINT fixed by a coarse pre-gate calibration). Before the
  gated seeds, a coarse grid (two_moons, beta=4, seed=0) over gamma {3,6,10} x
  proximity-factor {4,6,8} was run (experiments/calibrate.py) to fix ONE
  operating point by the E1 objective. Outcome: method complement-hit-rate
  (0.29-0.31) exceeds the misdirected decoy (0.16-0.17) by +0.13 to +0.14 in ALL
  nine configs, off-manifold rejection stays 0.5-1.8%, and the method drives mean
  true selection down (~0.665 vs decoy ~0.76) — so the qualitative separation
  does NOT depend on the operating point. Chosen: gamma=6, proximity-factor=6
  (clean, mid-grid, cap rarely active). The E0/E2 gates and the beta grid are
  pre-specified; no post-hoc selection of gamma or beta. Disclosed for honesty
  (analogous to fixing a training recipe before a gated run in the program's
  prior papers).
- 2026-07-06 (E0 threshold set to the brief's stricter both-sided form). A
  2-beta pre-gate preview (selector only) showed global Spearman 0.92-0.997 and
  complement Spearman 0.98-0.999 with ECE 0.009-0.014 — the reference cover
  preserves the selection RANKING into the complement. So the E0 gate is
  pre-registered at the brief's stricter `>0.7 both globally AND on the
  complement`, not a relaxed collar-only form.
- 2026-07-06 (real-corpus rung scope). E5 is limits-characterization only and
  gated by K5; a synthetic GO is never read as a real-corpus GO. E3 (MNIST) is
  the structured-data confirmation that does not rest on the 2D density.

## Outcome & deviations (dated, AFTER results)

- 2026-07-06 (VERDICT: honest negative-with-characterization). E0 and E1 pass:
  estimation is clean and the bridge demonstrably targets the censored complement
  ~2x more than the matched decoy. But the pre-registered E2 discriminator fails
  both kill conditions: K1 (no positive trend of the method-minus-decoy slice gap
  vs I(O;X); Spearman -0.18, p=0.84) and K2 (method does not exceed the matched
  decoy at beta=8; p=0.59, effect -0.07). The method also trails 1/s_hat
  reweighting (B3) by 3.8 pts. Diagnosed: synth label purity is 0.99 (NOT a
  mislabel bug); the negative is the honest central squeeze the design
  anticipated -- the identifiability gates confine useful generation to a collar
  that coincides with where B3 already works, while the recoverable headroom sits
  in the deep complement the gates (correctly) refuse. Reported as the negative
  headline + the central-squeeze characterization, per the atlas stamp rule
  ("negative-with-characterization ... is a shippable honest result").
- 2026-07-06 (E3 DESCOPED to future work). The cheap perceptual proxy we tried
  (MNIST {3,8} run through a PCA(16) latent with the same pipeline) was
  uninformative on both ends: the thickness-censored subpopulation stays easily
  classifiable (no downstream headroom to contest, B0 slice ~0.965) and a small
  latent diffusion is flagged fully off-manifold by the independent validator
  (reject ~1.0). We do not report it as evidence; the faithful pixel-space rung
  (EDM/twisted-SMC) is stated future work. The 2D go/no-go (decisive) plus the
  real-tabular limit already establish the negative-with-characterization. The
  src modules were made dimension-general (default d=2 preserves the pre-
  registered 2D behavior; 28 tests still green) for that proxy.

## Recency re-sweep log

- 2026-07-06: live web verification + newest-first sweep ("MNAR generative
  augmentation selection bias 2025 2026", "selection-guided diffusion estimated
  propensity out-of-support", "Doob h-transform bridge selection bias censored
  generation", "reject inference generative credit scoring 2025"). Findings in
  lit_notes.md. No hit occupies (a) selection-propensity-guided unit-level
  generation of the censored complement, or (b) the method-minus-decoy gap
  tracking selection entropy. Nearest neighbors — reject-inference generative
  (1904.11376 + 2606.18479), self-guided minority diffusion (2407.11555,
  self-likelihood guidance, on-manifold), training-free Doob (2602.16198),
  bias-corrected synthesis (2510.26046) — cited and distinguished. The mechanism
  is unoccupied.
