# PLAN — Spectral Domain Extension: Finite-Size Scaling of Learned Transfer Operators

Working plan per [info.txt](info.txt). Gates, thresholds, and parameters below were
pre-registered BEFORE running E0. Every deviation from the brief or from this plan is
recorded in "Deviations & decisions", dated.

## Status board — PROJECT COMPLETE (honest null: GATE S passes, K2 fires)

- [x] Package skeleton (src/specext), vendored ETDRK4 integrator, streaming estimators
- [x] tests green (15): integrator regression vs ornstein-dist; EDMD vs analytic
      linear sectors; Welch autocorr; scaling round-trip; conv-Koopman equivariance
- [x] E0 GATE S **PASS**: 1/L flow, holdout L=176 gamma 8.2%, S 2.1%; smooth-or-
      converged 6/7 (gamma), 7/7 (S); EDMD acf median R^2 0.94-0.99
- [x] E1 K3 **fallback invoked**: deep conv-Koopman reproduces the invariant
      density (S ~8%) but NOT per-mode resonances (omega 328% off, generate-and-
      reestimate) -> flow built on EDMD (a learned data-driven operator)
- [x] E2 flow **works**: L=1408 (64x) from zero large-L data — gamma 9.8%, S 5.1%,
      C(r) 6.6%, tau 6.1%; new-mode band: statics extrapolate (1.8%), dynamics
      do not (395%)
- [x] E3 **K2 FIRES** (pre-registered honest null): interp-88 beats the flow on
      0/5 headline metrics (gamma 3.4% vs 9.8%); strict tiling fails C(r) via
      spurious periodicity; oracle compute multiple 8x
- [x] E4 boundary: convergence to limit hits 4.3% (gamma) by L=88 = the mechanism;
      ladder flat past 88; odd-parity bulk transfers to 7.6% (healing 6.5 units);
      energy/microscale band split
- [x] Figures (6) + gen_paper_numbers (81 macros) + verify_regen **byte-identical**
- [x] paper/main.tex compiled clean (9pp); "What we do not claim" present
- [x] Pre-submission arXiv recency re-sweep logged below (mechanism unoccupied)
- [x] README + laymen.md + CITATION.cff + LICENSE + requirements.txt

- [x] E5 Nikolaevskiy (second system, author-requested main-track attempt):
      honest **NEGATIVE** — interp-88 beats the flow here too (gamma 16.3% vs
      21.4% at L=704), because S stays extensive and the low-k drift is
      non-monotone. Integrator tested (4 tests). The 1/L flow has no advantage
      window over interpolation in either regime.

**Atlas link-back:** card closes **negative-with-mechanism** (GREEN-cautious
appeal was honest: the characterization E0 is positive and novel; the domain-
extension MECHANISM does not pay off — interpolating the largest affordable box
beats the 1/L FSS flow in BOTH a fast-converging system (KS, flow redundant) and
a slow-converging one (Nikolaevskiy, flow overshoots). Report for ingestion +
honest stamping: executed = characterization-positive, domain-extension-flow-
has-no-advantage-window (two systems), with a measurable drift diagnostic and the
open question (does a clean-slow-convergence system exist where the flow wins?). DOI to be filled
at submission.

## Fixed numerical conventions (pre-registered)

- KS: u_t = -u u_x - u_xx - u_xxxx, periodic on [0, L). ETDRK4 (Kassam & Trefethen
  2005), rfft pseudospectral, 2/3-rule dealiasing — vendored from
  E:/GitHub/Research/ornstein-dist src/ornstein/ks.py with attribution.
- Grid: dx = 22/64 = 0.34375 for ALL sizes (N = 64·L/22). dt = 0.25. Sampling
  interval dt_s = 0.5 (2 steps/sample). Transient discarded: 500 time units.
- Sizes: TRAIN L in {22, 44, 66, 88}; HOLDOUT/validation L = 176; HEADLINE target
  L = 1408 = 64 x 22; E4 ladder adds 352, 704 (and 2816 stretch, 1 seed, if cheap).
  All sizes are multiples of 22, so k-multiples of 2*pi/22 exist at every size
  (common grid for pointwise flow fits).
- Trajectory lengths (time units, after transient): 60_000 for L<=88; 100_000 for
  176/352/704; 200_000 for 1408; 100_000 for 2816. Seeds: {0,1,2} everywhere
  (2816: seed 0 only).
- Mode retention: 0 < k <= 3.0 (k = 2*pi*m/L). k=0 excluded (conserved mean, set 0).
- Power conventions: P_m = 2*|u_hat_m/N|^2 (one-sided), Var(u) = sum_m P_m.
  Spectral DENSITY S(k) = P_m / (2*pi/L) — the thermodynamic-limit object.
  C(r) = sum_m P_m cos(k_m r).
- Per-sector resonances: for each retained k, Hankel-EDMD on the complex mode series
  u_hat_m(t)/N: dictionary = 8 delays at stride s samples, one-step map advances by
  s samples, K = G^+ A from streaming Gram matrices; strides s in {1, 8, 64, 512}
  all accumulated; the analysis selects the stride whose window 7*s*dt_s is closest
  to 3*tau_e(k) (tau_e = first time |rho_k(t)| <= 1/e from Welch autocorrelation).
  Leading resonance = eigenvalue with max spectral weight |a_j| among |mu| <= 1.005;
  lambda(k) = log(mu)/(s*dt_s); gamma = -Re lambda, omega = |Im lambda|.
- Welch temporal PSD: Hann, 4096-sample blocks (2048 tu), 50% overlap, all modes;
  additional 65536-sample blocks for k < 0.15 (slow tail). Autocorrelation rho_k(t)
  by inverse FFT of PSD.
- Cross-estimator check inside E0: gamma from EDMD vs gamma from autocorrelation
  e-folding; report agreement (no gate, robustness evidence).

## Gates (pre-registered)

- GATE S (E0). On the common grid K_c = {2*pi*m/22 : m = 1..7} (k <= 2.2), with
  seed-mean gamma(k;L) and density S(k;L) for L in {22,44,66,88}:
  (S1) per k: EITHER the 1/L fit y = y_inf + c1/L has R^2 >= 0.7, OR the total
       variation across sizes is <= 2x the seed-level SE (already-converged case —
       a constant flow is smooth too). Must hold for >= 5 of 7 k-points, for both
       gamma and S.
  (S2) holdout: fitting on {22,44,66,88} and predicting L=176, the median relative
       error over K_c is <= 10% for gamma and <= 10% for S.
  PASS = S1 and S2. FAIL -> K1: characterization paper.
  Functional form: fit both y_inf + c1/L and y_inf + c2/L^2; model chosen by AICc,
  reported per quantity; the paper states plainly which form the data supports.
- E1 gate (K3). At L=22, learned per-sector leading eigenvalues vs EDMD: median
  over k <= 2.2 of |lambda_learn - lambda_EDMD|/|lambda_EDMD| <= 15%; model-generated
  spectral density: median |log10(S_model/S_true)| <= 0.10 over k in [0.1, 2.2];
  C(r) relative L2 error <= 10%. FAIL -> fix once, else fall back to EDMD-based flow
  (allowed by brief).
- E2 headline metrics at L=176 and L=1408 (each predictor):
  in-range band k in [2*pi/88, 2.2]:
  - med rel err gamma(k); med abs err omega(k) (reported / typical omega);
  - S(k): median |log10 ratio|;
  - C(r): relative L2 over r in [0, 44];
  - tau_e(k): med rel err;
  - slow-mode subspace: principal angles, top-16 slowest predicted vs top-16 POD/DMD.
  new-mode band k < 2*pi/88 at L=1408 reported SEPARATELY (extrapolation in k,
  E4 territory — not part of the headline claim).
- K2 test (E3): the flow "adds value" iff it beats the best no-flow null
  (strict tiling, interp-22, interp-88, neural zero-shot) on the majority of
  headline metrics at L=1408 by more than the seed-level spread. If not, K2 fires
  and the paper reports the mechanism-null honestly (where the flow matters, if
  anywhere — e.g. only below L~44).

## Method skeleton

- EDMD-based flow ("fitted FSS flow"): pointwise 1/L fits of (gamma, omega, S) on
  the common grid + smoothing splines in k of (y_inf(k), c1(k)) fitted over ALL
  modes of all four training sizes (each mode contributes at its own (k, 1/L));
  evaluate on the refined k-grid of the target size.
- Learned flow ("learned operator FSS flow"): translation-equivariant Koopman
  autoencoder — circular-conv encoder phi (1->64->64->M=16 channels, GELU, kernel 9),
  LINEAR circular-conv propagator with size-conditioned kernels W(ell) = W0 + ell*W1,
  ell = 22/L (kernel width 33, bias-free), conv decoder. Trained jointly on
  L in {22,44,66,88}; loss = reconstruction + m-step latent linear consistency +
  m-step decoded prediction (m <= 8). Spectrum: eigenvalues of the propagator
  frequency response K_hat(kappa; ell), kappa = k*dx, per sector; noise covariance
  per sector fitted from one-step residuals -> discrete Lyapunov -> stationary
  latent covariance -> decoded Monte-Carlo statistics. Extrapolation = evaluate at
  ell = 22/1408 on the refined kappa grid (finite kernels give an analytic frequency
  response at any kappa).
- Zero-shot null: the same architecture trained at L=22 only (no ell-conditioning),
  evaluated at large L — the locality-route analog of arXiv 2606.14597 at our scale
  (plus cite-compare of the original, per house rule).
- Statistics reconstruction from spectra (used by EDMD-flow and nulls):
  S(k) -> C(r); C_k(t) = S(k) e^{-gamma(k)|t|} cos(omega(k) t) (leading-resonance
  approximation, declared in the paper); tau_e from it; slow modes = Fourier modes
  ranked by predicted gamma.

## Deviations & decisions (dated)

- 2026-07-05: Stride-selection rule amended BEFORE any KS run: the analytic linear
  test caught frequency aliasing (omega * s * dt_s > pi puts log(mu) on the wrong
  branch). Rule now: admissible strides satisfy omega_peak * s * dt_s <= 0.8*pi
  (omega_peak from the Welch PSD); among admissible, window closest to 3*tau_e;
  fallback = smallest stride. Also: EDMD-implied autocorrelation is reconstructed
  by stable matrix powers (G_hat K^n)_00, and leading-eigenvalue selection weights
  contributions by |mu|^{d/2} (the delay-stack's near-defective mu~0 cluster makes
  raw eigenvector weights pathological).

- 2026-07-05: Added L=66 to the training ladder (brief said 22-88; 66=3*22 keeps the
  common k-grid and gives 4 points per pointwise 1/L fit instead of 3 — pure win on
  the 3080, minutes of compute).
- 2026-07-05: "Boundary-condition sensitivity" (E4) implemented as odd-parity
  (Dirichlet-type u=u_xx=0) KS via odd extension on a doubled periodic domain with
  per-step parity projection — spectrally exact, reuses the validated integrator.
- 2026-07-05: Per-sector dictionaries are linear delay stacks (no nonlinear
  observables). Declared limitation; cross-checked against autocorrelation decay.
- 2026-07-05: The k=0 sector (conserved mean) is excluded everywhere; initial data
  are mean-zero.

- 2026-07-05 (post-E0, pre-E3): E0 shows gamma corrections are LARGE below L~66
  and mostly saturated by L=88, so interp-88 will be the toughest null (the K2
  test will adjudicate). Added, before running E3, an "aggressive base" variant:
  fitted flow trained on {22, 33, 44} only (L=33 measured additionally; enters
  only the smooth all-modes fit, no common-grid requirement) compared against
  interp-44 at 176/1408 — this isolates whether the flow can recover the
  thermodynamic-limit spectrum from strongly finite-size-affected sizes, which is
  where a flow is needed at all. Pre-registered comparison: same headline metrics;
  flow-vs-interp-44 wins counted the same way as the main K2 test.

- 2026-07-05 (training recipe, before any model consumed by a gate): first timing
  run showed ~15.5 min/model with 20k steps x batch 32 (GPU shared with another
  job). Recipe set to 12k steps x batch 64 (MORE window samples than before at
  ~35% less wall) uniformly for all models; per-size models are trained at L=22
  only, since no experiment consumes per-size models at 44/66/88 (the E2 flow
  model trains jointly from scratch; oracle models are separate). The one model
  trained under the old recipe was deleted and retrained — all models share one
  recipe.

- 2026-07-05 (E1 K3 gate: FAIL, one fix spent, K3 FALLBACK INVOKED). The learned
  conv-Koopman operator reproduces the INVARIANT SPECTRAL DENSITY at L=22
  (generative check: S med |log10| 0.023-0.057, i.e. ~3-14%; C(r) ~12%) but its
  per-sector LEADING RESONANCES do not match EDMD. This was tested four ways and
  the failure is robust: (a) io-weight eigenvalue selection (med rel err ~2.5);
  (b) operator-implied stationary autocovariance c(n)=D K^n Sigma D^H with
  data-free Sigma (~2.0); (c) effective-decoder regression D_eff on stationary
  samples (~2.0); (d) GENERATE-AND-REESTIMATE — generate a trajectory from the
  learned stochastic operator (spectral radius clipped to the unit disk; several
  sectors are spuriously |mu|>1) and run the IDENTICAL EDMD estimator used on real
  data (~1.3-2.4). gamma is order-correct but omega (per-mode oscillation
  frequency) is systematically wrong: the reconstruction+prediction loss is
  variance-dominated, so the operator matches the measure without pinning each
  mode's resonant phase. This is an honest, interesting finding about what
  conv-Koopman autoencoders learn, and it is precisely the K3 condition.
  DECISION (pre-registered in info.txt K3): the finite-size-scaling flow — the
  actual novel claim — is built on EDMD/data-driven-Koopman spectra (which GATE S
  already validated). EDMD is itself a learned (data-driven) transfer operator, so
  "domain-size extrapolation of learned operator spectra" (N2) stands; the deep
  autoencoder was the optional embellishment and K3 says the claim survives
  without it ("a numerics paper — allowed"). The deep operator is reported in E1
  as the measure-fidelity/resonance-infidelity result that motivates using EDMD.
  Consequences: E2/E3/E4 headline = the EDMD fitted flow only; the neural
  "learned_flow" is dropped from the spectral-flow claim (kept only as the E1
  ablation). Horizon raised 8->16 for the generative check's benefit; not retried
  as a gate fix (the failure is an order of magnitude from the 15% bar and is
  structural, not a tuning issue).

- 2026-07-05 (E3 K2 verdict: K2 FIRES — pre-registered honest-null outcome). The
  finite-size flow WORKS (E2 seed-mean/GATE-S point estimate: gamma 9.8%, S 12.5%
  (median density ratio), C(r) 9.6%, tau 6.1% at L=1408=64x,
  from zero large-L data) but does NOT beat the strongest no-flow null, interp-88
  (interpolate the L=88 spectrum in k: gamma 2.4%, S 0.2%, C(r) 1.9%, tau 2.6%):
  the flow loses on 0/5 headline metrics. Mechanism: KS bulk spectra converge so
  fast that by L=88 (4x base) they are within a few percent of the L=1408 values,
  so using the largest affordable small box directly beats extrapolating a 1/L
  flow (which over-corrects past the already-converged L=88). This is exactly
  info.txt K2 ("naive tiling / small-L statistics match the flow -> the mechanism
  adds nothing over translation invariance; report honestly, and note where the
  null MUST fail"). The paper is reframed accordingly: a CHARACTERIZATION of the
  finite-size behaviour of data-driven transfer-operator spectra (E0, positive and
  novel) + an HONEST NULL for domain extension (E3), with E4 mapping where each
  route breaks and a proposal (not claim) of where the flow would be necessary
  (long correlation length relative to the affordable box; long-range coupling;
  size-dependent instabilities). Strict tiling additionally fails on C(r)
  (rel L2 137%) via spurious L=22 periodicity — a second, weaker null. Oracle
  (full L=1408 EDMD) costs 8x our small-L route; interp-88 costs the same as ours.
  This is the atlas "negative-with-mechanism" close, and per house rule it is the
  null control that makes it a paper.

- 2026-07-05 (E5 added — main-track upgrade, at author's request). To turn the KS
  honest-null into a method result, add a SECOND system where the largest
  affordable box is NOT converged, so interpolation must fail and the flow wins:
  the Nikolaevskiy equation (marginal k=0 mode -> soft-mode turbulence, long
  correlation length). Same ETDRK4 + estimators + flow (system-agnostic); only the
  linear symbol changes: sigma(k)=k^2[r-(1-k^2)^2], mean-conserving Burgers
  nonlinearity. Tuning (2 seeds): the finite-size effect grows monotonically as
  r -> onset. r=0.2 is KS-like (converged, resid 88-vs-176 ~3.7%); r=0.05 is
  strongly unconverged (gamma flow 44->88 = 43%, resid 88-vs-176 = 16%, r2=0.99)
  BUT L=22 dies (rms=0, below the turbulence threshold in a small box). CHOSEN:
  r=0.1 with the SAME ladder {22,44,66,88} as KS (all robustly turbulent, rms
  0.55-0.61, r2~1.0) and a strong unconverged signal (gamma flow 44->88 = 25%,
  resid 88-vs-176 = 6.7%). Target L=704 (32x the base; Nikolaevskiy's slow soft
  modes make 64x costlier, and 704 is already 8x beyond the L=88 training ceiling
  -- enough to make interp-88 fail decisively). Criterion the paper reports:
  interp of the largest box suffices iff the correlation length xi << that box;
  KS (xi ~ 10 << 88) -> null wins; Nikolaevskiy r=0.1 (xi >~ 88) -> flow wins.
- 2026-07-05 (E5 OUTCOME: honest NEGATIVE, hypothesis refuted; the xi framing was
  WRONG). Ran the full Nikolaevskiy ladder {44,66,88}->176,704. L=22 excluded: it
  is TURBULENT at r=0.1 (rms~0.6, r2~1.0) but its per-sector spectrum is a small-box
  finite-size OUTLIER (few unstable modes fit), and excluding it is GENEROUS to the
  flow -- including L=22 raises the flow's target gamma error (30.2% vs 21.4%), so
  it never props up the negative. (The "rms->0 death" is the r=0.05 case above, a
  DIFFERENT r -- not r=0.1.) Findings that overturned the
  plan: (1) the correlation length is SHORT for BOTH systems (integral xi ~ 4.7 KS,
  5.6 Nik; C(r)->0 by r~40 for both) -- xi is NOT the discriminator; the earlier
  "xi~L" reading was an artifact of a buggy last-crossing metric that caught the
  periodic revival C(r)=C(L-r). (2) Nikolaevskiy's spectrum DOES converge slowly
  (decay-rate drift 35% at L=88 vs KS 4.3%), driven by the marginal k=0 mode, not
  a long xi -- so interp-88 IS genuinely degraded (gamma 16.3% at L=704 vs KS
  3.3%). (3) BUT the 1/L flow is WORSE STILL (21.4%): the spectral density stays
  extensive (converges by L=44) so interp is near-exact on S, and the low-k
  decay-rate drift is NON-MONOTONE across sizes, so the 1/L fit overshoots on an
  8x extrapolation. Verdict: flow beats the best no-flow null on 0/5 metrics for
  Nikolaevskiy too. CONCLUSION (author-approved, "ship honest two-system
  negative"): the 1/L FSS flow has NO advantage window over wavenumber
  interpolation for these 1-D operator spectra -- redundant where the spectrum
  converges (KS), overshooting where it does not (Nikolaevskiy). The operative
  diagnostic is the measurable small-domain spectral DRIFT (not xi). Paper reframed
  to the honest two-system negative + drift diagnostic + mechanism; title changed
  to "Interpolation Beats Finite-Size Scaling ...". The one regime that could still
  favor the flow -- slow BUT clean/monotone extensive convergence -- was not found
  in either system and is flagged as the open question.

## Recency re-sweep log

- 2026-07-05: queries "finite-size scaling learned dynamics operator domain
  extension neural 2026", "domain extension neural operator", "renormalization
  group finite-size scaling Ruelle-Pollicott Koopman transfer operator KS",
  newest-first. Findings: (1) the named neural domain-extension baseline remains
  arXiv:2606.14597 (de Villeroche et al., Jun 2026, attention-locality route) --
  cited and cite-compared; domain-decomposition operator learning 2504.00510
  cited. (2) RG applied to KS exists but targets a DIFFERENT object: dynamic RG of
  the NOISY KS field gives KPZ-class long-wavelength scaling EXPONENTS
  (Ueno 2005; and a Jun-2026 functional-RG inviscid-scaling paper 2605.23364) --
  not the finite-size dependence of the transfer-operator SPECTRUM. Added Ueno
  2005 to related work to distinguish. (3) No hit does finite-size scaling of
  learned RP/Koopman transfer-operator spectra for domain extension: the mechanism
  is unoccupied. The claim (as narrowed: characterization + honest null) stands;
  no repositioning needed.
