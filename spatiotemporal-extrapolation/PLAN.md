# PLAN — Spectral Domain Extension: Finite-Size Scaling of Learned Transfer Operators

Working plan per [info.txt](info.txt). Gates, thresholds, and parameters below were
pre-registered BEFORE running E0. Every deviation from the brief or from this plan is
recorded in "Deviations & decisions", dated.

## Status board

- [ ] Package skeleton (src/specext), vendored ETDRK4 integrator, streaming estimators
- [ ] tests green (integrator regression vs ornstein-dist; EDMD vs analytic linear
      sectors; scaling-fit round-trip; conventions)
- [ ] E0 ground-truth scaling study (L = 22/44/66/88 train, 176 holdout) — GATE S
- [ ] E1 learned operator at L=22 matches EDMD spectra + statistics (K3 gate)
- [ ] E2 scaling flow fitted on L<=88, extrapolated to 176 and 1408, validated
- [ ] E3 baselines: tiling nulls, interpolated small-L nulls, neural zero-shot,
      direct large-L oracle (compute multiple), limited-data EDMD (K2 check)
- [ ] E4 honest boundary: error vs L, BC sensitivity (odd-parity), high-k band,
      new low-k modes
- [ ] Figures + gen_paper_numbers + verify_regen (house rule 1)
- [ ] paper/main.tex compiled clean; "What we do not claim" present
- [ ] Pre-submission arXiv recency re-sweep logged below
- [ ] README + laymen.md; atlas link-back note

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

## Recency re-sweep log

- (to be filled at submission; queries: "domain extension neural operator",
  "finite-size scaling learned dynamics", newest-first)
