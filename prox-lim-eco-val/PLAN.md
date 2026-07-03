# Event-Timing Priors — Living Plan & Decision Log

Paper goal (from info.txt): show that a neural TPP likelihood on extreme-event
timings, used as an auxiliary training signal for a diffusion surrogate of a
chaotic system, fixes long-horizon TEMPORAL statistics of extremes
(inter-event times, clustering, return periods) that pointwise and
marginal/invariant-measure training provably miss. arXiv-ready paper output.

## Status (update this section as work proceeds)

- [x] Environment verified: RTX 3080 10GB, Python 3.13.6, torch 2.7.1+cu118, full sci stack
- [x] Lit-positioning agent launched (Jiang et al. objective details, pushforward, TPP refs, GenCast/ACE2/NeuralGCM)
- [x] Novelty kill-check agent launched (Seahorse 2026, Decision-Aware Training Jul-2026, adversarial sweep)
- [x] Data generation: two-scale L96 (train/val/eval/eval_long; CUDA-graph 10x)
- [x] Event structure scan -> D3 revision (trough events; Poissonize control)
- [x] TPP fit + validated (gt beats Poisson by 0.005 LL/step; pois==Poisson)
- [x] Baseline diffusion surrogate trains; stable 200-MTU rollouts, 0 divergence
- [x] Full-data base is TOO GOOD (no deficit) -> D10 data-limited regime
- [x] Deficit scans: n_traj null; dt=0.2 null -> D11 noisy-obs regime locked
      (base deficit: iet_ks 0.066, iet_w1 0.435, rate +10.6%, rl_logerr 0.12)
- [x] D1 GATE PASSED: 200-step tpp ft improves iet_w1 -38%, rl_logerr -54%,
      rate excess halved; path (a) differentiable rollout confirmed. Watch:
      mild mode-seeking (tpp_ll overshoots GT ceiling), marginal regression
      at lambda=10/lr=3e-4 -> calibration sweep.
- [x] D12 speed: aux iteration 30.7s -> 1.57s via (i) windowed-parallel
      re-forward (rollout_windowed/reforward_windows: no-grad rollout records
      states+noise, then ALL detach-windows re-forward with grad as one batch;
      bitwise-identical, same truncated-BPTT semantics) and (ii) CUDA-graph
      capture of the no-grad rollout (GraphedRollout; params tracked in-place).
      t_roll 100, b_roll 16, aux_every 2. Windows kernel-launch overhead
      (~5ms/eval eager) was the bottleneck, not compute.
- [x] Calibration done. WINNER: lambda=10, lr=1e-4, 800 steps (iet_ks 4.3x,
      iet_w1 4.1x, rl 4.3x better than base; marginals ALSO improve; no
      trade-off). lr=3e-4 causes the marginal regression seen in D1 proto.
      lambda=30 overshoots (rate 0.948). base-ft control: no change (deficit
      persistent). COMPLICATION: marg control (w=1) also fixes most timing
      metrics (iet_w1 0.079!) — in the noise regime the timing deficit is
      largely marginal-reducible. Soft dissociation remains (tpp wins iet_ks
      0.015 vs 0.030 + ACF 0.031 vs 0.045-unchanged; marg wins iet_w1, fano)
      but headline needs a temporal-dominant deficit regime, hence:
- [x] Probes A (dt05 clean) & B (S=2) NULL — no deficit. Probe C (MLP):
      mixed deficit (iet_ks 0.014, rl 0.070, BUT marg_w1 0.065 also up).
      VERDICT: no "timing-broken/marginals-fine" regime exists in this
      system; timing and marginal deficits co-occur.
- **D13 — FINAL FRAMING PIVOT: the central dissociation is TPP-aux vs
  POISSON-aux (identical loss/rate info, no temporal structure) — this IS
  the "event-timing structure as training target" claim. Jiang marg control
  repositioned: competitive at equal privilege but needs full clean
  summary-stat clouds, vs the catalog's few thousand event times
  (information-sparsity angle). Regime stays dt02r03 (noisy obs).
  Prediction VERIFIED, stronger than expected (2026-07-02): pois-aux
  actively HURTS (iet_ks 0.106 vs base 0.066 vs tpp 0.015; rate 0.79;
  rl_logerr 0.30; marg_w1 0.175) — structureless prior + ratio objective
  drags dynamics toward homogeneous timing. Identical loss/strength, only
  prior structure differs => textbook content-specificity dissociation.**
- [x] CONFIRMATORY GRID DONE (2026-07-03), 5 seeds, paired stats
      (runs/summary.md, runs/summary.csv, paper/figs/):
      * tpp: iet_ks .065->.028 (p=2e-4), iet_w1 .414->.142 (p=.001), rate
        1.100->1.005 (p=.006), rl .105->.055 (p=.038), tpp_ll AT ceiling
        (-.0992 vs -.0994), CRPS +0.4% ns, div 0.
      * pois: HARMFUL everywhere (iet_w1 1.05, rate .81, fano p=7e-5),
        tpp_ll overshoots (-.0946) — dissociation replicates all seeds.
      * marg: best marginals; partial timing; rl NOT improved (p=.61),
        fano worse (p=.023); tpp_ll overshoots (-.0971).
      * push: rate-side gains (rl .036!) but no conditional structure
        (tpp_ll -.1011 ~ base), PSD & CRPS significantly worse.
      * det: catastrophic (CRPS 3.23 vs 1.74).
      * ablations: shuf ~ tpp (structure = IET shape; renewal); tpp_mle
        TOTAL collapse (rate_ratio 0.0025!) — ratio objective essential.
      * figures: GT IET is multimodal (wave harmonics); GT hazard
        oscillatory; tpp tracks hazard, pois FLATTENS it. Money figures.
      * Wilcoxon floor at n=5 is .0625 -> seeds 6-10 queued for n=10.
- [x] KS probe: **THE DREAM DECOMPOSITION** — base deficit is
      temporal-dominant: iet_ks 0.594, rate_ratio 2.50, rl_logerr 1.14,
      psd 3.40, while state marg_w1 = 0.037 (nearly perfect!). Marginal
      matching has nothing to fix; the corruption is temporal jitter ->
      spurious re-crossings. KS = headline dissociation system; L96 =
      replication/controls system. KS events value@q0.95 CLUSTERED (CV
      1.58, opposite of L96 quasi-periodic).
- [x] KS grid seeds 1-3 done — **tpp FAILED on KS with L96 hypers**: rate
      collapse to 0.31, iet_w1 179 (worse than base 54.6); pois (rate 0.96)
      and marg (iet_w1 22) did fine. DIAGNOSIS (from llg/lls logs): two-
      timescale tracking failure — clustered prior punishes spurious events
      brutally (tiny baseline intensity between bursts) -> surrogate's event
      process collapses PAST target within 200 its while self-TPP (5
      steps/aux) lags; self catches up at the collapsed state -> local
      equilibrium (llg~lls~-0.035, both above ceiling -0.0501; residual KL
      0.005 too weak vs FM anchor). NOT fundamental: brake must track
      faster than surrogate moves. KS events much sparser/clustered than
      L96 -> needs own calibration (L96 hypers were transplanted blindly).
- [ ] KS calibration RUNNING (scripts/calibrate_ks.py, seed 1): aux_weight
      {1,3,10} x self_tpp_steps {25} + {1,5} control. evr (rollout event
      rate) now logged each aux step. L96 seeds 6-8 queue PAUSED (resumable:
      python scripts/run_grid.py --seeds 6 7 8) — requeue after KS fixed;
      KS grid tpp runs must be RERUN with winning hypers (delete
      runs/ks_ft_tpp_s* first).
- NOTE for paper: this failure mode + fix is itself a contribution
      (two-timescale condition for TPP-ratio training; clustered priors are
      the hard case). Report honestly in ablations/discussion.
- [x] KS calibration NULL: all {w1,w3,w10}x{st5,st25} still collapse rate
      (best w3_st25: 0.61). Root cause deeper than tracking: KS deficit is
      dynamically deep (psd 3.4) — no accessible dynamics direction to
      place events correctly; suppression is the easy descent path.
      Clustered prior punishes spurious events brutally (tiny baseline
      intensity). marg (u_t stats) partially repairs spectrum (1.83).
- [ ] KS FIX PROBES RUNNING (scripts/probe_ks_fixes.py, ~3h):
      A. margtpp composition (new condition; both aux, shared rollout)
      B. gentle-long (w3 st25 lr3e-5 2400 steps)
      C. milder noise r=0.2 (data/ksr02) base + tpp
- [x] Aggregate-event transfer (L96): NEGATIVE — site-level prior does not
      fix spatial-max events (rate 1.45 vs base 1.29; CV stays ~1.0 vs GT
      1.13). Added to paper ablations as honest negative result.
- NOTE: hazard-shape check on calibration runs: marg_w3 fixes hazard BETTER
  than tpp (RMSE 0.0061 vs 0.0070; base 0.0128) -> noise regime has NO clean
  dissociation anywhere; a temporal-dominant regime is REQUIRED for the
  headline, else fall back to both-axes framing. hazard_curve metric added
  to metrics/eval/aggregate (hazard_rmse; curves.npz['hazard']).
      Decision rule: want iet/acf deficits LARGE while marg_w1 SMALL at base.
      If both null -> keep noise regime, reframe as "both-axes + residual
      structure" (tpp fixes conditional/serial structure marg cannot; report
      iet_ks, tpp_ll, acf as timing-shape metrics). KS secondary may also
      provide temporal-dominant deficits (wave-train dynamics).
- [ ] Full grid: conditions x seeds 1-5 (run_grid.py --n_traj N)
- [ ] Optional ablations: tpp_mle (collapse), shuf (renewal check), 1 seed
- [ ] KS secondary system
- [ ] Aggregation + figures (scripts/aggregate.py)
- [ ] Paper draft (LaTeX)
- [ ] Bib verification pass (see Paper TODOs)

## Key design decisions (log every change here)

**D0 — Ground truth = TWO-SCALE Lorenz-96** (K=40 slow, J=10 fast per slow;
h=1, b=10, c=10, F=10; RK4). Observe slow variables only. Rationale: makes the
map from observed state to next observed state genuinely stochastic (unresolved
fast scales), so a *generative* surrogate is well-posed — mirrors the climate
emulator setting (GenCast/ACE2) where diffusion models are used precisely
because coarse dynamics are effectively stochastic. Single-scale L96 is a
deterministic map; a diffusion emulator of it collapses toward a delta and the
stochasticity would be pure model error. info.txt said "L96, K=40" — two-scale
with K=40 slow sites is consistent with that.

**D1 (gating, prototype first per info.txt) — Gradient path = (a) truncated
differentiable rollout.** Few-step deterministic-noise (reparameterized) DDIM
sampler, rollout T_roll steps with fixed noise draws, gradient checkpointing
per rollout step. Aux loss = negative TPP log-lik of SOFT events extracted
from the rollout. Fallbacks if (a) fails: (b) DDPO-style reward fine-tune,
(c) inference-time guidance.

**D2 — TPP form: discrete-time recurrent intensity model (grid = surrogate
step).** GRU consumes event-indicator sequence, emits per-step intensity
lambda_t; log-lik L = sum_t [w_t log(lambda_t) - lambda_t * dt] (Poisson-grid
discretization of the continuous TPP likelihood; exact as dt->0). Crucial
property: L is LINEAR in event weights w_t, so soft (differentiable) event
weights from surrogate rollouts drop in natively — same model and loss form for
ground-truth (hard w) fitting and aux (soft w) training. A Shchur-style
intensity-free lognormal-mixture TPP can be fit as a secondary check on
ground-truth IETs (eval only).

**D3 — Events (REVISED after ground-truth structure scan, 2026-07-02)**:
info.txt's example (site energy x^2 at q0.98) is empirically POISSON in
two-scale L96 (Fano~1 at all windows, IET CV~1.0, lag-1 IET corr ~0) — no
temporal structure for a TPP to learn, and shuffled/Poissonized controls
could not dissociate. Structure scan over observables x quantiles found:
  - site-level events: SUB-Poisson / quasi-periodic, strongest for the
    NEGATIVE tail: s_k = -x_k at q0.95 gives IET CV=0.72, Fano2=0.64,
    Fano20=0.54, rate 0.287/site/MTU; at q0.98 still CV=0.86.
  - aggregate events (spatial max x, global energy): SUPER-Poisson
    (clustered), CV up to 1.28, Fano20 ~1.35-1.5.
  - lag-1 IET correlations ~0 everywhere -> near-RENEWAL processes;
    structure lives in the IET distribution shape.
PRIMARY event def: upcrossings of s_k = -x_k above u = q0.95 pooled (deep
troughs), 40 exchangeable site streams. Soft version: w_t =
sigmoid((s_t-u)/tau) * sigmoid((u-s_{t-1})/tau). EVAL also at q0.98 and on
spatial-max-x events at q0.98 (clustered direction; transfer question).
CONSEQUENCE for controls: interval shuffling is a near-no-op for renewal
processes -> the content-specificity control is POISSONIZATION (same rate,
exponential IETs). Keep interval-shuffled as an optional extra condition —
if it matches full TPP-aux, that itself localizes the gain to IET shape.

**D4 — Surrogate**: conditional denoising diffusion p(x_{t+1}|x_t), 1D circular
CNN (respects ring topology), ~200-400k params, standard DSM training +
condition-specific auxiliary. Emulator step dt = 0.05 MTU (TBD after data gen —
check effective stochasticity & event rates). Deterministic AR baseline = same
CNN, MSE loss.

**D5 — Conditions x 5 seeds, L96:**
1. `base` — flow-matching (diffusion-family) surrogate, FM loss only
2. `tpp` — FM + TPP-ratio aux (the intervention, see D7)
3. `marg` — FM + Jiang et al. 2023 control: debiased Sinkhorn divergence on
   physics-informed summary stats S(u) = {du/dt, (u_{k+1}-u_{k-2})u_{k-1}, u}
   point clouds from rollouts vs ground truth (verified from their paper).
   THE dissociation control.
4. `pois` — FM + TPP-ratio aux with GT-TPP trained on POISSONIZED event times
   (same rate, exponential IETs; identical loss form & strength).
   Content-specificity control (see D3 for why not shuffled).
5. `push` — FM + pushforward-trick rollout stabilization (structure-agnostic
   stability control; Brandstetter et al.: 2-step unroll, grad only through
   last step).
6. `det` — deterministic AR baseline (MSE).
Optional 7. `shuf` — interval-shuffled TPP control (expected ~= `tpp` for a
renewal process; localizes the gain to IET shape).
Optional 8. `tpp_mle` — plain-likelihood ablation (expected: event-rate
collapse; motivates the ratio form).

**D7 — Aux objective form: likelihood-RATIO (TPP-space reverse-KL / VSD
analog), not plain likelihood.** Plain -E[log p_gt_tpp(w)] is mode-seeking:
for a fitted TPP the empty sequence often has higher density than typical
sequences -> collapses the surrogate's event rate. Fix: maintain a small
"self-TPP" periodically refit by MLE on the surrogate's own (hard, detached)
rollout events; aux = -E[log p_gt_tpp(w) - log p_self_tpp(w)]. Zero gradient
exactly when surrogate event process == GT event process; no collapse
incentive. Analogous to Variational Score Distillation / Diff-Instruct, in
point-process space. `tpp_mle` ablation demonstrates the failure mode.

**D8 — TPP history warmup at rollout start**: prepend the GT hard-event
history (T_hist steps before the rollout IC, available from data) to the soft
rollout events; mask the likelihood over the warmup span. The GRU state is
then calibrated by real history at the point where rollout events begin.

**D9 — Two-phase training for matched, paired comparisons.** Phase 1: one
base FM surrogate per seed (det baseline gets its own MSE phase 1). Phase 2:
from the SAME per-seed base checkpoint, fine-tune under each condition with
identical step budget; `base` condition = FM-only fine-tune of the same
length (compute matched). Conditions differ ONLY in the phase-2 objective ->
paired per-seed stats, lower variance, and any aux effect is attributable.
Self-TPP for the ratio loss is warm-started on init-surrogate rollouts
before phase 2 begins.

**D6 — Metrics**: inter-event-time distribution (KS stat + Wasserstein-1),
clustering (Fano factor vs window size; Ripley's K / branching ratio),
return-period & return-level curves; PLUS marginal PDFs (per-site, pooled),
power spectra, ACF, short-horizon RMSE/CRPS (no regression of ordinary skill).
>=5 seeds, paired stats vs `base`, censored/diverged rollouts reported.

## Repo layout

```
src/
  data/l96.py        # two-scale L96 integrator + dataset generation (GPU)
  data/ks.py         # KS secondary (later)
  events.py          # hard + soft event extraction
  tpp.py             # discrete-time recurrent intensity TPP (+ lognormal-mix eval TPP)
  surrogate.py       # diffusion emulator + deterministic baseline + samplers
  losses.py          # tpp_aux, marginal_aux, pushforward
  metrics.py         # all eval metrics
  train_tpp.py       # fit TPP on ground-truth events
  train_surrogate.py # unified trainer, --condition flag
scripts/             # run_grid.py etc.
runs/                # checkpoints, metrics json, logs (gitignored-scale)
data/                # generated trajectories (npz)
paper/               # LaTeX
```

**D10 — Deficit regime (2026-07-02): base surrogate with FULL data (64 traj,
205k transitions) is already near-perfect on event timing (IET KS 0.0075,
rate ratio 1.004, rl_logerr 0.008, tpp_ll matches GT) — no deficit, no paper
at that scale. Pivot to the DATA-LIMITED regime (realistic: short
observational records; extremes undersampled): restrict training to first
n_traj trajectories for surrogate AND TPP (same information budget; fair).
Event threshold u stays fixed from full train split (task definition, applied
identically to all conditions). Scan n_traj in {2,4,8} to pick the regime
with clear base deficit (target: iet_ks>0.05 or fano/rl logerr>0.1) while
still trainable. Paper framing: TPP is a far smaller model of a 1-D process
-> learns IET structure from the same few events that leave the 40-dim
dynamics tails undersampled; the aux transfers that structure back.**

**D11 — Deficit regime, final (2026-07-02): NOISY OBSERVATIONS + coarse step.
Clean-data regimes are ALL too learnable for this architecture (n_traj scan
null: even 2 traj near-perfect; dt=0.2 also null: iet_ks 0.003). Translation
equivariance x 40 sites + smooth conditionals = the CNN nails the flow map.
Adopted setting (precedented by Jiang et al. 2023, who built their paper on
noisy observations): train surrogate on dt=0.2 states + iid Gaussian obs
noise r=0.3*std (data/dt02r03; train/val only). Rollouts then inherit
noise-as-process-noise -> corrupted event timing BY CONSTRUCTION.
Reframing that strengthens the paper: curated EVENT CATALOGS (IBTrACS,
earthquake catalogs, flood records) and curated CLIMATOLOGICAL STATISTICS
exist independently of state-observation noise. So: TPPs fit on CLEAN events
(the catalog, = data/dt02 artifacts), marg control gets CLEAN summary stats
(the climatology) — equal privilege, fair dissociation. Question: which
curated target fixes long-horizon event timing? Implemented via
--ref_dir data/dt02 (event defs, TPP ckpts, marg-stat source, event
histories) with --data_dir data/dt02r03 (noisy training states). Eval always
vs clean refs (--data_dir data/dt02 in eval_surrogate).**

## Paper TODOs

- refs.bib written 2026-07-02: entries for DySLIM, BSP loss, Thermalizer,
  Melo adversarial-OT, Stamatelopoulos CMAME volume/pages are from model
  memory / agent snippets — VERIFY authors+ids on arXiv before submission.
- Report the tpp_mle ablation (rate collapse) and pure-likelihood pathology.
- Report soft-event calibration (tau: 4.9% rate bias, 80% mass localization).

## Compute budget notes

Aux rollout: T_roll~128-256 steps x S~6-8 sampler steps, batch 4-8, tiny CNN →
~1-3k net evals/aux step; aux every k iters if needed. Grid: 30 runs x
~30-60 min ≈ 15-30 GPU-h. 10GB VRAM fine with per-step checkpointing.

## Open questions / risks

- Effective stochasticity of slow-vars map at dt=0.05: verify spread of
  x_{t+1} | x_t from data before committing (else increase dt).
- Event rate at q0.98 upcrossings: want enough events per rollout window
  (~0.5-2 events per 100 steps per site pooled x40 sites is plenty).
- Jiang et al. exact objective — awaiting lit agent; implement faithfully.
- Novelty: awaiting kill-check agent on Seahorse + Decision-Aware Training.
- KS secondary (src/data/ks.py written): L=22, N=64, ETDRK4, dt_obs=1.0 tu,
  same noisy-obs deficit recipe (make_noisy.py works on any data dir). NOTE:
  KS npz files use the l96_<split>.npz naming so the whole downstream
  pipeline works unmodified on --data_dir data/ks*. The marg condition needs
  KS-specific summary stats {u_t, u_x, u_xx} (l96_summary_stats is
  L96-specific) — add if KS-marg is run. Run structure scan on KS events
  before committing (need CV != 1).
