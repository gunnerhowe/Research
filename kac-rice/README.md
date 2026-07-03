# kac-rice — Level-crossing density as a spatial-domain high-frequency loss for INRs

Implementation and full experimental study of the idea in
[GREEN_SPEC_kacrice_spectral.md](GREEN_SPEC_kacrice_spectral.md): use the **Kac–Rice /
Rice level-crossing density** of an implicit neural representation's output field as a
**differentiable, FFT-free, mesh-free** auxiliary loss against spectral bias.

**Paper:** [paper/main.pdf](paper/main.pdf) (LaTeX source in `paper/`, compiled with
`tools/tectonic.exe`). **Verdict (measured honestly):** on scarce scattered supervision
every auxiliary spectral loss adds +2.3–3.0 dB over MSE-only on PE-MLP (+1.4–1.8 dB on
SIREN); Kac–Rice *ties* FFL on natural images (SSIM matching or better, PSNR behind by
0.2–0.5 dB), *beats* FFL by 0.6 dB on
statistically homogeneous texture (the regime Rice theory actually describes), is
uniquely insensitive to gradient-target quality (oracle diagnostic), and composes with
SIREN (+1.5 dB) — though FFL and Sobolev compose comparably. On dense grids auxiliary
drives are unnecessary and finite-difference gradient targets actively hurt. Raw
result JSONs in `results/paper/`.

## The estimator (co-area formula, Monte-Carlo)

For an INR `f_θ` and level `u`, over sampled coordinates `{x_i}`:

```
c(u) ≈ (1/N) Σ_i δ_ε(f_θ(x_i) − u) · |∇_x f_θ(x_i)|
```

with a Gaussian smoothed Dirac `δ_ε`. `∇_x f_θ` is free via autograd. By the co-area
formula this estimates crossings per unit length (1D) / level-set length per unit area
(2D) / level-set area per unit volume (3D) — no grid, no FFT. For stationary Gaussian
fields its expectation is the Rice formula `(1/π)√(λ₂/λ₀)·exp(−u²/2λ₀)`, monotone in the
RMS derivative — i.e. a direct high-frequency proxy. We never assume Gaussianity: the
training target is the *empirical* crossing density of the ground truth computed on the
**same batch points** (correlated MC noise partially cancels).

The loss (`kacrice.crossing.KacRiceLoss`) matches the crossing-density profile over
~16 levels placed at GT value quantiles:  `L = mean_j (c_θ(u_j) − c_gt(u_j))²`,
added to reconstruction MSE with weight β.

## Layout

```
src/kacrice/
  crossing.py    # the contribution: differentiable crossing-density estimator + loss
  losses.py      # baselines: Sobolev/H¹ gradient matching, Focal Frequency Loss (local impl)
  models.py      # SIREN, FINER (variable-periodic, detached |z|+1), PE-MLP backbones
  data.py        # signals, images, grid/non-uniform sampling, GT gradients, griddata resampling
  metrics.py     # PSNR, HF-PSNR, radial per-band spectral error
  train.py       # fitting loop with pluggable aux losses
tests/           # Rice-formula & co-area sanity checks, FFL parity vs official package
experiments/
  exp1_1d.py           # grid sanity check, 1D multisine
  exp1_2d.py           # grid sanity check, 2D image
  exp2_nonuniform.py   # THE DECISIVE ONE: scattered non-uniform samples
```

## Run

```
pip install -e .            # or just: pip install torch numpy scipy matplotlib imageio
python tests/test_correctness.py
python experiments/exp1_1d.py
python experiments/exp1_2d.py
python experiments/exp2_nonuniform.py --mode blobs --oracle
```

Outputs (curves, per-band spectral errors, reconstructions) land in `results/`.

## Experimental design honesty (from the spec, enforced in code)

- **Exp 1 (grids) is a sanity check** — Kac–Rice is *expected* to roughly tie
  FFL/Sobolev there. A tie is not a failure; a loss is.
- **Exp 2 (scattered samples) is where the win must live.** All losses receive only the
  scattered `{(x_i, y_i)}`. FFL must interpolate a grid target (smears high
  frequencies). Sobolev/Kac–Rice estimate GT gradients from the *same* interpolated
  grid — no information advantage — but Kac–Rice consumes them only through a
  distributional statistic, hypothesized to be more robust to pointwise interpolation
  error. `--oracle` adds true-gradient diagnostics to separate method potential from
  gradient-estimation noise.
- **Kill condition:** ties on grids *and* no advantage off-grid → negative result,
  report honestly.
- **Not claimed:** coarse-to-fine curricula (FreeNeRF exists), or "crossings measure
  frequency" (textbook). The claim is the differentiable-loss instantiation for INRs
  and its mesh-free advantage.
- Known nuisance hyperparameter: Dirac bandwidth ε (default `0.15·std(GT)`); budget a
  sweep (`eps_scale`).

## Baselines

| Baseline | Where | Notes |
|---|---|---|
| Focal Frequency Loss (Jiang et al., ICCV 2021) | `losses.FocalFrequencyLoss` | local impl, parity-tested against the official `focal-frequency-loss` package |
| Sobolev / H¹ gradient matching | `losses.SobolevLoss` | closest spatial-domain competitor, also mesh-free |
| SIREN (Sitzmann et al., 2020) | `models.SIREN` | architecture baseline |
| FINER (Liu et al., CVPR 2024) | `models.FINER` | `sin(ω(|z|+1)z)`, first-layer bias U(−k,k) |
| FreeNeRF | — | stretch goal, NeRF few-shot setting only |
