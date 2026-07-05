# Spectral Domain Extension: Finite-Size Scaling of Learned Transfer Operators

Learn a neural transfer/Koopman operator for 1-D Kuramoto–Sivashinsky on SMALL
domains, extract its leading Ruelle–Pollicott resonances per translation sector,
fit the finite-size dependence of the spectrum across a few small sizes
(L = 22–88), and predict the statistics of a 64x larger domain (L = 1408) with
ZERO large-domain data. Predicted at long horizon are **statistics and slow
modes, not trajectories**: correlation functions, power spectra, decay-of-
correlation rates, slow-mode subspaces.

Everything runs on one RTX 3080 (10 GB).

## Layout

- `src/specext/` — vendored ETDRK4 KS integrator (from
  [ornstein-dist](../ornstein-dist), with attribution), streaming per-sector
  Hankel-EDMD + Welch estimators, finite-size scaling fits, translation-
  equivariant conv Koopman autoencoder with size-conditioned propagator
  `W(ell) = W0 + ell*W1`, tiling nulls, statistics/metrics.
- `experiments/exp0..exp4` — the gated experiment ladder (see `PLAN.md` and
  `info.txt`): E0 ground-truth scaling (GATE S), E1 learned operator at small L
  (K3), E2 the scaling flow + 64x validation, E3 nulls/oracles (K2), E4 honest
  boundary.
- `results/` — committed JSON outputs (every paper number derives from these).
- `paper/` — LaTeX; `gen_paper_numbers.py` regenerates `numbers.tex`;
  `verify_regen.py` checks byte-identity (house rule 1).

## Reproduce

```bash
pip install -e .
python -m pytest tests/ -q                       # integrator + estimator tests
python experiments/exp0_scaling_study.py         # ~5 min sims + GATE S
python experiments/exp1_learned_operator.py      # trains 12 small models (GPU)
python experiments/exp2_scaling_flow.py          # flow + L=1408 validation
python experiments/exp3_baselines.py             # nulls, oracles, K2 verdict
python experiments/exp4_boundary.py              # boundary probes
python experiments/make_figures.py
python paper/gen_paper_numbers.py && python paper/verify_regen.py
```

Simulation caches live in `runs/` and training data in `data/` (both
gitignored); `results/*.json` are committed.
