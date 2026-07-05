# Finite-Size Scaling of Data-Driven Transfer-Operator Spectra

Learn a data-driven Koopman/transfer operator for 1-D Kuramoto–Sivashinsky on
SMALL domains, extract its leading Ruelle–Pollicott resonances per translation
sector, fit the finite-size (1/L) dependence of the spectrum across a few small
sizes (L = 22–88), and predict the statistics of a 64× larger domain (L = 1408)
with ZERO large-domain data. Predicted at long horizon are **statistics and slow
modes, not trajectories**: correlation functions, power spectra, decay-of-
correlation rates, slow-mode subspaces.

**Headline result (honest, pre-registered).** The finite-size flow *works*
(L = 1408 to ~10% on decay rates, ~12% on the spectral density), but the null
control — just interpolate the spectrum of the largest affordable small box
(L = 88) with no size flow — does *better* (~3%), because KS spectra converge to
within a few percent of the thermodynamic limit already by L ≈ 88. Kill-condition
K2 fires: the scaling machinery is unnecessary for KS. We report this, quantify
the fast convergence that drives it, and delimit where the null must fail. See
`paper/` and `PLAN.md`.

Everything runs on one RTX 3080 (10 GB); the headline flow is pure numerics (no
GPU). A deep conv-Koopman autoencoder (E1) is trained as an ablation.

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
pip install -e .                                 # or: pip install -r requirements.txt
python -m pytest tests/ -q                       # 15 unit tests

# E0 also generates every KS trajectory the later stages reuse (writes data/, runs/)
python experiments/exp0_scaling_study.py         # ~6 min sims + GATE S
python experiments/exp1_learned_operator.py      # trains 3 conv-Koopman models (GPU); K3
python experiments/exp2_scaling_flow.py --skip-sim   # EDMD flow + L=1408 validation
python experiments/exp3_baselines.py             # nulls, limited-data EDMD, K2 verdict
python experiments/exp4_boundary.py              # convergence, ladder, odd-parity, bands
python experiments/make_figures.py               # ~2 min (re-analyzes cached runs); not a hang

python paper/gen_paper_numbers.py                # emits paper/numbers.tex + tables
python paper/verify_regen.py                     # house rule 1: byte-identical regen
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The remaining large-L / short-trajectory simulations (L = 1408, 2816, the
odd-parity run, the limited-data runs) are produced on demand by exp2–exp4 if
their caches are absent. Simulation caches live in `runs/` and trajectory data in
`data/` (both gitignored); `results/*.json` are committed and are the sole source
of every number in the paper.
