# Interpolation Beats Finite-Size Scaling for Domain Extension of Learned Transfer-Operator Spectra

Learn a data-driven Koopman/transfer operator for a 1-D chaotic PDE on SMALL
domains, extract its leading Ruelle–Pollicott resonances per translation sector,
fit the finite-size (1/L) dependence of the spectrum, and extrapolate to a much
larger domain — predicting **statistics and slow modes, not trajectories**
(correlation functions, power spectra, decay rates), with ZERO large-domain data.

**Headline result (honest two-system negative).** The finite-size flow *works* in
absolute terms (Kuramoto–Sivashinsky, L = 1408 = 64×, to ~10% on decay rates,
~12% on spectral density) — **but it never beats the simplest baseline**:
interpolating the spectrum of the largest affordable box in wavenumber, no size
flow. Shown across two systems that bracket the finite-size convergence rate:

- **Kuramoto–Sivashinsky** (fast convergence): interp-88 (3.3%) beats the flow
  (9.8%) — the spectrum is converged by L = 88, so the flow is *redundant*.
- **Nikolaevskiy** (marginal k=0 mode → slow spectral convergence): interp is
  genuinely degraded (16.3%), but the 1/L flow is *worse still* (21.4%) — the
  spectral density stays extensive and the flow overshoots the non-monotone
  low-k drift.

The 1/L flow has no advantage window. The operative diagnostic is the measurable
**small-domain spectral drift** (not the correlation length, which is short for
both). `paper/` gives the mechanism and the one regime — clean slow convergence —
that could still reverse it, which we did not find.

Everything runs on one RTX 3080 (10 GB); the analysis is pure numerics (no GPU).
A deep conv-Koopman autoencoder (E1) is an ablation. Experiments: `exp0..exp4`
(KS), `exp5` (Nikolaevskiy).

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
python experiments/exp4_boundary.py              # KS convergence/drift, ladder, odd-parity, xi
python experiments/exp5_nikolaevskiy.py          # 2nd system: flow vs interp (the honest negative)
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
