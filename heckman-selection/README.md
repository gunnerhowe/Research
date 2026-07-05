# Selection Models for Machine Learning

A two-paper program on Heckman-type selection models for ML, sharing one
theoretical backbone and one codebase (`src/heckesel/`). Author: Gunner
Levi Howe (`gunnerlevihowe@gmail.com`), Independent Researcher.

> ML training data is routinely collected by a **selection process the
> model never sees**. The field's default fixes (importance weighting,
> covariate-shift correction, MAR imputation) assume selection is ignorable
> given observables. Econometrics solved the harder problem in 1979:
> Heckman's two-equation model corrects the outcome model for selection on
> **unobservables** — the case where importance weighting is structurally
> helpless. These two papers instantiate that machinery where ML bleeds
> from it.

- **Paper A** (`paper/`): Heckman-Corrected Epistemic Uncertainty.
  Selection on unobservables defeats importance weighting even with oracle
  propensities; a joint selection/outcome error model fixes calibration in
  selected-against regions when an instrument exists.
- **Paper B** (`paper2/`): Survivor Bias in Learning-Curve Surrogates.
  Successive halving fits surrogates on curves that survived earlier rungs;
  survival selects on noise, biasing naive extrapolation. Quantified,
  corrected, and honestly bounded (the bite is confined to noisy regimes).

## The identification caveat (read this first)

Heckman identification is robust only with an **exclusion restriction** — a
variable affecting selection but not the outcome. Without one, it rides on
the bivariate-normal functional form and is fragile. Both papers include
instrument-present **and** instrument-absent conditions and report the
degradation, not just the best case.

## Layout

```
src/heckesel/         shared core
  selection.py        probit, two-step Heckman, joint MLE (bivariate-normal)
  synth.py            controlled selection generators (MAR / unobservables)
  deep.py             NN-feature-map Heckman + ensembles (Paper A)
  uq.py               UQ baselines: ensembles, MC dropout, GP, oracle-IW
  metrics.py          calibration / coverage split by sampled-density region
  lc.py               pow3 curves, SH censoring, EB + Heckman surrogates (B)
  datasets.py         Mroz87 / RandHIE loaders (faithfulness references)
experiments/          expA_*.py, expB_*.py, make_figures.py, convert_*.py
tests/                A-E0 faithfulness gate + machinery tests
results/              committed JSONs (every paper number derives from these)
paper/  paper2/       main.tex, references.bib, numbers.tex (auto), figs/
data/                 reference datasets + LCBench/PD1 caches (see below)
```

## Reproduce

```bash
pip install -e .
python -m pytest tests/ -q                 # A-E0 gate + machinery (green)

# Paper A
python experiments/expA_e0.py              # faithfulness numbers
python experiments/expA_e1.py              # synthetic demo (rho sweep, GPU)
python experiments/expA_e2.py              # semi-real tabular MNAR
python experiments/expA_e4.py              # benchmark-panel vignette
python experiments/expA_fig1data.py

# Paper B (needs data caches, see below)
python experiments/expB_calib.py           # LCBench noise calibration
python experiments/expB_calib_pd1.py
python experiments/expB_e0.py              # survivor-bias gate
python experiments/expB_e1.py              # corrected surrogate
python experiments/expB_e2.py              # LCBench + PD1 replays
python experiments/expB_fig1data.py

# figures + numbers + regenerate-and-diff
python experiments/make_figures.py
python paper/gen_paper_numbers.py  && python paper/verify_regen.py
python paper2/gen_paper_numbers.py && python paper2/verify_regen.py
```

## Data

Small reference datasets are committed under `data/` (Mroz87, RandHIE, the
Cameron–Trivedi Stata reference output). Large caches are built by the
converters:

- **LCBench** (`data_2k_lw.zip`, figshare 21188598) →
  `python experiments/convert_lcbench.py` → `data/lcbench_cache.npz`.
- **PD1** (`pd1.tar.gz`, Google Research) →
  `python experiments/convert_pd1.py` → `data/pd1_cache.npz`.

Papers-with-Code evaluation dump (`data/pwc_results.csv`, frozen 2021
snapshot mirror) drives the A-E4 vignette.

## House discipline

Every number in both papers is machine-generated (`numbers.tex` via
`gen_paper_numbers.py`); `verify_regen.py` enforces byte-identical
regeneration at submission. ≥3 seeds (8 for headline comparisons),
mean±sd, paired Wilcoxon tests. Pre-registered gates and kill conditions
are in `PLAN.md`; deviations are logged there. Discipline inherited from
the event-driven-efficiency program (DOI 10.5281/zenodo.21205335).
