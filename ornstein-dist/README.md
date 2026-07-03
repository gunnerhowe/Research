# ornstein-dist — Ornstein d̄-distance as a dynamical-fidelity metric for chaotic surrogates

Implementation and paper for [GREEN_SPEC_ornstein_chaos.md](GREEN_SPEC_ornstein_chaos.md):
Ornstein's d̄-distance (ergodic isomorphism theory) as a practical evaluation metric that
catches surrogates which reproduce a chaotic system's *invariant measure* but not its
*dynamics* — a failure mode Wasserstein-on-measure structurally cannot see.

- **Paper (arXiv-ready): [paper/main.pdf](paper/main.pdf)** — source in `paper/`
- **Plain-English explanation (no math): [laymen.md](laymen.md)**
- Verified math + spec corrections: [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md)
- Working notes from the initial go/no-go phase: [REPORT.md](REPORT.md)

## Layout

- `src/ornstein/` — library
  - `systems.py` — Lorenz-63 (numba RK4; `speed` rescaling = same invariant measure, different clock)
  - `ks.py` — Kuramoto–Sivashinsky, ETDRK4 (Kassam–Trefethen), L=22
  - `symbolize.py` — partitions (sign, quantile, box); always built on truth, applied to all
  - `surrogates.py` — IAAFT, phase-randomized, time reversal
  - `entropy.py` — block/LZ78 entropy rates, finite-n Fano-type certified d̄ lower bounds
  - `dbar.py` — d̄_n = exact OT between empirical n-block distributions (Hamming cost),
    with same-process noise floors as the honesty device
  - `baselines.py` — W1 marginal/state/delay, PSD, ACF, Rosenstein λ1
  - `esn.py` — echo state network surrogate (reservoir + [r; r²] ridge readout)
- `tests/validate.py` — estimator validated against exactly solvable cases (15 checks)
- `experiments/`
  - `exp0_gonogo.py` — entropy-rate pre-check (the spec's one-hour go/no-go)
  - `exp1_convergence.py` — d̄_n convergence, noise floors, sample-size scaling
  - `exp2_decisive.py` — single-seed metric × surrogate matrix + partition/τ sensitivity
  - `exp3_multiseed.py` — 8-seed Lorenz matrix (paper Table 1)
  - `exp3b_lambda_check.py` — Rosenstein fit-window study (paper App. B)
  - `exp4_ks.py` — Kuramoto–Sivashinsky, symmetry-reduced (paper Table 3)
  - `exp5_esn.py` / `exp5b_esn_degraded.py` — learned surrogates (paper Table 4)
- `results/` — JSON outputs + logs (all paper numbers come from these)
- `paper/` — LaTeX source; `gen_paper_numbers.py` regenerates `numbers.tex` from
  `results/*.json` (no hand-typed numbers in the paper); `make_figures.py` builds
  `paper/figs/*.pdf`

## Reproduce

```
python -m venv .venv
.venv\Scripts\pip install numpy scipy matplotlib pot numba
.venv\Scripts\python tests\validate.py
.venv\Scripts\python experiments\exp0_gonogo.py      # ~1 min
.venv\Scripts\python experiments\exp1_convergence.py # ~3 min
.venv\Scripts\python experiments\exp2_decisive.py    # ~6 min
.venv\Scripts\python experiments\exp3_multiseed.py   # ~9 min
.venv\Scripts\python experiments\exp4_ks.py          # ~30 min
.venv\Scripts\python experiments\exp5_esn.py         # ~10 min
.venv\Scripts\python experiments\exp5b_esn_degraded.py # ~3 min
.venv\Scripts\python paper\make_figures.py
.venv\Scripts\python paper\gen_paper_numbers.py
tectonic -X compile paper\main.tex
```

All CPU; complete suite reproduces in under two hours single-threaded.
