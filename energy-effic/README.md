# eventrice — Rice's Formula for Event-Driven Networks

Paper #4 of the Kac-Rice program (Gunner Levi Howe). Delta-network events are
level crossings of activation traces; this repo predicts their rate from
activation statistics (Rice / empirical crossing densities), allocates
per-layer delta thresholds analytically by inverting the predicted rate
curves, and tests a differentiable one-sided crossing budget as a training
objective for temporal sparsity.

**Headline results (all machine-generated into the paper):**
- **E0 / GATE V** — the temporal segment estimator and Rice predictions match
  closed forms to ≤1.9% (Rice on known spectra), ≤1.0% (discrete-time
  Gaussian on sampled OU), ≤0.2% (multisine).
- **E1 / GATE P** — an empirical calibration-split crossing profile predicts
  held-out per-channel crossing rates to 1.8–3.1% median error on a GRU
  keyword spotter (Speech Commands v2), a row-sequential-MNIST GRU, and an
  enwik8 char-transformer. Rice's Gaussian formula holds where activations
  are near-Gaussian and fails predictably with kurtosis (two-sided); an iid
  baseline is 2–17× off.
- **E2** — analytic threshold allocation matches a 48-configuration random
  search on the accuracy–events Pareto front at ~2% of its tuning cost.
- **E3 / E4 (honest negative)** — fine-tuning against the crossing budget
  makes traces comply but does **not** beat post-hoc thresholding of the
  unmodified network at matched events; neither do L1-on-deltas, rate
  regularization, or plain fine-tuning. Mechanism: send-on-delta event rates
  are approximately scale-free in θ/σ_δ, so smoothness training rescales
  dynamics without restructuring them.

## Layout

```
src/eventrice/     estimator.py (vendored + temporal), rice.py, delta.py
                   (faithful Neil et al. cells), budget.py, energy.py,
                   data.py, train.py
experiments/       exp0..exp4, exp3b, train_base, bench_timing, make_figures
tests/             test_correctness.py  (14 tests)
results/           committed result JSONs; the paper regenerates from these
paper/             main.tex, references.bib, numbers.tex (auto), gen_paper_numbers.py
PLAN.md            working plan + kill-check log + deviations
```

## Reproduce

```bash
python -m pytest tests/ -q
python experiments/exp0_validation.py
python experiments/train_base.py --task all
python experiments/train_base.py --task sc2 --seeds 3,4,5,6,7   # 8 seeds total
python experiments/exp1_prediction.py --task all
python experiments/exp2_allocation.py
python experiments/exp3_budget_training.py
python experiments/exp3b_absolute_theta.py
python experiments/exp4_controls.py
python experiments/bench_timing.py
python experiments/make_figures.py         # -> paper/figs, paper/tables
python paper/gen_paper_numbers.py          # -> paper/numbers.tex
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Environment: Python 3.13, PyTorch 2.7.1+cu118, one RTX 3080 (10 GB), Windows.
Speech Commands v2 and MNIST download automatically; enwik8 is read from the
sibling SemRF cache. Every figure, table, and prose number in the paper is
regenerated from `results/*.json`.

## Scope / non-claims

Event counts and Horowitz-table **modeled** energy, never measured watts. No
spiking/spike-coding claims. The training negative is reported for the tested
scale (GRUs to 128 units, 2-layer transformers, three tasks) and does not
claim to generalize to from-scratch training with event costs in the loss.
