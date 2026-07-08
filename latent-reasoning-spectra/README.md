# Spectral Invariants of Latent Reasoning

Predicting and causally explaining anchor/branch structure in continuous chain-of-thought
(Coconut-family) reasoning, from the dynamical invariants of the latent-thought
autoregressive map c_{t+1} = F_t(c_t), validated against ProsQA's ground-truth DAG with
matched pause-token and pruned linear-chain null controls.

Pre-registration: [PLAN.md](PLAN.md) (committed before any results; kill conditions K1-K3).
Paper: `paper/main.tex` (all numbers machine-generated into `paper/numbers.tex`).

## Setup

```bash
pip install torch transformers huggingface_hub numpy scipy scikit-learn matplotlib pytest
```

Fetch data + checkpoints (~3 GB into `data/`, `models/`; both gitignored):

```bash
python scripts/fetch_assets.py
```

## Reproduce

```bash
python -m pytest tests/            # harness + label sanity (needs CUDA for GPU tests)
python scripts/sanity_accuracy.py  # reproduce published M1-M4 ProsQA accuracies
python scripts/run_all.py          # full E0-E3 pipeline (~10 h on one RTX 3080)
python experiments/make_figures.py # figures + paper/numbers.tex from results/*.json
python paper/verify_regen.py       # check every paper number regenerates
```

Pipeline stages can also be run individually; see headers of `experiments/exp0_gate.py`
(spectral prediction + null controls), `exp1_causal.py` (interventions),
`exp2_koopman.py` (routedness + EDMD), `exp3_robustness.py`.

## Layout

- `src/lrspec/` — harness (checkpoint loading, latent-phase capture, step maps,
  counterfactual reruns), `prosqa.py` (DAG labels, pruned-twin nulls), `spectra.py`
  (Jacobian invariants), `causal.py`, `koopman.py`, `stats.py`
- `experiments/` — E0-E3 + figure generation
- `results/` — committed JSON summaries; large arrays live in `runs/` (gitignored)
- `paper/` — LaTeX source; build with `tools/tectonic.exe main.tex`

## Assets

- Checkpoints: [bmarti44/coconut-curriculum-checkpoints](https://huggingface.co/bmarti44/coconut-curriculum-checkpoints)
  (GPT-2 124M; M1 CoT / M2 Coconut / M3 pause / M4 pause-multipass)
- Data: [facebookresearch/coconut](https://github.com/facebookresearch/coconut) `data/prosqa_*.json`
