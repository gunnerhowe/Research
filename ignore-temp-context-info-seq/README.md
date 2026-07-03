# Semantic Reference Frames (SemRF)

Code and paper for **"Semantic Reference Frames: Representing Time and Context
Relative to Learned Anchors in Sequence Models."**

SemRF is an attention bias that represents context and time *relative to a small
set of learned semantic anchor vectors* instead of absolute positions. Each
token is softly assigned to its nearest anchor and encoded as a residual offset
within that frame; attention logits receive (i) an anchor-frame affinity,
(ii) a within-frame residual alignment, and (iii) a **frame-conditioned temporal
decay** whose rate is a learned function of the token's semantic frame. With the
content terms gated off and a shared slope per head, SemRF reduces exactly to
ALiBi — and it is initialized at that operating point.

## Layout

```
semrf/                 library: model, attention, positional schemes, tasks
  positions.py         NoPE | sinusoidal | learned | RoPE | ALiBi | T5 | SemRF
  model.py             decoder-only transformer with pluggable positions
  data/synthetic.py    associative recall / temporal recency / selective copy
  data/charlm.py       enwik8/text8 pipeline (90M/5M/5M)
  train.py, eval.py    training loop; accuracy, bpc, extrapolation protocols
experiments/
  run_synthetic.py     synthetic sweep driver (resumable, one JSON per run)
  run_charlm.py        enwik8 sweep driver (resumable)
scripts/
  run_all.py           one-command reproduction of everything
  analyze.py           aggregates results -> summary CSVs + significance tests
  make_figures.py      publication figures + LaTeX tables from results
  anchor_analysis.py   anchor interpretability (clusters, per-frame time decay)
  smoke_test.py        fast end-to-end sanity check of all 7 schemes
paper/                 LaTeX source; compiles with tectonic
results/               one JSON per run + summary CSVs (created by experiments)
```

## Setup

```bash
pip install -r requirements.txt          # torch with CUDA recommended
python scripts/smoke_test.py             # ~2 min, verifies all 7 schemes
```

## Reproduce everything

```bash
python -u scripts/run_all.py             # synthetic sweep + enwik8 + analysis + figures
```

Or stage by stage:

```bash
python -u -m experiments.run_synthetic --ablations --seeds 0 1 2
python -u -m experiments.run_charlm --corpus enwik8 --steps 10000 --seeds 0 1
python -m scripts.analyze
python -m scripts.make_figures
python -m scripts.anchor_analysis
```

Every run writes a self-contained JSON (config + metrics) under `results/`;
completed runs are skipped on re-launch, so interrupted sweeps just resume.

## Paper

```bash
cd paper && tectonic main.tex            # or latexmk -pdf main.tex
```

Figures/tables are generated into `paper/figures` and `paper/tables` by
`scripts/make_figures.py`; numeric claims are centralized in
`paper/results_macros.tex`.

## Notes on experimental design

- All positional schemes share one backbone; only the position module differs.
- Length extrapolation grows the *span* while holding memory load fixed
  (e.g. fixed #pairs with a growing, in-distribution filler gap), isolating
  positional generalization from recall capacity (cf. Zoology/MQAR).
- The synthetic model is d=256/8-heads because the induction (content-matching)
  circuit does not reliably form in smaller models within budget — and it forms
  *late* (a phase transition between 6k and 15k steps at d=256).
