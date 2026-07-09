# faithful-selection

**CoT faithfulness as endogenous sample selection: a Heckman-style test and
correction for the verbalization confound.**

A model's chain-of-thought is a *selected* projection of its latent
computation. Verbalization-based faithfulness metrics therefore compute
selected-sample statistics. This repo casts *what-gets-verbalized* as a
probit selection equation and *latent reliance on an injected hint* as an
outcome equation with correlated errors (Heckman 1979), yielding:

- a scalar **test** for the verbalization-selection confound (H0: rho = 0),
- an inverse-Mills-ratio **correction** to naive faithfulness estimates,
  including a closed-form estimate of the reliance hidden in
  non-verbalizing CoTs,
- an observation-only variant (probit-with-selection) for API models.

Validated against hint-excision causal ground truth on open-weight models.

**Outcome: honest negative (see [STATUS.md](STATUS.md)).** With verbalization
measured by a validated LLM judge rather than the field-standard lexical
detector — which we show over-counts verbalization 2–9× — the naive
faithfulness estimate is approximately unbiased and the IMR correction does
not beat it (it overshoots, a bivariate-normality artifact). The reframing and
the ρ-test survive as diagnostics; the correction does not deliver, and the
more actionable finding is that lexical verbalization metrics badly mis-measure
the quantity faithfulness audits count.

## Layout

- `PLAN.md` — pre-registration (committed before results): endpoints,
  instrument validation, kill conditions K1-K3.
- `src/faithsel/` — `selection.py` (estimators + tests + estimands),
  `hints.py` (hint injection, instrument, V detector), `gen.py` (GPU
  engine), `data.py`, `analysis.py`, `figures.py`.
- `experiments/` — `run_model.py` (GPU driver), `fit_all.py` (fits),
  `exp2_claude.py` (API arm), `judge_v.py` (detector validation).
- `tests/` — 36 tests gating the estimator core and pipeline.
- `results/` — committed fit JSONs (`results/raw/` JSONL is gitignored,
  regenerable via the drivers; the paper's numbers derive from the JSONs).
- `paper/` — `main.tex`, machine-generated `numbers.tex`
  (`gen_paper_numbers.py`, checked by `verify_regen.py`).

## Reproduce

```bash
pip install -e .
pytest tests/                                   # estimator gates
python experiments/run_model.py --model qwen7b --n 600 \
    --hints sycophancy,authority,metadata,consistency --seed 0 \
    --out results/raw/qwen7b_e0.jsonl           # ~single-digit 3080 hours
python experiments/fit_all.py --raw results/raw/qwen7b_e0.jsonl \
    --tag qwen7b_e0 --heckprob
python paper/gen_paper_numbers.py && python paper/verify_regen.py
```
