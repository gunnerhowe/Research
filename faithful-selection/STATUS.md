# STATUS — faithful-selection

**CoT faithfulness as endogenous sample selection: a Heckman-style test and
correction for the verbalization confound.**
Author: Gunner Levi Howe. Last updated: 2026-07-09.

## One-line

We recast chain-of-thought faithfulness metrics as a Heckman
selection-on-unobservables problem: a probit selection equation for
*what-gets-verbalized* and an outcome equation for *latent hint reliance*,
linked by correlated errors, giving a scalar test for the verbalization
confound (H0: rho=0) and an inverse-Mills correction to naive faithfulness
estimates — validated against causal ground truth on open models and applied
observation-only to a closed API model.

## Outcome: HONEST NEGATIVE (K1 in substance)

The selection-theoretic reframing is sound and the rho-test is a usable
diagnostic, but **the closed-form correction does not improve faithfulness
estimates on validated data**, and the more actionable finding is that the
field-standard lexical verbalization detector badly mis-measures V.

- **Detectors over-count verbalization 2.2–9.5x** vs an LLM judge that reads
  each trace (Nemotron 53.5%→24.8%, Phi 34.2%→14.0%, Qwen 26.1%→2.7%,
  Claude 41%→9%). True verbalization is rare (2.7–24.8%), matching the
  sub-20% literature. A detector-based "positive" evaporated once V was
  validated — so all estimation uses judge-V (judged every hinted instance).
- **On judge-V the correction never beats the naive count.** Nemotron
  (V=0.245): rho-test rejects (rho=-0.68, LR p=0.023) but naive is already
  near-unbiased (1.28 vs true 1.00) and the IMR correction *overshoots*
  (2.74). rho-sensitivity shows truth recovered at rho≈0 → the rejection is a
  bivariate-normality artifact (heavy-tailed reliance), not a correctable
  bias. Phi (V=0.141): test null (rho=-0.03). Qwen (V=0.027): unfittable
  (13 verbalizers). Placebo: null.
- **Salvage / contribution:** (1) the formal selection framework + scalar
  rho-test (novel, unclaimed bridge); (2) a measurement caution — lexical
  verbalization metrics mis-measure the quantity faithfulness audits count;
  (3) the Gaussian IMR is the wrong tool for heavy-tailed reliance →
  rho-sensitivity bounds are the honest deliverable. Shippable as a
  cautionary methods note (info.txt listed honest-negative as shippable).

## What was run

- **Estimator core** (`src/faithsel/selection.py`): probit / Heckman two-step /
  joint MLE / fixed-rho MLE, rho LR+Wald tests, probit-with-selection
  (`heckprob`) for binary observation-only proxies, IMR-corrected estimands
  (population + hidden reliance), rho-sensitivity grid, bootstrap. Gated by
  36 tests incl. agreement with statsmodels and synthetic parameter recovery.
- **Measurement** (`hints.py`, `gen.py`, `data.py`, `analysis.py`): Turpin-style
  hint injection (sycophancy / authority / metadata / consistency + placebo),
  a randomized **disclosure instrument**, a deterministic verbalization
  detector (LLM-judge validated), pooled MMLU/ARC/OpenbookQA, a 4-bit GPU
  engine producing greedy CoT + letter-read reliance (R_TE total effect,
  R_NDE hint-excision direct effect) + logit-lens pre-verbalization
  commitment (R_pre).
- **Roster**: Llama-3.1-Nemotron-Nano-8B (primary, local, ground-truth
  validated — chosen by a verbalization probe), Qwen2.5-7B and Phi-3.5-mini
  (robustness, local, spanning lower verbalization rates), Claude Haiku 4.5
  (observation-only, API). Mistral dropped (redundant); Llama-3.1-8B / Gemma
  gated for this HF account.

## Experiment ladder

- **E0 gate** — FAILED the "correction beats naive" half: rho rejects but the
  IMR correction overshoots ground truth on validated V (K1 in substance).
- **E1 identification** — disclosure instrument first stage strong (z≈8);
  exclusion holds for direct effect R_NDE, violated for total effect R_TE
  (K2 caught by balance test → identify on R_NDE). The one part that works.
- **E2 observation-only** — the probit-with-selection pipeline runs
  end-to-end on Claude via API (judge-V 0.09); reported as deployability
  evidence only, since the correction is not validated.
- **E3 robustness** — 3 open models span validated V 0.03–0.25; the
  correction beats naive in none; placebo null; detector-vs-judge over-count
  documented; bootstrap CIs throughout.

## Pre-registration & amendments

`PLAN.md` was committed before any results. Amendments A1–A5 (token budget,
detector recall, disclosure instrument, parsimonious separation-robust design,
primary-model selection by verbalization probe) were made after small pilots
diagnosing measurement validity, before any rho/confound result was read, and
are logged in `PLAN.md`. Kill conditions K1–K3 pre-registered.

## Reproduce

```bash
pip install -e . && pytest tests/          # 36 estimator gates
bash experiments/run_all.sh                # sweep -> fits -> figures -> numbers
python paper/verify_regen.py               # numbers.tex byte-identity
```

Raw run records (`results/raw/*.jsonl`) regenerate from the drivers; committed
`results/*.json` carry the fitted numbers the paper cites.

## Deliverables

- `paper/main.tex` (+ machine-generated `numbers.tex`, `verify_regen.py`).
- `PLAN.md` (pre-registration), `laymen.md` (plain-language), this `STATUS.md`.
- `results/*.json` (committed fits), `paper/figs/` (regenerated figures).
