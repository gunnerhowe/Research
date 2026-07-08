#!/usr/bin/env bash
# Full experimental sweep, sequential on one GPU. Resumable (each run skips
# already-recorded qids). Run from repo root.
set -e
cd "$(dirname "$0")/.."

N=600
HINTS_MAIN="sycophancy,authority,metadata,consistency"
HINTS_E3="sycophancy,authority,metadata,consistency,placebo"

# --- Primary model: full pipeline incl. logit-lens (R_pre for E1) ---
python experiments/run_model.py --model qwen7b   --n $N --hints $HINTS_E3 --seed 0 --out results/raw/qwen7b_e0.jsonl

# --- Robustness models: R_TE + R_NDE only (no lens -> ~2x faster) ---
python experiments/run_model.py --model mistral7b --n $N --hints $HINTS_E3 --seed 0 --out results/raw/mistral7b_e3.jsonl --no-lens
python experiments/run_model.py --model phi35     --n $N --hints $HINTS_E3 --seed 0 --out results/raw/phi35_e3.jsonl --no-lens

# --- E2 observation-only reasoning model (no lens; unblinded after) ---
python experiments/run_model.py --model nemotron8b --n $N --hints $HINTS_MAIN --seed 7 \
    --out results/raw/nemotron8b_e2.jsonl --no-lens

# --- fits ---
python experiments/fit_all.py --raw results/raw/qwen7b_e0.jsonl   --tag qwen7b_e3    --heckprob --n-boot 1000
python experiments/fit_all.py --raw results/raw/mistral7b_e3.jsonl --tag mistral7b_e3 --n-boot 1000
python experiments/fit_all.py --raw results/raw/phi35_e3.jsonl     --tag phi35_e3     --n-boot 1000
python experiments/fit_all.py --raw results/raw/nemotron8b_e2.jsonl --tag nemotron8b_e2 --heckprob --n-boot 1000
python experiments/fit_all.py --raw results/raw/claude_e2.jsonl    --tag claude_e2    --heckprob --n-boot 300

# --- detector validation + paper ---
python experiments/judge_v.py --raw results/raw/qwen7b_e0.jsonl --n 120 --out results/judge_v_qwen7b.json
python experiments/make_figures.py
python paper/gen_paper_numbers.py
python paper/verify_regen.py
