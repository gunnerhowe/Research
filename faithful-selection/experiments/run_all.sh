#!/usr/bin/env bash
# Full experimental sweep, sequential on one GPU. Resumable (each run skips
# already-recorded qids). Run from repo root.
#
# Roster (set by the verbalization probe, PLAN amendment A5):
#   PRIMARY  nemotron8b  -- reasoning model, V~0.65, strong disclosure-instrument
#            first stage; full logit-lens for R_pre; ground-truth validated.
#   ROBUST   qwen7b, mistral7b, phi35 -- additional open models (lower V).
#   OBS-ONLY claude (separate driver) + nemotron heckprob-unblind bridge.
set -e
cd "$(dirname "$0")/.."

N=600
HINTS="sycophancy,authority,metadata,consistency,placebo"
HINTS_NP="sycophancy,authority,metadata,consistency"

# --- PRIMARY: Nemotron, full pipeline incl. logit-lens (R_pre for E1) ---
python experiments/run_model.py --model nemotron8b --n $N --hints $HINTS --seed 0 \
    --out results/raw/nemotron8b_e0.jsonl --gen-bs 12 --lens-bs 4 --chunk 32 --max-new-tokens 512

# --- ROBUSTNESS: R_TE + R_NDE only (no lens). Two open models spanning
#     lower verbalization rates (Mistral dropped: redundant + tokenizer flake). ---
python experiments/run_model.py --model qwen7b --n $N --hints $HINTS --seed 0 --out results/raw/qwen7b_e3.jsonl --gen-bs 16 --chunk 32 --max-new-tokens 512 --no-lens
python experiments/run_model.py --model phi35  --n $N --hints $HINTS --seed 0 --out results/raw/phi35_e3.jsonl  --gen-bs 16 --chunk 32 --max-new-tokens 512 --no-lens

# --- fits ---
python experiments/fit_all.py --raw results/raw/nemotron8b_e0.jsonl --tag nemotron8b_e0 --heckprob --n-boot 1000
python experiments/fit_all.py --raw results/raw/qwen7b_e3.jsonl     --tag qwen7b_e3     --n-boot 1000
python experiments/fit_all.py --raw results/raw/phi35_e3.jsonl      --tag phi35_e3      --n-boot 1000
python experiments/fit_all.py --raw results/raw/claude_e2.jsonl     --tag claude_e2     --heckprob --n-boot 300

# --- detector validation + paper ---
python experiments/judge_v.py --raw results/raw/nemotron8b_e0.jsonl --n 120 --out results/judge_v_nemotron8b.json
python experiments/make_figures.py
python paper/gen_paper_numbers.py
python paper/verify_regen.py
