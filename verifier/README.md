# verifier — signal vs. substance in LLM research judges

Does an LLM-as-judge for research ideas score *substantive novelty* (real
conceptual distance from prior work) or merely *surface novelty-signaling*
(rhetoric that asserts novelty)? If the latter, the judge is a reward-hackable
verifier — unfit for the autonomous-research loop the field is racing toward —
and the bias may be a single linear direction that steering can correct.

This is project **P1** of a program seeded by *The Verification Bottleneck in
Automated AI Research* (`compass_artifact_*.md`).

## The claim

LLM research judges track surface novelty-signal (G), not substantive novelty
(S), holding correctness fixed. A crossed **2×2 (S × G)** design isolates the two;
a **reward-hackability** search shows signal farms score without substance; a
**linear probe** localizes the signal response; and **activation steering** tests
whether removing that direction recovers calibration. This reconciles two 2026
results that found opposite-signed novelty bias (RQ-Bench over-rates, RINoBench
under-rates — see `lit_notes.md`).

## Status

Pre-registration drafted (`PLAN.md`), nothing run yet. See `STATUS.md`.

## Layout

- `PLAN.md` — pre-registration (frozen before code).
- `lit_notes.md` — competitive landscape and the gap.
- `STATUS.md` — current state and next actions.
- `src/novjudge/` — package (schemas, construction, judging, estimation, steering).
- `experiments/` — run drivers (`run_all.sh` sweep → fits → figures → numbers).
- `tests/` — gates (estimator recovery, schema/validation, prompt-freeze checks).
- `results/` — committed fits the paper cites.
- `paper/` — LaTeX + machine-generated `numbers.tex` + `verify_regen.py`.

## Reproduce (once built)

```bash
pip install -e . && pytest tests/
bash experiments/run_all.sh
python paper/verify_regen.py   # numbers.tex byte-identity
```

## House rules

Prereg before code; every paper number machine-generated and byte-checked;
committed results JSON; bootstrap/cluster CIs; frozen judge prompts + signal
lexicon; recency re-sweep before submission. Author: Gunner Levi Howe.
