# Structure-specific representational priors and the grokking delay

Does the grokking delay measure the time needed to form *task-structured* representations,
or just generically favorable geometry/optimization state? We inject a supervised-contrastive
auxiliary loss on hidden representations whose positive sets encode the task's true equivalence
structure (c = (a+b) mod p), and compare against an exactly-matched shuffled-structure control,
a Grokfast baseline, and a weight-norm-matched control.

## Layout
- `src/` — model (1-layer no-LayerNorm transformer, Nanda-style), data, losses, Grokfast, metrics, train loop
- `scripts/run_grid.py` — grid runner (phase A: baseline/supcon/grokfast; phase B: norm-matched)
- `analysis/analyze.py` — aggregates runs, stats, publication figures
- `paper/` — LaTeX source
- `lit_notes.md` — literature notes + positioning against the June 2026 must-reads

## Reproduce
```
python scripts/run_grid.py --phase A --seeds 0 1 2 3 4 --lams 0.1 0.3 1.0
python scripts/run_grid.py --phase B --seeds 0 1 2 3 4 --primary_lam 0.3
python analysis/analyze.py
```
Hardware: single RTX 3080, ~3-5 h total.
