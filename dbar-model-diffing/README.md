# d-bar model diffing

Ornstein's d̄ — an ergodic-theory metric on stationary stochastic processes — applied as a
computational-equivalence metric for comparing neural models on a fixed low-entropy symbolic
readout. Question: can a process-level distance separate *same-output / same-marginal but
different-process* model pairs (a model vs. its noise-injected / distilled / pruned twin) that
static representation metrics (CKA) and deterministic-conjugacy metrics (DSA) do not?

- `PLAN.md` — pre-registered design, gate, and kill conditions (committed before results).
- `src/dbar_diff/` — d̄ OT estimator (n-block distributions, Hamming cost), entropy/wall
  diagnostics, HMM task generators, GRU/Transformer models and constructed twins, CKA + DSA
  baselines.
- `experiments/` — E0 existence proof (hard gate), E1 generality sweeps, E2 transformer readout.
- `results/` — committed JSONs; every paper number is generated from these
  (`paper/gen_paper_numbers.py`, verified by `paper/verify_regen.py`).
- `tests/` — analytic estimator cases, HMM theory checks, model mechanics, baseline sanity.

Run: `pip install -r requirements.txt`, then `python experiments/exp0_existence.py` (E1/E2 are
gated on E0's outcome). One consumer GPU suffices.
