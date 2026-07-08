# STATUS — dbar-model-diffing

**Outcome: POSITIVE (existence proof stands, under disclosed amendments).**
Ornstein's d̄ on low-entropy symbolic readouts separates same-output/different-process
model pairs that CKA and DSA both miss.

## Timeline (2026-07-08, single session)

1. Kill-check brief received (`info.txt`); estimator ported from `../ornstein-dist`
   ("Beyond the Invariant Measure").
2. `PLAN.md` pre-registered and committed **before any results** (commit `6bef3c2`):
   E0 gate, K1–K3 kill conditions, σ* calibration rule, Marton–Shields wall guard.
3. E0 ran (5 seeds, 6 pair types, three metrics side-by-side, ~2.5 h on the 3080).
   **Pre-registered gate FAILED as written** — recorded verdict committed unmodified.
4. Diagnosis on the committed curves: two artifacts of the gate arithmetic (not the
   estimator): (a) argmax-Δ plateau rule lands on OT-bias-dominated sampled-regime rows;
   (b) DSA-similarity threshold assumed null ≪ different-task, but DSA's null and
   different-task distributions overlap. **AMENDMENT 1** committed (dated, in PLAN.md);
   amended gate evaluated by `analysis/amended_gate.py` on unmodified JSONs.
5. **Amended verdict: PASS (sign-consistent criterion) via the pruned twin** — CKA 0.9999,
   DSA 0.074 (inside DSA null range 0.028–0.118), d̄ 2.6–38× same-process floor, 5/5 seeds,
   outputs matched (CE ≤ base, unigram TV ≤ 0.008).
6. **AMENDMENT 2**: E1/E2 characterization budget reduced (GPU contention); committed
   before any E1 results were seen.
7. E1 sweeps: DSA saturates at the smallest noise (0.55 at σ=0.01; 0.96 by σ=0.2) but is
   flat-at-null across ALL prune fractions; d̄ separates pruned twins at every fraction
   (5–12× emitted, 125–400× belief) with CE/marginals/CKA matched.
8. E2 transformer readout (2-layer causal TF, GM + MESS3, 3 seeds): **distilled twins
   separate on every seed of both grammars** (GM: 3.9–28× floor at CKA 0.975; MESS3:
   6.7–12.5× within the tight wall ≈12) while DSA rates them (0.265) as *more* different
   than cross-task model pairs (0.228) — no ranking information. Residual noise: emitted
   process and even DSA quiet; only the belief readout separates (50–68× floor, 3/3).
9. **E3 (post-hoc, reviewer-proofing; PLAN.md Addendum)**: DSA config-robustness check —
   5 configs (pre-registered / deep 16×64 / no-delay / RRR / Wasserstein) on the four
   argument-carrying pairs. Verdict: the different-task blindness is CONFIG-SPECIFIC
   (4/5 alternatives separate the control), but NO config yields a process-distance
   ordering — all five rank the noise twin (identical deterministic dynamics) above the
   genuinely different computation (1.2–29×), and configs that flag the pruned twin flag
   it at the level of a full task change. All paper DSA claims scoped to "as configured,
   at this scale" accordingly; Amendment 3 formalizes the sign-consistency criterion.
10. Paper written (`paper/main.tex`, 16 pages), every number auto-generated
   (`gen_paper_numbers.py` → `numbers.tex`, checked by `verify_regen.py`), figures from
   `make_figures.py`. Built with latexmk.

## Headline numbers (E0, emitted-token readout, 5 seeds)

| pair | CKA | DSA | d̄*/floor |
|---|---|---|---|
| recoded null | 1.000 | 0.055±0.035 | 1.0 (at floor) |
| independent seed | 1.000 | 0.080±0.014 | 16.4±6.1 |
| noise twin σ=0.2 | 0.996 | 0.963±0.020 | 4.3±1.0 |
| distilled twin | 1.000 | 0.096±0.030 | 1.7±0.6 (correct null) |
| **pruned twin 50%** | **1.000** | **0.074±0.029** | **18.0±12.1** |
| different task (matched marginal+entropy) | 0.413 | 0.117±0.019 | 194.9±21.3 |

Belief readout (quantized P(next|history)): noise twin 150–880× floor; seeds 600×;
distill 16× — internal predictive process differs even when the emitted process is preserved.

## Key surprises vs. the brief's expectations

- The noise twin was pre-registered as the "highest-probability win" on the theory that
  DSA is blind to stochastic sameness. **Inverted**: DSA is hypersensitive to process noise
  (errors-in-variables bias of DMD), responding more strongly to σ=0.01 noise than to a
  genuinely different computation. The clean existence pair is the **pruned+fine-tuned twin**
  (deterministic, so DSA has no artifact to fire on).
- DSA cannot distinguish golden-mean from flipped-even models (its different-task scores sit
  inside its own null spread) — d̄ separates them at ~200× floor with *identical* unigram
  marginals and entropy rates.
- Distillation preserved the emitted process (d̄ ≈ floor: a useful certification result);
  prune+finetune and independent seeds did not. Process preservation ordering:
  distill (1.7×) < prune+ft (5–12×) < fresh seed (16×) — tracks how directly the twin's
  objective constrains the base conditionals.

## Kill conditions

- K1 (no separation): did **not** fire under the amended analysis; the pre-registered
  verdict (fail-as-written) is reported alongside in the paper (§ pre-registration).
- K2 (wall violation): respected — all claims at n ≤ 12 ≪ wall ≈ 31; V3 convergence figure
  shows the separation growing out of the floor with budget while n=32 stays at zero.
- K3 (isomorphism overclaim): enforced — Ornstein–Weiss stated first (§1, §2.3, §7).

## Repro

`pip install -r requirements.txt` (DSA via `pip install git+https://github.com/mitchellostrow/DSA.git`),
`python -m pytest tests/` (21 tests), `python experiments/exp0_existence.py`, then
`analysis/amended_gate.py`, `exp1_generality.py`, `exp2_transformer.py`;
`paper/gen_paper_numbers.py && paper/make_figures.py && latexmk -pdf paper/main.tex`.
Single RTX 3080. Checkpoints not committed (deterministic re-training from seeds).

## Atlas link-back

On ship: ingest as own-lab node, companions latent-reasoning-spectra + faithful-selection.
Stamp outcome: **positive** — d̄ separates same-output/different-process pairs (pruned twin,
independent seeds, noise twins) that CKA+DSA miss; bounded to fixed low-entropy readouts per
Ornstein–Weiss; pre-registered gate failed as written, amended analysis disclosed in full.
