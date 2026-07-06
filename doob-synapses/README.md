# doobsyn — intrinsic-noise consolidation via a Doob-barrier-conditioned diffusion

Cast per-synapse memory consolidation as a **Doob *h*-transform**: condition each
weight's stochastic dynamics on the event of never crossing a memory-critical
barrier around its consolidated value. The conditioned diffusion acquires an extra
drift `σ²·∂_w log h` — a restoring force toward the memory **amplified by the noise
variance itself**. The load-bearing, pre-registered prediction: increasing the
intrinsic (analog device) noise **non-monotonically improves** sequential-task
retention — an inverted-U the anchored-drift methods whose drift we share cannot
produce.

**Honest claim split.** The anchored drift `-s(w-μ)` is *not* new — it is the limit
of OUA / MESU / EWC, and we surrender it. We claim only the conjunction of (a) the
Doob barrier-conditioning as a synaptic rule and (b) the inverted-U. See
[`paper/main.tex`](paper/main.tex) and [`lit_notes.md`](lit_notes.md).

## Results (this repo)

- **GATE F (pre-registered go/no-go): PASS.** On Split-MNIST (8 seeds) the rule
  lifts retention by ~11 points at an interior noise optimum; matched OU/EWC/MESU
  anchors are monotone-decreasing in noise.
- **Mechanism isolation:** ablating the conditioning removes the effect; the
  optimum tracks the barrier.
- **BSS-2 emulation:** the inverted-U survives a device-faithful noise emulation
  (colored, multiplicative, fixed-pattern, 6-bit).
- **BSS-2 silicon:** the chip's intrinsic MAC noise was **measured on real
  BrainScaleS-2** (via EBRAINS; `results/bss2_silicon_noise.json`) — additive,
  white, CV 1.6–12%, `num_sends` averages as 1/√N. The emulation re-calibrated to
  the measurement keeps the inverted-U. On-chip **training** (measured retention +
  joules) is the remaining step (K2).
- **Second modality:** reproduces on continual Yin-Yang.

## Layout

```
src/doobsyn/     diffusion.py (Doob rule + baselines), sim.py (CL trainer),
                 data.py, models.py, energy.py, bss2.py (device noise + port), stats.py
experiments/     exp0_falsifier .. exp4_modality + make_figures.py
tests/           23 unit tests (pytest)
results/         committed *.json (every paper number derives from these)
paper/           main.tex, numbers.tex (AUTO-GENERATED), gen_paper_numbers.py,
                 verify_regen.py, references.bib, figs/
PLAN.md          pre-registration (committed BEFORE results): GATE F, K1-K3, fixed config
```

## Reproduce

```bash
pip install -r requirements.txt
pytest -q                                   # 23 tests
python experiments/exp0_falsifier.py        # GATE F
python experiments/exp1_isolation.py
python experiments/exp2_bss2.py
python experiments/exp3_baselines.py
python experiments/exp4_modality.py
python experiments/make_figures.py
python paper/gen_paper_numbers.py && python paper/verify_regen.py
```

MNIST is loaded from a local copy if present, else downloaded once. A CUDA GPU is
used if available (the full ladder runs in about an afternoon on one RTX 3080).

## What we do not claim

The drift is not novel; the BSS-2 result is emulation, not measured silicon; the
benefit is realized at the noise optimum, not everywhere; this is a mechanism and a
signature, not a retention leaderboard entry. See the paper's "What we do not claim".
