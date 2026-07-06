# STATUS — doobsyn

**Outcome: POSITIVE; mechanism DEMONSTRATED ON REAL SILICON.** The pre-registered
GATE F passes; the mechanism is isolated to the barrier conditioning; the inverted-U
survives a BrainScaleS-2 noise emulation, a second modality, and the hardware-faithful
forward-noise realization. The chip's intrinsic noise was MEASURED on real
BrainScaleS-2 (additive, white, CV up to 12%, num_sends ~ 1/sqrt(N)). And --- the
headline --- the barrier-conditioned consolidation was RUN on real BrainScaleS-2 with
the chip in the training loop (hardware-in-the-loop, hxtorch): its own intrinsic
noise, steered by the conditioning, retains a prior task +15.6 pts better than the
matched unconditioned control. Single-seed proof of concept, retention not joules;
a full on-silicon noise sweep with energy is the next study.

## Gates / kill conditions

- **GATE F (E0): PASS.** Split-MNIST, 8 seeds. Barrier-conditioned rule = retention
  inverted-U, ~+11 pts at an interior noise optimum (paired Wilcoxon p≈0.004 both
  ends); matched OU/EWC/MESU/naive anchors monotone-decreasing in noise. K1 does
  not fire.
- **E1 isolation: PASS.** κ:1→0 (ablate conditioning) flattens the curve; optimum
  tracks the barrier scale. K3 does not fire.
- **E2 BSS-2 (EMULATION): inverted-U survives** colored+multiplicative+fixed-pattern
  +6-bit device noise; color scan maps the boundary.
- **E5 ON-SILICON: intrinsic noise MEASURED** (chip hxcube7fpga3chip61_1): additive
  94.6%, white, CV 1.6-12%, num_sends ~ 1/sqrt(N). Emulation re-calibrated to the
  measurement -> inverted-U survives (13.2 pts).
- **E6 FORWARD-noise realization:** mechanism survives noise in the MAC/forward (the
  hardware-relevant case): doob inverted-U +12.9 pts p=0.004, ou flat. The importance
  CLAMP is the fix (noclamp -> collapse). Coupling tunes the optimum to reachable ~5% CV.
- **E7 ON-SILICON TRAINING (headline):** barrier-conditioned consolidation RUN on real
  BrainScaleS-2 (hardware-in-the-loop, chip noise in the loop) -> **+15.6 pts retention**
  vs the matched control. Single seed, ~79 min chip time, retention not joules.
- **E3 baselines:** ours at the noise optimum vs matched-budget incumbents
  (OU/OUA, MESU, EWC, Benna-Fusi, replay, naive) + energy/noise-tax.
- **E4 modality:** continual Yin-Yang reproduces the inverted-U; noise-optimum vs
  task-similarity mapped.

## House discipline

- PLAN.md pre-registered and committed BEFORE results (git-verifiable).
- 23 unit tests green.
- Every paper number is a machine-generated macro (paper/gen_paper_numbers.py →
  numbers.tex); paper/verify_regen.py checks byte-identity.
- Live citation verification + recency sweep (lit_notes.md); the (a)+(b)
  conjunction is unoccupied. Unverifiable "node-perturbation on BSS-2" citation
  dropped.

## Atlas link-back

Card closes **positive; mechanism demonstrated on real silicon**: the Doob
barrier-conditioning (a) + intrinsic-noise inverted-U (b) hold in sim, in an
emulation calibrated to MEASURED BrainScaleS-2 noise, in the hardware-faithful
forward-noise realization, AND on real BrainScaleS-2 with the chip in the training
loop (+15.6 pts retention vs the matched control, single-seed proof of concept). The
remaining study is a full on-silicon noise sweep with measured joules. Ship honest;
stamp positive-on-silicon (proof-of-concept) / full-sweep-and-energy-pending. DOI at
submission.
