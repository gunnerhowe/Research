# STATUS — doobsyn

**Outcome: POSITIVE; K2 first half resolved on real silicon.** The pre-registered
GATE F passes; the mechanism is isolated to the barrier conditioning; the inverted-U
survives a BrainScaleS-2 noise emulation and a second modality. The chip's intrinsic
noise was MEASURED on real BrainScaleS-2 (chip hxcube7fpga3chip61_1, EBRAINS-25.10):
additive, white, CV up to 12%, num_sends ~ 1/sqrt(N) -- the benign noise class the
mechanism needs, at a reachable amplitude; the emulation calibrated to it keeps the
inverted-U (lift 13.2 pts). The remaining step is on-chip TRAINING (measured
retention + joules) -- K2 second half.

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
  measurement -> inverted-U survives (13.2 pts). K2 first half RESOLVED. On-chip
  training (retention + joules) = remaining step; no on-chip retention/joules claimed.
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

Card closes **positive; intrinsic-noise structure verified on real silicon,
on-chip training pending**: the Doob barrier-conditioning (a) + intrinsic-noise
inverted-U (b) hold in sim and in an emulation calibrated to MEASURED BrainScaleS-2
noise (additive/white, reachable amplitude, num_sends knob); the remaining
validation is the on-chip retention curve + joules. Ship honest; stamp
positive-sim-plus-measured-noise / on-chip-training-pending. DOI at submission.
