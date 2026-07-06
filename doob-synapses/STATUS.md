# STATUS — doobsyn

**Outcome: POSITIVE (in simulation + device-faithful emulation).** The
pre-registered GATE F passes; the mechanism is isolated to the barrier
conditioning; the inverted-U survives a BrainScaleS-2 noise emulation and a second
modality. The one result NOT obtained is measured silicon (no hardware access);
that on-chip run is the pre-registered remaining step (K2).

## Gates / kill conditions

- **GATE F (E0): PASS.** Split-MNIST, 8 seeds. Barrier-conditioned rule = retention
  inverted-U, ~+11 pts at an interior noise optimum (paired Wilcoxon p≈0.004 both
  ends); matched OU/EWC/MESU/naive anchors monotone-decreasing in noise. K1 does
  not fire.
- **E1 isolation: PASS.** κ:1→0 (ablate conditioning) flattens the curve; optimum
  tracks the barrier scale. K3 does not fire.
- **E2 BSS-2 (EMULATION): inverted-U survives** colored+multiplicative+fixed-pattern
  +6-bit device noise; color scan maps the boundary. Real silicon = pending (K2);
  no measured joules claimed.
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

Card closes **positive-with-mechanism in simulation/emulation, silicon pending**:
the Doob barrier-conditioning (a) + intrinsic-noise inverted-U (b) hold in sim and
in a device-faithful BSS-2 emulation; the on-silicon measurement (retention +
joules) is the stated remaining validation. Ship honest; stamp positive-sim /
silicon-pending. DOI at submission.
