"""Intrinsic-noise consolidation: a Doob-h-transform barrier-conditioned
diffusion for on-chip continual learning.

Public surface:
  diffusion.py -- the barrier-conditioned (Doob) per-synapse rule and the
                  matched anchored-drift baselines (unconditioned OU, EWC, MESU,
                  Benna-Fusi, replay).
  sim.py       -- sequential continual-learning testbeds (Split-MNIST domain-IL,
                  continual Yin-Yang) and the Euler-Maruyama trainer.
  energy.py    -- compute/energy accounting for the retention-vs-cost axis.
  bss2.py      -- BrainScaleS-2 intrinsic-noise model (device-faithful emulation)
                  and the real-hardware port hooks (gated behind hxtorch).
  stats.py     -- seed statistics, paired Wilcoxon, inverted-U test.
"""
__version__ = "0.1.0"
