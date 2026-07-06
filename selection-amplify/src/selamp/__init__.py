"""selamp -- selection-corrected data amplification.

LOCATE (selection.py) an MNAR selector by density-ratio classification, GENERATE
(bridge.py) whole labeled units in the recoverable collar of the censored
complement with a Doob guided bridge, and characterize the gain against the
selection-entropy axis (entropy.py). validate.py guards the manifold; stats.py
and downstream.py evaluate.
"""
__all__ = ["data", "selection", "diffusion", "bridge", "entropy", "validate",
           "stats", "downstream"]
