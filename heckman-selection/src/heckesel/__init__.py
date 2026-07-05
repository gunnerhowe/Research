"""heckesel: Heckman-type selection models for machine learning.

Shared core for the two-paper program:
- Paper A: Heckman-corrected epistemic uncertainty (paper/)
- Paper B: survivor bias in learning-curve surrogates (paper2/)
"""

from .selection import probit_fit, heckman_two_step, heckman_mle, inverse_mills

__all__ = ["probit_fit", "heckman_two_step", "heckman_mle", "inverse_mills"]
