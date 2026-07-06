"""Selection-entropy axis.

The headline scalar is the mutual information I(O;X) between the observation
indicator O in {0,1} and the covariate X:

    I(O;X) = H(O) - H(O|X) = Hb(E_p[s]) - E_p[Hb(s(X))]     (bits),

with Hb the binary entropy. I(O;X)=0 under ignorable missingness (s constant
in x, e.g. beta=0's uniform thinning) and upper-bounds the recoverable gain.
We deliberately do NOT use propensity-spread entropy H(s(X)) alone: a uniform
50% thinning has zero censoring but nonzero spread once s varies.

The complement-specific driver is the KL divergence of the censored complement
from the population:

    D_comp = KL( p(x | O=0) || p(x) ),   p(x|O=0) ∝ (1 - s(x)) p(x)   (nats),

estimated self-normalized over a population sample.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def binary_entropy_bits(p):
    p = np.clip(p, _EPS, 1 - _EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def mutual_info_OX(s_values):
    """I(O;X) in bits from s(x) sampled on a population sample {x_i}~p."""
    s = np.clip(np.asarray(s_values, float), _EPS, 1 - _EPS)
    H_O = binary_entropy_bits(s.mean())            # H(O)
    H_O_given_X = binary_entropy_bits(s).mean()    # E_p[Hb(s(x))]
    return float(H_O - H_O_given_X)


def d_comp(s_values):
    """KL(p(x|O=0) || p(x)) in nats, self-normalized over {x_i}~p."""
    s = np.clip(np.asarray(s_values, float), _EPS, 1 - _EPS)
    w = (1.0 - s)
    w = w / w.mean()                               # importance weights, mean 1
    return float(np.mean(w * np.log(w)))


def selection_entropy_report(s_values):
    return {
        "I_OX_bits": mutual_info_OX(s_values),
        "D_comp_nats": d_comp(s_values),
        "obs_frac": float(np.mean(s_values)),
    }
