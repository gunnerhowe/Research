"""Estimator tests against analytic cases."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dbar_diff.dbar import dbar_n, dbar_pair_curve, encode_blocks, plateau
from dbar_diff.entropy import entropy_rate, fano_lower_bound, lz78_entropy

RNG = np.random.default_rng(0)


def test_identical_streams_zero():
    x = RNG.integers(0, 2, 50_000).astype(np.int8)
    for n in (1, 2, 8):
        v, _ = dbar_n(x, x.copy(), n, 2)
        assert v < 1e-12


def test_iid_bernoulli_marginal():
    # d̄ between iid Bernoulli(p) and Bernoulli(q) equals |p - q| at every n.
    p, q, N = 0.2, 0.5, 400_000
    x = (RNG.random(N) < p).astype(np.int8)
    y = (RNG.random(N) < q).astype(np.int8)
    for n in (1, 4, 8):
        v, _ = dbar_n(x, y, n, 2)
        assert abs(v - abs(p - q)) < 0.02, (n, v)


def test_multichain_pooling_matches_single():
    x = RNG.integers(0, 2, 40_000).astype(np.int8)
    codes_single = encode_blocks(x, 4, 2)
    chains = [x[:20_000], x[20_000:]]
    codes_multi = encode_blocks(chains, 4, 2)
    # multi-chain drops the 3 cross-boundary blocks, keeps everything else
    assert len(codes_single) - len(codes_multi) == 3
    v_pair, _ = dbar_n(chains, [x[10_000:30_000]], 4, 2)
    assert v_pair < 0.02


def test_period_two_vs_iid_blocks():
    # alternating 0101... has same unigram as fair iid, differs from n=2 on
    x = np.tile([0, 1], 30_000).astype(np.int8)
    y = RNG.integers(0, 2, 60_000).astype(np.int8)
    v1, _ = dbar_n(x, y, 1, 2)
    v4, _ = dbar_n(x, y, 4, 2)
    assert v1 < 0.01
    assert v4 > 0.15


def test_pair_curve_and_plateau():
    x = (RNG.random(100_000) < 0.5).astype(np.int8)
    y = np.tile([0, 1], 50_000).astype(np.int8)
    x2 = (RNG.random(100_000) < 0.5).astype(np.int8)
    y2 = np.tile([0, 1], 50_000).astype(np.int8)
    rows = dbar_pair_curve(x, y, x2, y2, 2, ns=(1, 2, 4, 8), repeats=2)
    win = plateau(rows, k_wall=10)
    assert win["n"] >= 2 and win["delta"] > 0.1
    r1 = next(r for r in rows if r["n"] == 1)
    assert r1["dbar"] < 0.02


def test_entropy_and_fano():
    x = (RNG.random(500_000) < 0.5).astype(np.int8)
    h, _, _ = entropy_rate(x, 2)
    assert abs(h - 1.0) < 0.02
    # LZ78 converges slowly from above (~+20% at this N); it is a cross-check only
    assert 0.9 < lz78_entropy(x, 2) < 1.3
    assert fano_lower_bound(0.0, 2) == 0.0
    # g(0.11) ≈ 0.4999 bits for m=2, so a gap of 0.5 bits inverts to ≈ 0.110
    assert abs(fano_lower_bound(0.5, 2) - 0.110) < 0.01
