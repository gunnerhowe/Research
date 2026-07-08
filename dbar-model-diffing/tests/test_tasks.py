"""HMM generators vs. theory."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dbar_diff.dbar import encode_blocks
from dbar_diff.entropy import entropy_rate
from dbar_diff.tasks import flipped_even, golden_mean, mess3


def test_golden_mean_theory():
    gm = golden_mean()
    assert gm.is_unifilar()
    assert abs(gm.entropy_rate() - 2 / 3) < 1e-12
    assert np.allclose(gm.pi, [2 / 3, 1 / 3])
    assert abs(gm.marginal()[1] - 1 / 3) < 1e-12


def test_flipped_even_matches_gm_h_and_marginal():
    fe = flipped_even()
    gm = golden_mean()
    assert fe.is_unifilar()
    assert abs(fe.entropy_rate() - gm.entropy_rate()) < 1e-12
    assert abs(fe.marginal()[1] - gm.marginal()[1]) < 1e-12


def test_gm_sample_forbids_11_and_entropy():
    gm = golden_mean()
    x = gm.sample(16, 40_000, seed=1)
    codes = encode_blocks([x[i] for i in range(16)], 2, 2)
    assert (codes == 3).sum() == 0          # "11" never occurs
    h, _, _ = entropy_rate([x[i] for i in range(16)], 2)
    assert abs(h - 2 / 3) < 0.01
    assert abs((x == 1).mean() - 1 / 3) < 0.01


def test_feven_sample_even_zero_runs():
    fe = flipped_even()
    x = fe.sample(4, 50_000, seed=2)
    # interior runs of 0s between 1s must have even length
    for i in range(4):
        s = "".join(map(str, x[i]))
        runs = [len(r) for r in s.strip("0").split("1") if r]
        assert all(r % 2 == 0 for r in runs)


def test_mess3_rows_and_sampling():
    m3 = mess3()
    assert m3.m == 3 and m3.S == 3
    x = m3.sample(8, 20_000, seed=3)
    counts = np.bincount(x.ravel(), minlength=3) / x.size
    assert np.allclose(counts, m3.marginal(), atol=0.02)
