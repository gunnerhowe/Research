"""Integrator regression against the validated ornstein-dist implementation, and
self-consistency of the generalized streaming/batched variant."""
import sys
from pathlib import Path

import numpy as np
import pytest

from specext.ks import ks_trajectory, ks_stream_batch

ORNSTEIN_SRC = Path("E:/GitHub/Research/ornstein-dist/src")


@pytest.mark.skipif(not ORNSTEIN_SRC.exists(), reason="ornstein-dist not present")
def test_vendored_matches_ornstein_dist():
    sys.path.insert(0, str(ORNSTEIN_SRC))
    from ornstein.ks import ks_trajectory as ks_ref
    ref = ks_ref(50, dt_sample=1.0, dt=0.25, L=22.0, N=64, speed=1.0, seed=3,
                 transient=100.0)
    ours = ks_trajectory(50, dt_sample=1.0, dt=0.25, L=22.0, N=64, seed=3,
                         transient=100.0)
    np.testing.assert_allclose(ours, ref, rtol=1e-9, atol=1e-10)


def test_stream_batch_matches_reference_short_horizon():
    # identical ETDRK4 core + init family; mean-zero projection differs only at
    # roundoff, so short horizons must agree tightly
    ref = ks_trajectory(50, dt_sample=0.5, dt=0.25, L=22.0, N=64, seed=7,
                        transient=5.0)
    chunks = list(ks_stream_batch(22.0, 64, 50, seeds=[7], dt_sample=0.5,
                                  transient=5.0, chunk_samples=16))
    ours = np.concatenate(chunks)[:, 0]
    np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-8)


def test_stream_chunking_deterministic():
    a = np.concatenate(list(ks_stream_batch(22.0, 64, 40, seeds=[1, 2],
                                            transient=2.0, chunk_samples=7)))
    b = np.concatenate(list(ks_stream_batch(22.0, 64, 40, seeds=[1, 2],
                                            transient=2.0, chunk_samples=40)))
    np.testing.assert_array_equal(a, b)


def test_odd_parity_boundaries_and_stability():
    chunks = list(ks_stream_batch(22.0, 64, 200, seeds=[0], bc="odd",
                                  transient=200.0, chunk_samples=100))
    u = np.concatenate(chunks)[:, 0]  # (200, 64) on [0, L)
    assert np.abs(u[:, 0]).max() < 1e-10          # u(0) = 0 (Dirichlet-type)
    assert np.isfinite(u).all()
    assert 0.1 < u.std() < 5.0                     # on-attractor, bounded
