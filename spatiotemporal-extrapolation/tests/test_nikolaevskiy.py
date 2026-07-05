"""Nikolaevskiy integrator: dispersion symbol, marginal k=0 mode, mean
conservation, boundedness, and deterministic chunking."""
import numpy as np

from specext.nikolaevskiy import nik_symbol, nik_stream_batch


def test_symbol_marginal_and_band():
    k = np.linspace(0, 2.5, 400)
    r = 0.3
    sig = nik_symbol(k, r)
    assert abs(sig[0]) < 1e-12                     # sigma(0) = 0 exactly (marginal)
    # unstable band exists around k=1 for r>0; stable at small k and high k
    assert sig.max() > 0
    assert nik_symbol(np.array([0.2]), r)[0] < 0   # soft/slow long-wave mode
    assert nik_symbol(np.array([2.2]), r)[0] < 0   # -k^6 damping tail
    kmax = k[np.argmax(sig)]
    assert 0.7 < kmax < 1.3                         # most-unstable near k=1


def test_mean_conserved_and_bounded():
    L, N = 44.0, 128
    chunks = list(nik_stream_batch(L, N, 300, seeds=[0], r=0.3, dt=0.1,
                                   dt_sample=0.5, transient=500.0, chunk_samples=150))
    u = np.concatenate(chunks)[:, 0]                # (300, N)
    assert np.isfinite(u).all()
    assert np.abs(u.mean(axis=1)).max() < 1e-6      # mean-zero sector conserved
    assert 0.1 < u.std() < 20.0                     # on-attractor, bounded


def test_chunking_deterministic():
    a = np.concatenate(list(nik_stream_batch(44.0, 128, 80, seeds=[0, 1], r=0.25,
                                             transient=200.0, chunk_samples=13)))
    b = np.concatenate(list(nik_stream_batch(44.0, 128, 80, seeds=[0, 1], r=0.25,
                                             transient=200.0, chunk_samples=80)))
    np.testing.assert_array_equal(a, b)


def test_is_chaotic_seed_divergence():
    # two nearby seeds decorrelate (sensitive dependence) -> extensive chaos
    L, N = 66.0, 192
    u = np.concatenate(list(nik_stream_batch(L, N, 400, seeds=[0, 1], r=0.3,
                                             transient=1000.0, chunk_samples=200)))
    late = u[-200:]
    c = np.corrcoef(late[:, 0].ravel(), late[:, 1].ravel())[0, 1]
    assert abs(c) < 0.5                              # distinct trajectories
