"""CKA and DSA sanity: invariances and known-different cases."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dbar_diff.baselines import cka_states, dsa_distance, linear_cka

RNG = np.random.default_rng(0)


def test_cka_self_and_orthogonal_invariance():
    X = RNG.standard_normal((500, 16))
    assert abs(linear_cka(X, X) - 1.0) < 1e-9
    Q, _ = np.linalg.qr(RNG.standard_normal((16, 16)))
    assert abs(linear_cka(X, X @ Q) - 1.0) < 1e-9
    Y = RNG.standard_normal((500, 16))
    assert linear_cka(X, Y) < 0.3


def _lds_traj(A, B_batch=16, T=120, d=None, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    d = A.shape[0]
    X = np.empty((B_batch, T, d))
    x = rng.standard_normal((B_batch, d))
    for t in range(T):
        x = x @ A.T + noise * rng.standard_normal((B_batch, d))
        X[:, t] = x
    return X


def test_dsa_conjugate_vs_different():
    # two orthogonally-conjugate linear systems -> DSA ~ 0;
    # a system with different eigenvalues -> DSA large.
    d = 8

    def rot_sys(angles):
        A = np.zeros((d, d))
        for i, a in enumerate(angles):
            A[2 * i:2 * i + 2, 2 * i:2 * i + 2] = \
                np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * 0.95
        return A

    A = rot_sys((0.3, 0.7, 1.1, 1.7))
    Q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((d, d)))
    A_conj = Q @ A @ Q.T
    A_diff = rot_sys((0.05, 0.45, 1.4, 2.6))  # different rotation frequencies

    X = _lds_traj(A, seed=1, noise=0.05)
    Y = _lds_traj(A_conj, seed=2, noise=0.05)
    Z = _lds_traj(A_diff, seed=3, noise=0.05)
    kw = dict(n_delays=4, rank=8, iters=400, skip=4, device="cpu")
    d_same = dsa_distance(X, Y, **kw)
    d_diff = dsa_distance(X, Z, **kw)
    assert d_same < 0.5 * d_diff, (d_same, d_diff)


def test_cka_states_shape():
    A = RNG.standard_normal((4, 60, 16))
    assert abs(cka_states(A, A, skip=10) - 1.0) < 1e-9
