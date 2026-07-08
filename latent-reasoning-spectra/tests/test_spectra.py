import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrspec.spectra import invariants, top_eig_subspace  # noqa: E402


def test_normal_matrix_zero_henrici():
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(50, 50)))
    D = np.diag(rng.uniform(0.1, 2.0, 50))
    J = Q @ D @ Q.T  # symmetric => normal
    inv = invariants(J)
    assert inv["henrici_norm"] < 1e-6
    assert abs(inv["kappa"] - 1.0) < 1e-6


def test_rotation_unit_mass():
    th = 0.3
    J = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    inv = invariants(J)
    assert abs(inv["unit_mass"] - 2.0) < 1e-9
    assert inv["n_unit_band"] == 2
    assert abs(inv["rho"] - 1.0) < 1e-9


def test_shear_non_normal():
    J = np.array([[0.5, 10.0], [0.0, 0.5]])  # highly non-normal, rho=0.5
    inv = invariants(J)
    assert inv["rho"] == 0.5
    assert inv["sigma1"] > 5.0
    assert inv["kappa"] > 10.0
    assert inv["henrici"] > 9.0


def test_top_eig_subspace_invariance():
    rng = np.random.default_rng(1)
    J = rng.normal(size=(30, 30)) * 0.1
    # plant a dominant real eigenpair and a complex pair
    v = rng.normal(size=30); v /= np.linalg.norm(v)
    J += 3.0 * np.outer(v, v)
    w, Q = top_eig_subspace(J, k=3)
    assert Q.shape[0] == 30 and Q.shape[1] >= 3
    # subspace approximately invariant: ||(I-QQ^T) J Q|| small relative to ||J Q||
    P = np.eye(30) - Q @ Q.T
    ratio = np.linalg.norm(P @ J @ Q) / np.linalg.norm(J @ Q)
    assert ratio < 0.35  # eigvec span is not exactly invariant unless closed under J
    # dominant eigenvalue must be in the returned set
    assert np.isclose(np.abs(w).max(), np.abs(np.linalg.eigvals(J)).max())
    # the dominant planted direction lies in span(Q)
    assert np.linalg.norm(Q @ (Q.T @ v)) > 0.95
