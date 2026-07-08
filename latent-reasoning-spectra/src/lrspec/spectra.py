"""Spectral invariants of step Jacobians (PLAN.md sec. 4, E0).

All invariants are computed on a 768x768 real matrix J (numpy float64):
  - eigenspectrum lambda_i (complex), spectral radius rho = max |lambda_i|
  - sigma_1 (top singular value): transient expansion  [PRIMARY branch predictor]
  - n_expanding = #{sigma_i > 1}
  - Henrici departure from normality dep_F = sqrt(max(0, ||J||_F^2 - sum |lambda_i|^2)),
    plus normalized dep_F / ||J||_F
  - kappa = sigma_1 / rho (non-normal amplification ratio)
  - near-unit spectral mass = sum_{|lambda_i| in [0.9, 1.1]} |lambda_i|
    (RKSP-style diagnostic)  [PRIMARY anchor predictor]
  - top singular directions u_1, v_1
"""

from __future__ import annotations

import numpy as np


def invariants(J: np.ndarray, unit_band: tuple[float, float] = (0.9, 1.1)) -> dict:
    J = np.asarray(J, dtype=np.float64)
    assert J.ndim == 2 and J.shape[0] == J.shape[1]
    eig = np.linalg.eigvals(J)
    abs_eig = np.abs(eig)
    rho = float(abs_eig.max())
    fro2 = float((J * J).sum())
    dep2 = max(0.0, fro2 - float((abs_eig ** 2).sum()))
    U, S, Vt = np.linalg.svd(J)
    sigma1 = float(S[0])
    lo, hi = unit_band
    band = (abs_eig >= lo) & (abs_eig <= hi)
    return {
        "rho": rho,
        "sigma1": sigma1,
        "sigma2": float(S[1]),
        "n_expanding": int((S > 1.0).sum()),
        "henrici": float(np.sqrt(dep2)),
        "henrici_norm": float(np.sqrt(dep2 / fro2)) if fro2 > 0 else 0.0,
        "kappa": sigma1 / rho if rho > 0 else np.inf,
        "unit_mass": float(abs_eig[band].sum()),
        "n_unit_band": int(band.sum()),
        "trace": float(np.trace(J)),
        "fro": float(np.sqrt(fro2)),
        "eig_abs_sorted": np.sort(abs_eig)[::-1][:16].tolist(),  # top 16 for storage
        "u1": U[:, 0].astype(np.float32),   # top left singular vector
        "v1": Vt[0, :].astype(np.float32),  # top right singular vector (input direction)
    }


def scalar_invariants(inv: dict) -> dict:
    """Drop vector-valued entries (for JSON storage)."""
    return {k: v for k, v in inv.items() if k not in ("u1", "v1")}


def top_eig_subspace(J: np.ndarray, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Top-k eigenvalues (by modulus) and a REAL orthonormal basis spanning the
    corresponding eigenvector directions (conjugate pairs realified, then QR)."""
    J = np.asarray(J, dtype=np.float64)
    w, V = np.linalg.eig(J)
    order = np.argsort(-np.abs(w))
    cols: list[np.ndarray] = []
    used: set[int] = set()
    for i in order:
        if len(cols) >= k:
            break
        if i in used:
            continue
        used.add(i)
        v = V[:, i]
        if np.abs(w[i].imag) > 1e-12:
            cols.append(v.real)
            cols.append(v.imag)
            # mark the conjugate partner as used (nearest conjugate eigenvalue)
            j = int(np.argmin(np.abs(w - np.conj(w[i])) + np.array(
                [1e9 if m in used else 0.0 for m in range(len(w))])))
            used.add(j)
        else:
            cols.append(v.real)
    M = np.stack(cols, axis=1)
    Q, _ = np.linalg.qr(M)
    return w[order[:k]], Q
