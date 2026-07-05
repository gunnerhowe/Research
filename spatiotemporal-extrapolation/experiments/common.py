"""Shared measurement machinery for all experiments (PLAN.md conventions)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from specext.ks import ks_stream_batch                     # noqa: E402
from specext.edmd import (ModeGrid, SpectrumAccum, WelchAccum, HankelAccum,     # noqa: E402
                          edmd_eig, leading_resonance, tau_e_from_autocorr,
                          pick_stride, edmd_autocorr_r2, omega_peak_from_welch)

RESULTS = ROOT / "results"
RUNS = ROOT / "runs"
DATA = ROOT / "data"
for p in (RESULTS, RUNS, DATA):
    p.mkdir(exist_ok=True)

DX = 22.0 / 64
DT = 0.25
DT_S = 0.5
TRANSIENT = 500.0
K_MAX = 3.0
SEEDS = [0, 1, 2]
SIZES_TRAIN = [22.0, 44.0, 66.0, 88.0]
L_HOLDOUT = 176.0
L_TARGET = 1408.0
STRIDES = [1, 8, 64, 512]
D_DELAYS = 8
WELCH_BLOCK = 4096
WELCH_BLOCK_LONG = 65536
K_LOW = 0.15               # long-Welch subset
K_GATE_MAX = 2.2
K_FULL_LO = 2 * np.pi / 22.0        # smallest wavenumber present at ALL train sizes
K_INRANGE_LO = 2 * np.pi / 88.0     # smallest wavenumber present at ANY train size
# headline band = full 4-size support [2pi/22, 2.2]; partial-support band
# [2pi/88, 2pi/22) and new-mode band (k < 2pi/88) are reported separately.
COMMON_M22 = np.arange(1, 8)        # k = 2 pi m / 22, m = 1..7 (k <= 2.0)


def n_of(L):
    n = 64 * L / 22.0
    assert abs(n - round(n)) < 1e-9
    return int(round(n))


def t_sim(L):
    if L <= 88:
        return 60_000.0
    if L <= 704:
        return 100_000.0
    if L <= 1408:
        return 200_000.0
    return 100_000.0


def measure_size(L, T=None, seeds=SEEDS, bc="periodic", store_fields=False,
                 tag=None, log=print, system="ks", sys_kwargs=None):
    """Simulate the chosen PDE at (L, N(L)) for all seeds in lockstep and
    accumulate the streaming estimators. system='ks' (default, Kuramoto-
    Sivashinsky) or 'nik' (Nikolaevskiy; sys_kwargs may set r, dt, transient).
    Saves runs/measure_{tag}.npz and returns its path. analyze_measurement is
    system-agnostic (it consumes the mode series)."""
    L = float(L)
    N = n_of(L)
    T = t_sim(L) if T is None else float(T)
    tag = tag or f"L{L:g}" + ("" if bc == "periodic" else f"_{bc}")
    n_samples = int(round(T / DT_S))
    grid = ModeGrid(L, N, K_MAX)
    S = len(seeds)
    spec = SpectrumAccum(S, N)
    wel = WelchAccum(WELCH_BLOCK, S, len(grid.m_idx))
    low_sel = np.nonzero(grid.k < K_LOW)[0]
    wel_long = (WelchAccum(WELCH_BLOCK_LONG, S, len(grid.m_idx), mode_sel=low_sel)
                if len(low_sel) else None)
    han = HankelAccum(STRIDES, D_DELAYS, S, len(grid.m_idx))
    mms = None
    if store_fields:
        mms = [np.lib.format.open_memmap(DATA / f"{tag}_s{seed}.npy", mode="w+",
                                         dtype=np.float32, shape=(n_samples, N))
               for seed in seeds]
    snippet = []
    chunk = 2048 if N >= 2048 else 4096
    sys_kwargs = dict(sys_kwargs or {})
    if system == "ks":
        stream = ks_stream_batch(L, N, n_samples, seeds, dt=DT, dt_sample=DT_S,
                                 transient=TRANSIENT, bc=bc, chunk_samples=chunk)
    elif system == "nik":
        from specext.nikolaevskiy import nik_stream_batch
        if bc != "periodic":
            raise ValueError("nik: periodic only")
        stream = nik_stream_batch(L, N, n_samples, seeds, dt_sample=DT_S,
                                  chunk_samples=chunk, **sys_kwargs)
    else:
        raise ValueError(f"unknown system {system!r}")
    t0 = time.time()
    written = 0
    for field in stream:
        modes_full = np.fft.rfft(field, axis=-1) / N
        spec.add_modes(modes_full)
        modes = np.ascontiguousarray(modes_full[:, :, grid.m_idx])
        wel.add(modes)
        if wel_long is not None:
            wel_long.add(modes)
        han.add(modes)
        if mms is not None:
            for si in range(S):
                mms[si][written:written + field.shape[0]] = field[:, si]
        if written < 2000:
            snippet.append(field[:min(2000 - written, field.shape[0]), 0])
        written += field.shape[0]
    if mms is not None:
        for m in mms:
            m.flush()
    wall = time.time() - t0
    out = {"L": L, "N": N, "T": T, "seeds": np.asarray(seeds), "bc": bc,
           "k": grid.k, "m_idx": grid.m_idx, "dt_s": DT_S, "wall_s": wall,
           "p_mean": spec.power(),
           "welch_psum": wel.p_sum, "welch_nblocks": wel.n_blocks,
           "snippet": np.concatenate(snippet).astype(np.float32),
           "npairs": np.asarray([han.npairs[s] for s in STRIDES]),
           "strides": np.asarray(STRIDES)}
    for s in STRIDES:
        out[f"gz_{s}"] = han.gz[s]
    if wel_long is not None and wel_long.n_blocks > 0:
        out["welch_long_psum"] = wel_long.p_sum
        out["welch_long_nblocks"] = wel_long.n_blocks
        out["welch_long_sel"] = low_sel
    path = RUNS / f"measure_{tag}.npz"
    np.savez_compressed(path, **out)
    log(f"measured L={L:g} bc={bc} N={N} T={T:g} ({S} seeds) in {wall:.0f}s -> {path.name}")
    return path


def analyze_measurement(path, r2_tmax_mult=6.0):
    """Per-seed per-mode leading resonances and statistics from a measurement npz.

    Returns dict with k (M,), and (S, M) arrays: gamma, omega, s_density, tau_acf,
    omega_pk, r2, stride, weight_frac.
    """
    z = np.load(path, allow_pickle=False)
    k = z["k"]
    m_idx = z["m_idx"]
    L = float(z["L"])
    S = len(z["seeds"])
    M = len(k)
    c = WelchAccumFromSums(z["welch_psum"], int(z["welch_nblocks"]), WELCH_BLOCK)
    acf = c.autocorr(max_lag=WELCH_BLOCK // 2)
    tau, censored = tau_e_from_autocorr(acf, DT_S)
    om_pk = omega_peak_from_welch(z["welch_psum"], WELCH_BLOCK, DT_S)
    if "welch_long_psum" in z.files:
        cl = WelchAccumFromSums(z["welch_long_psum"], int(z["welch_long_nblocks"]),
                                WELCH_BLOCK_LONG)
        acf_l = cl.autocorr(max_lag=WELCH_BLOCK_LONG // 2)
        tau_l, _ = tau_e_from_autocorr(acf_l, DT_S)
        om_l = omega_peak_from_welch(z["welch_long_psum"], WELCH_BLOCK_LONG, DT_S)
        sel = z["welch_long_sel"]
        tau[:, sel] = tau_l
        om_pk[:, sel] = om_l
    tau_mean = tau.mean(axis=0)
    om_mean = om_pk.mean(axis=0)
    npairs = {s: n for s, n in zip(z["strides"].tolist(), z["npairs"].tolist())}
    # only strides with enough pairs for a stable Gram (short-trajectory safe)
    avail = [s for s in STRIDES if npairs[s] >= 50 * (D_DELAYS + 1)]
    if not avail:
        avail = [min(STRIDES)]
    stride_sel = pick_stride(tau_mean, om_mean, avail, D_DELAYS, DT_S)
    gz = {s: z[f"gz_{s}"] for s in STRIDES}
    gamma = np.full((S, M), np.nan)
    omega = np.full((S, M), np.nan)
    r2 = np.full((S, M), np.nan)
    wfrac = np.full((S, M), np.nan)
    for mi in range(M):
        s = int(stride_sel[mi])
        for si in range(S):
            mu, w = edmd_eig(gz[s][si, mi], npairs[s])
            lam, frac = leading_resonance(mu, w, s * DT_S)
            gamma[si, mi] = -lam.real
            omega[si, mi] = abs(lam.imag)
            wfrac[si, mi] = frac
            use_long = ("welch_long_psum" in z.files and
                        mi in set(z["welch_long_sel"].tolist()))
            if use_long:
                sel_l = z["welch_long_sel"].tolist().index(mi)
                cme = acf_l[:, si, sel_l]
            else:
                cme = acf[:, si, mi]
            r2[si, mi] = edmd_autocorr_r2(gz[s][si, mi], npairs[s], s, cme, DT_S,
                                          t_max=min(r2_tmax_mult * tau_mean[mi],
                                                    0.45 * (WELCH_BLOCK_LONG if use_long
                                                            else WELCH_BLOCK) * DT_S))
    p_ret = z["p_mean"][:, m_idx]
    s_density = p_ret / (2 * np.pi / L)
    return {"k": k, "L": L, "gamma": gamma, "omega": omega, "s_density": s_density,
            "tau_acf": tau, "omega_pk": om_pk, "r2": r2, "stride": stride_sel,
            "weight_frac": wfrac, "p_mean": z["p_mean"], "N": int(z["N"]),
            "wall_s": float(z["wall_s"])}


def resonances_from_modeseries(modes, dt_s=DT_S):
    """Leading per-mode resonances from a complex mode series (T, n_modes) using
    the SAME estimator pipeline as analyze_measurement (Hankel-EDMD + Welch stride
    selection). Returns (gamma, omega, r2) each (n_modes,). Single realization."""
    T, Mret = modes.shape
    x = np.ascontiguousarray(modes[:, None, :])          # (T, 1 seed, M)
    han = HankelAccum(STRIDES, D_DELAYS, 1, Mret)
    wel = WelchAccum(WELCH_BLOCK, 1, Mret)
    for i0 in range(0, T, 4096):
        ch = x[i0:i0 + 4096]
        han.add(ch)
        wel.add(np.ascontiguousarray(ch))
    acf = wel.autocorr(max_lag=WELCH_BLOCK // 2)
    tau, _ = tau_e_from_autocorr(acf, dt_s)
    om = omega_peak_from_welch(wel.p_sum, WELCH_BLOCK, dt_s)
    npairs = {s: han.npairs[s] for s in STRIDES}
    avail = [s for s in STRIDES if npairs[s] >= 50 * (D_DELAYS + 1)] or [min(STRIDES)]
    stride = pick_stride(tau.mean(0), om.mean(0), avail, D_DELAYS, dt_s)
    gamma = np.full(Mret, np.nan)
    omega = np.full(Mret, np.nan)
    r2 = np.full(Mret, np.nan)
    for mi in range(Mret):
        s = int(stride[mi])
        mu, w = edmd_eig(han.gz[s][0, mi], npairs[s])
        lam, _ = leading_resonance(mu, w, s * dt_s)
        gamma[mi] = -lam.real
        omega[mi] = abs(lam.imag)
        r2[mi] = edmd_autocorr_r2(han.gz[s][0, mi], npairs[s], s, acf[:, 0, mi],
                                  dt_s, t_max=min(6 * tau.mean(0)[mi],
                                                  0.45 * WELCH_BLOCK * dt_s))
    return gamma, omega, r2


class WelchAccumFromSums:
    """Autocorr extraction from saved Welch sums (mirrors WelchAccum.autocorr)."""

    def __init__(self, p_sum, n_blocks, block):
        self.p_sum, self.n_blocks, self.block = p_sum, n_blocks, block
        self.win = np.hanning(block)

    def autocorr(self, max_lag):
        c_raw = np.fft.ifft(self.p_sum / self.n_blocks, axis=0)[:max_lag + 1]
        rw = np.correlate(self.win, self.win, mode="full")[self.block - 1:
                                                           self.block - 1 + max_lag + 1]
        return c_raw / rw[:, None, None]


def common_grid_indices(k, m22=COMMON_M22):
    """Indices in a retained-k array matching k = 2 pi m / 22 (m in m22)."""
    kc = 2 * np.pi * np.asarray(m22) / 22.0
    idx = []
    for kk in kc:
        j = np.argmin(np.abs(k - kk))
        assert abs(k[j] - kk) < 1e-9, "common-grid wavenumber missing"
        idx.append(j)
    return np.asarray(idx), kc


def save_json(name, payload):
    path = RESULTS / name
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=_np_default)
    print(f"wrote {path}")
    return path


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")
