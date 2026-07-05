"""EDMD/Welch estimators against an analytic linear case: independent complex OU
sectors with known resonances lambda(k) = -nu k^2 - i c k (stochastically forced
advection-diffusion), exactly discretized."""
import numpy as np

from specext.edmd import (HankelAccum, WelchAccum, SpectrumAccum, edmd_eig,
                          leading_resonance, tau_e_from_autocorr, pick_stride,
                          edmd_autocorr_r2, omega_peak_from_welch)

DT_S = 0.5
NU, C = 0.08, 0.7
KS = np.array([0.3, 0.6, 1.2])
LAM = -NU * KS ** 2 - 1j * C * KS


def _ou_modes(T, seeds, rng_offset=0):
    a = np.exp(LAM * DT_S)
    sig_stat = np.array([1.0, 0.5, 0.2])
    out = np.empty((T, len(seeds), len(KS)), dtype=np.complex128)
    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(seed + rng_offset)
        x = np.zeros(len(KS), dtype=np.complex128)
        drive = sig_stat * np.sqrt(1 - np.abs(a) ** 2)
        for t in range(T):
            xi = (rng.standard_normal(len(KS)) + 1j * rng.standard_normal(len(KS))) / np.sqrt(2)
            x = a * x + drive * xi
            out[t, si] = x
    return out


def test_hankel_edmd_recovers_linear_resonances():
    T = 120_000
    modes = _ou_modes(T, seeds=[0, 1])
    han = HankelAccum(strides=[1, 8], d=8, n_seeds=2, n_modes=len(KS))
    wel = WelchAccum(block=2048, n_seeds=2, n_modes=len(KS))
    for i0 in range(0, T, 4096):
        chunk = modes[i0:i0 + 4096]
        han.add(chunk)
        wel.add(chunk)
    c = wel.autocorr(max_lag=512)
    tau, _ = tau_e_from_autocorr(c, DT_S)
    om = omega_peak_from_welch(wel.p_sum, wel.block, DT_S)
    for mi, k in enumerate(KS):
        s = pick_stride(tau[:, mi].mean(), om[:, mi].mean(), [1, 8], 8, DT_S)[0]
        assert om[:, mi].mean() * s * DT_S <= 0.8 * np.pi  # no aliased branch
        errs = []
        for seed in range(2):
            mu, w = edmd_eig(han.gz[s][seed, mi], han.npairs[s])
            lam, frac = leading_resonance(mu, w, s * DT_S)
            errs.append(abs(lam - LAM[mi]) / abs(LAM[mi]))
            r2 = edmd_autocorr_r2(han.gz[s][seed, mi], han.npairs[s], s,
                                  c[:, seed, mi], DT_S, t_max=50.0)
            assert r2 > 0.95, f"k={k} seed={seed}: r2={r2:.3f}"
        assert np.mean(errs) < 0.05, f"k={k}: rel err {np.mean(errs):.3f}"


def test_welch_autocorr_matches_analytic():
    T = 120_000
    modes = _ou_modes(T, seeds=[5], rng_offset=100)
    wel = WelchAccum(block=2048, n_seeds=1, n_modes=len(KS))
    for i0 in range(0, T, 8192):
        wel.add(modes[i0:i0 + 8192])
    c = wel.autocorr(max_lag=100)
    t = np.arange(101) * DT_S
    for mi in range(len(KS)):
        rho_est = c[:, 0, mi] / c[0, 0, mi]
        rho_true = np.exp(LAM[mi] * t)
        assert np.abs(rho_est - rho_true).max() < 0.08


def test_spectrum_accum_single_cosine():
    N, L = 64, 22.0
    x = L * np.arange(N) / N
    amp, m = 0.8, 3
    field = amp * np.cos(2 * np.pi * m * x / L)[None, None, :] * np.ones((10, 2, 1))
    acc = SpectrumAccum(n_seeds=2, N=N)
    acc.add_modes(np.fft.rfft(field, axis=-1) / N)
    p = acc.power()
    assert np.allclose(p[:, m], amp ** 2 / 2, rtol=1e-12)
    assert np.allclose(np.delete(p, m, axis=1), 0.0, atol=1e-15)
    # Parseval: sum P = Var
    assert np.allclose(p.sum(axis=1), field.var(axis=(0, 2)), rtol=1e-10)
