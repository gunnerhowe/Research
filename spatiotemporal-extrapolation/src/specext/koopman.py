"""Translation-equivariant neural Koopman/transfer operator with a size-conditioned
linear propagator (the learned finite-size-scaling flow).

Architecture (PLAN.md):
  encoder  phi: circular Conv1d stack 1 -> 64 -> 64 -> M (GELU, kernel 9)
  propagator: LINEAR bias-free circular Conv1d with kernel W(ell) = W0 + ell*W1,
              ell = 22/L (per-size models use W0 only), kernel width 33
  decoder  : circular Conv1d stack M -> 64 -> 64 -> 1

Because every block is translation-equivariant, the propagator block-diagonalizes
over lattice wavenumbers kappa = k*dx: K_hat(kappa; ell) = sum_j W_j e^{i kappa (j-c)}
is an M x M complex matrix per sector whose eigenvalues are the learned resonances
mu(kappa); lambda = log(mu)/dt_s. Finite kernel support makes K_hat analytic in
kappa, so evaluating on the refined grid of a LARGER domain (and at extrapolated
ell) is exact — that evaluation is the domain-extension prediction.

Sector "leading" eigenvalue selection is data-free: input-output resonance weights
|D_hat(kappa) V_j| * |(V^-1 E_hat(kappa))_j| from the linearized encoder/decoder
frequency responses (validated against eigencoordinate/data correlations at the
training sizes).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

M_LATENT = 16
K_ENC = 9
K_PROP = 33
DT_STEP = 0.5   # one propagator application = one sample interval
L_BASE = 22.0


def _cconv(ch_in, ch_out, k):
    return nn.Conv1d(ch_in, ch_out, k, padding=k // 2, padding_mode="circular")


class ConvKoopman(nn.Module):
    def __init__(self, m_latent=M_LATENT, width=64, flow=False):
        super().__init__()
        self.flow = flow
        self.enc = nn.Sequential(_cconv(1, width, K_ENC), nn.GELU(),
                                 _cconv(width, width, K_ENC), nn.GELU(),
                                 _cconv(width, m_latent, K_ENC))
        self.dec = nn.Sequential(_cconv(m_latent, width, K_ENC), nn.GELU(),
                                 _cconv(width, width, K_ENC), nn.GELU(),
                                 _cconv(width, 1, K_ENC))
        w0 = torch.zeros(m_latent, m_latent, K_PROP)
        w0[torch.arange(m_latent), torch.arange(m_latent), K_PROP // 2] = 1.0
        w0 += 0.01 * torch.randn_like(w0)
        self.w0 = nn.Parameter(w0)
        self.w1 = nn.Parameter(torch.zeros(m_latent, m_latent, K_PROP)) if flow else None

    def kernel(self, ell):
        return self.w0 + ell * self.w1 if self.flow else self.w0

    def encode(self, u):     # (B, N) -> (B, M, N)
        return self.enc(u.unsqueeze(1))

    def decode(self, z):     # (B, M, N) -> (B, N)
        return self.dec(z).squeeze(1)

    def propagate(self, z, ell):
        w = self.kernel(ell)
        zp = torch.nn.functional.pad(z, (K_PROP // 2, K_PROP // 2), mode="circular")
        return torch.nn.functional.conv1d(zp, w)


def train_model(datasets, seed, flow=False, steps=12000, batch=64, horizon=16,
                lr=1e-3, device="cuda", log_every=2000, log_fn=print):
    """datasets: list of dicts {"L": float, "data": float32 memmap (T, N)} (one per
    size for the flow model, a single entry for per-size models)."""
    torch.manual_seed(seed)
    model = ConvKoopman(flow=flow).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 100)
    rng = np.random.default_rng(seed)
    tensors = []
    for d in datasets:
        tensors.append({"L": d["L"], "ell": L_BASE / d["L"], "data": d["data"]})
    t0 = time.time()
    model.train()
    for step in range(steps):
        d = tensors[step % len(tensors)]
        T = d["data"].shape[0]
        idx = rng.integers(0, T - horizon - 1, size=batch)
        seq = np.stack([d["data"][i:i + horizon + 1] for i in idx])  # (B, h+1, N)
        u = torch.from_numpy(seq).to(device).float()
        B, H1, N = u.shape
        z_all = model.encode(u.reshape(B * H1, N)).reshape(B, H1, -1, N)
        u_rec = model.decode(z_all.reshape(B * H1, -1, N)).reshape(B, H1, N)
        loss = torch.mean((u_rec - u) ** 2)
        z = z_all[:, 0]
        for m in range(1, H1):
            z = model.propagate(z, d["ell"])
            loss = loss + torch.mean((z - z_all[:, m]) ** 2) / horizon
            loss = loss + torch.mean((model.decode(z) - u[:, m]) ** 2) / horizon
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if log_every and (step + 1) % log_every == 0:
            log_fn(f"  step {step+1}/{steps} loss {loss.item():.5f}")
    return model, time.time() - t0


# ------------------------------------------------------- spectrum extraction

def freq_response(w, kappa):
    """K_hat(kappa)[a, b] = sum_j w[a, b, j] exp(i kappa (j - c)).
    w: (M, M, kw) numpy; kappa: (nk,). Returns (nk, M, M) complex."""
    kw = w.shape[-1]
    j = np.arange(kw) - kw // 2
    phase = np.exp(1j * np.outer(kappa, j))            # (nk, kw)
    return np.einsum("abj,nj->nab", w, phase)


@torch.no_grad()
def _sector_response(fn, base_in, kappa_idx, n_ch, N, device, eps=1e-3, batch=256):
    """Linear frequency response of a translation-equivariant map at lattice bins.

    fn maps (B, C_in, N) -> (B, C_out, N). base_in: (C_in, N). kappa_idx: rfft bin
    indices. Returns (n_kappa, C_out, C_in) complex response matrices.
    Perturbations are built lazily per batch (an upfront list is O(GB) at large N).
    """
    x = np.arange(N)
    out0 = fn(base_in.unsqueeze(0)).squeeze(0)          # (C_out, N)
    n_out = out0.shape[0]
    resp = np.zeros((len(kappa_idx), n_out, n_ch), dtype=np.complex128)
    specs = [(qi, a, ph) for qi in range(len(kappa_idx))
             for a in range(n_ch) for ph in (0, 1)]
    for i0 in range(0, len(specs), batch):
        chunk = specs[i0:i0 + batch]
        xb = base_in.unsqueeze(0).repeat(len(chunk), 1, 1)
        for bi, (qi, a, ph) in enumerate(chunk):
            kap = 2 * np.pi * kappa_idx[qi] / N
            f = np.cos if ph == 0 else np.sin
            xb[bi, a] += torch.from_numpy(eps * f(kap * x)).to(device).float()
        yb = fn(xb) - out0.unsqueeze(0)                 # (b, C_out, N)
        Y = torch.fft.rfft(yb, dim=-1).cpu().numpy() / eps
        for (qi, a, ph), y in zip(chunk, Y):
            v = 2.0 * y[:, kappa_idx[qi]] / N           # rfft -> amplitude of e^{i kap x}
            resp[qi, :, a] += v if ph == 0 else 1j * v
        del xb, yb
    return resp / 2.0   # cos+i sin response = response to e^{i kap x}


@torch.no_grad()
def model_spectrum(model, ell, m_idx, N, device="cuda", n_half=8, mu_clip=0.999):
    """Per-sector eigenvalues + leading resonance from the operator's own implied
    stationary autocovariance (data-free, valid at any kappa and ell).

    The observable-sector autocovariance implied by the learned stochastic
    operator under isotropic sector forcing is c(n) = D_hat K^n Sigma_iso D_hat^H
    with Sigma_iso = sum_{n>=0} K^n K^dagger^n; its poles are the eigenvalues
    mu_j of K_hat(kappa) with amplitudes alpha_j = (D_hat V)_j (V^-1 Sigma_iso
    D_hat^H)_j. Leading resonance = argmax |alpha_j| |mu_j|^n_half among stable
    mu — the same persistence-weighted rule as the EDMD extractor. m_idx: rfft
    bin indices on an N-point grid (kappa = 2 pi m / N; physical k = kappa/dx).
    """
    model.eval()
    w = model.kernel(ell).detach().cpu().numpy().astype(np.float64)
    kappa = 2 * np.pi * np.asarray(m_idx) / N
    K = freq_response(w, kappa)                          # (nk, M, M)
    mu, V = np.linalg.eig(K)
    # stabilized copy for the covariance sum (clip |mu| at mu_clip)
    scale = np.minimum(1.0, mu_clip / np.maximum(np.abs(mu), 1e-12))
    Vinv = np.linalg.inv(V)
    Kc = np.einsum("nij,nj,njk->nik", V, mu * scale, Vinv)
    sig = np.tile(np.eye(K.shape[1]), (len(kappa), 1, 1)).astype(np.complex128)
    A = Kc.copy()
    for _ in range(13):                                   # sum of 2^13 terms
        sig = sig + A @ sig @ np.conj(A.transpose(0, 2, 1))
        A = A @ A
    zbar = model.encode(torch.zeros(1, N, device=device)).squeeze(0)
    D = _sector_response(lambda x: model.dec(x), zbar, list(m_idx),
                         zbar.shape[0], N, device)        # (nk, 1, M)
    dv = np.einsum("nim,nmj->nj", D, V)                   # (D V)_j
    vsd = np.einsum("njm,nmi->nj", Vinv,
                    sig @ np.conj(D.transpose(0, 2, 1)))  # (V^-1 Sig D^H)_j
    alpha = dv * vsd
    weights = np.abs(alpha) * np.abs(mu) ** n_half
    lam = np.full(len(m_idx), np.nan, dtype=np.complex128)
    mu_sel = np.full(len(m_idx), np.nan, dtype=np.complex128)
    for i in range(len(m_idx)):
        ok = np.abs(mu[i]) <= 1.005
        if not ok.any():
            continue
        j = np.argmax(np.where(ok, weights[i], -np.inf))
        mu_sel[i] = mu[i][j]
        lam[i] = np.log(mu[i][j]) / DT_STEP
    return {"mu": mu, "weights": weights, "mu_sel": mu_sel, "lam": lam,
            "kappa": kappa}


@torch.no_grad()
def sector_stationary(model, data, ell, N, device="cuda", n_eval=20000, batch=256):
    """Per-sector residual noise covariance and stationary latent covariance from
    data at a TRAINING size (E1 generative check): returns Sigma_n, Sigma_z
    (n_sectors_all, M, M) for all rfft bins 1..N//2-1, plus encoded sector series
    statistics."""
    model.eval()
    T = data.shape[0]
    idx = np.linspace(0, T - 2, n_eval).astype(int)
    zs, zs1 = [], []
    for i0 in range(0, len(idx), batch):
        ii = idx[i0:i0 + batch]
        u0 = torch.from_numpy(np.ascontiguousarray(data[ii])).to(device).float()
        u1 = torch.from_numpy(np.ascontiguousarray(data[ii + 1])).to(device).float()
        zs.append(torch.fft.rfft(model.encode(u0), dim=-1).cpu().numpy() / N)
        zs1.append(torch.fft.rfft(model.encode(u1), dim=-1).cpu().numpy() / N)
    z0 = np.concatenate(zs)     # (T', M, N//2+1)
    z1 = np.concatenate(zs1)
    m_all = np.arange(1, N // 2)
    kappa = 2 * np.pi * m_all / N
    w = model.kernel(ell).detach().cpu().numpy().astype(np.float64)
    K = freq_response(w, kappa)
    sig_n = np.zeros((len(m_all), K.shape[1], K.shape[1]), dtype=np.complex128)
    sig_z = np.zeros_like(sig_n)
    for i, m in enumerate(m_all):
        a0 = z0[:, :, m]
        a1 = z1[:, :, m]
        r = a1.T - K[i] @ a0.T
        sig_n[i] = (r @ r.conj().T) / a0.shape[0]
        s = a0.T @ a0.conj() / a0.shape[0]
        # stationary solve: Sigma = K Sigma K^dag + Sigma_n (iterate; contracts iff
        # the sector is stable — else keep the empirical covariance)
        if np.abs(np.linalg.eigvals(K[i])).max() < 0.995:
            sig = s.copy()
            for _ in range(500):
                sig = K[i] @ sig @ K[i].conj().T + sig_n[i]
            sig_z[i] = sig
        else:
            sig_z[i] = s
    return {"m_all": m_all, "sig_n": sig_n, "sig_z": sig_z}


@torch.no_grad()
def generate_stationary(model, sig_z, m_all, N, n_samples=4096, seed=0,
                        device="cuda", batch=512):
    """Sample latent sector Gaussians ~ CN(0, Sigma_z), decode, return decoded
    fields (n_samples, N) float32."""
    model.eval()
    rng = np.random.default_rng(seed)
    M = sig_z.shape[1]
    chol = np.zeros_like(sig_z)
    for i in range(sig_z.shape[0]):
        s = 0.5 * (sig_z[i] + sig_z[i].conj().T)
        ev = np.linalg.eigvalsh(s).min()
        jitter = max(0.0, -ev) + 1e-12 * max(1.0, np.trace(s).real / M)
        chol[i] = np.linalg.cholesky(s + jitter * np.eye(M))
    out = np.empty((n_samples, N), dtype=np.float32)
    for i0 in range(0, n_samples, batch):
        b = min(batch, n_samples - i0)
        xi = (rng.standard_normal((b, sig_z.shape[0], M)) +
              1j * rng.standard_normal((b, sig_z.shape[0], M))) / np.sqrt(2.0)
        amp = np.einsum("nij,bnj->bni", chol, xi)        # (b, n_sec, M)
        spec = np.zeros((b, M, N // 2 + 1), dtype=np.complex128)
        spec[:, :, m_all] = amp.transpose(0, 2, 1)
        z = np.fft.irfft(spec * N, n=N, axis=-1)         # real latent field
        zb = torch.from_numpy(z).to(device).float()
        u = model.decode(zb).cpu().numpy()
        out[i0:i0 + b] = u
    return out


@torch.no_grad()
def generate_and_reestimate(model, ell, m_idx, N, device="cuda", T=60000,
                            seed=0, mu_clip=0.999, chunk=4096):
    """Honest resonance-faithfulness probe (E1): drive the learned stochastic
    operator with isotropic per-sector white noise, decode to physical fields,
    and return the per-mode complex mode series (T, n_modes) so the CALLER can run
    the identical EDMD estimator it uses on real data. The sector operator's
    spectral radius is clipped to the unit disk (the trained operator has spurious
    |mu|>1 latent directions that would otherwise diverge the surrogate process).
    """
    model.eval()
    w = model.kernel(ell).detach().cpu().numpy().astype(np.float64)
    kappa = 2 * np.pi * np.asarray(m_idx) / N
    K = freq_response(w, kappa)
    Mlat = K.shape[1]
    mu, V = np.linalg.eig(K)
    Vinv = np.linalg.inv(V)
    scale = np.minimum(1.0, mu_clip / np.maximum(np.abs(mu), 1e-12))
    K = np.einsum("nij,nj,njk->nik", V, mu * scale, Vinv)
    Kt = torch.from_numpy(K).to(device).to(torch.complex64)
    midx_t = torch.as_tensor(np.asarray(m_idx), device=device)
    g = torch.Generator(device=device).manual_seed(int(seed) * 2654435761 % (1 << 31))
    z = torch.zeros(len(kappa), Mlat, dtype=torch.complex64, device=device)
    out = np.empty((T, len(m_idx)), dtype=np.complex128)
    buf = []
    written = 0
    for n in range(T):
        eta = (torch.randn(len(kappa), Mlat, device=device, generator=g) +
               1j * torch.randn(len(kappa), Mlat, device=device, generator=g)) / np.sqrt(2)
        z = torch.einsum("kij,kj->ki", Kt, z) + eta.to(torch.complex64)
        buf.append(z.clone())
        if len(buf) == chunk or n == T - 1:
            Z = torch.stack(buf)
            spec = torch.zeros(Z.shape[0], Mlat, N // 2 + 1, dtype=torch.complex64,
                               device=device)
            spec[:, :, midx_t] = Z.transpose(1, 2)
            zfield = torch.fft.irfft(spec * N, n=N, dim=-1).float()
            u = model.decode(zfield)
            uh = (torch.fft.rfft(u, dim=-1) / N).cpu().numpy()[:, m_idx]
            out[written:written + uh.shape[0]] = uh
            written += uh.shape[0]
            buf = []
    return out


@torch.no_grad()
def eigcoord_corr_selection(model, data, ell, m_idx, N, device="cuda",
                            n_eval=20000, batch=256):
    """Data-based leading-eigenvalue selection (validation of the io rule):
    for each retained sector, the eigen-coordinate with max |corr| against the
    physical mode series. Returns lam (nk,) complex."""
    model.eval()
    T = data.shape[0]
    idx = np.linspace(0, T - 1, n_eval).astype(int)
    zs, us = [], []
    for i0 in range(0, len(idx), batch):
        ii = idx[i0:i0 + batch]
        u = torch.from_numpy(np.ascontiguousarray(data[ii])).to(device).float()
        zs.append(torch.fft.rfft(model.encode(u), dim=-1).cpu().numpy() / N)
        us.append(np.fft.rfft(data[ii], axis=-1) / N)
    z = np.concatenate(zs)      # (T', M, nbins)
    uh = np.concatenate(us)     # (T', nbins)
    kappa = 2 * np.pi * np.asarray(m_idx) / N
    w = model.kernel(ell).detach().cpu().numpy().astype(np.float64)
    K = freq_response(w, kappa)
    mu, V = np.linalg.eig(K)
    lam = np.full(len(m_idx), np.nan, dtype=np.complex128)
    for i, m in enumerate(m_idx):
        wc = np.linalg.solve(V[i], z[:, :, m].T)         # (M, T') eigencoords
        um = uh[:, m]
        um = um - um.mean()
        cn = np.abs((wc - wc.mean(axis=1, keepdims=True)) @ um.conj())
        den = np.sqrt((np.abs(wc - wc.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
                      * (np.abs(um) ** 2).sum())
        corr = cn / np.maximum(den, 1e-300)
        ok = np.abs(mu[i]) <= 1.005
        if not ok.any():
            continue
        j = np.argmax(corr * ok)
        lam[i] = np.log(mu[i][j]) / DT_STEP
    return lam
