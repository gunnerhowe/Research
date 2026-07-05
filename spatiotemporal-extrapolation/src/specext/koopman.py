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


def train_model(datasets, seed, flow=False, steps=20000, batch=32, horizon=8,
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
    indices. Returns (n_kappa, C_out, C_in) complex response matrices."""
    x = np.arange(N)
    out0 = fn(base_in.unsqueeze(0)).squeeze(0)          # (C_out, N)
    n_out = out0.shape[0]
    resp = np.zeros((len(kappa_idx), n_out, n_ch), dtype=np.complex128)
    perts = []
    for qi, mi in enumerate(kappa_idx):
        kap = 2 * np.pi * mi / N
        for a in range(n_ch):
            for ph, f in ((0, np.cos), (1, np.sin)):
                p = torch.zeros(n_ch, N, device=device)
                p[a] = torch.from_numpy(eps * f(kap * x)).to(device).float()
                perts.append((qi, a, ph, p))
    for i0 in range(0, len(perts), batch):
        chunk = perts[i0:i0 + batch]
        xb = torch.stack([base_in + p for (_, _, _, p) in chunk])
        yb = fn(xb) - out0.unsqueeze(0)                 # (b, C_out, N)
        Y = torch.fft.rfft(yb, dim=-1).cpu().numpy() / eps
        for (qi, a, ph, _), y in zip(chunk, Y):
            v = 2.0 * y[:, kappa_idx[qi]] / N           # rfft -> amplitude of e^{i kap x}
            resp[qi, :, a] += v if ph == 0 else 1j * v
        del xb, yb
    return resp / 2.0   # cos+i sin response = response to e^{i kap x}


@torch.no_grad()
def model_spectrum(model, ell, m_idx, N, device="cuda"):
    """Per-sector eigenvalues + io-weight-selected leading resonance.

    m_idx: rfft bin indices of the retained sectors on an N-point grid (kappa =
    2 pi m / N; physical k = kappa/dx). Returns dict with mu (nk, M), lam (nk,),
    weights (nk, M)."""
    model.eval()
    w = model.kernel(ell).detach().cpu().numpy().astype(np.float64)
    kappa = 2 * np.pi * np.asarray(m_idx) / N
    K = freq_response(w, kappa)                          # (nk, M, M)
    mu, V = np.linalg.eig(K)
    base_u = torch.zeros(1, N, device=device)
    enc_fn = lambda x: model.enc(x)
    dec_fn = lambda x: model.dec(x)
    E = _sector_response(enc_fn, base_u, list(m_idx), 1, N, device)   # (nk, M, 1)
    zbar = model.encode(base_u).squeeze(0)               # (M, N)
    D = _sector_response(dec_fn, zbar, list(m_idx), zbar.shape[0], N, device)  # (nk, 1, M)
    Vinv = np.linalg.inv(V)
    win = np.einsum("njm,nmi->nj", Vinv, E)              # (nk, M)  (V^-1 E)
    wout = np.einsum("nim,nmj->nj", D, V)                # (nk, M)  (D V)
    weights = np.abs(win) * np.abs(wout)
    lam = np.full(len(m_idx), np.nan, dtype=np.complex128)
    mu_sel = np.full(len(m_idx), np.nan, dtype=np.complex128)
    for i in range(len(m_idx)):
        ok = np.abs(mu[i]) <= 1.005
        if not ok.any():
            continue
        j = np.argmax(weights[i] * ok)
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
