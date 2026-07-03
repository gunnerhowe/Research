"""Evaluation metrics for long-horizon rollouts vs ground truth.

All event metrics use the shared event definition from data/events.json.
Diverged rollouts are censored at first divergence and reported explicitly.
"""

import numpy as np
import torch
from scipy import stats as sps

from events import hard_events, streams

FANO_WINDOWS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
FANO_WINDOWS_COARSE = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]  # dt >= 1
RL_QUANTILES = [0.90, 0.95, 0.98, 0.99, 0.995]


def default_fano_windows(dt: float):
    return FANO_WINDOWS_COARSE if dt >= 1.0 else FANO_WINDOWS


def censor(xn: torch.Tensor, thresh: float = 8.0):
    """xn: (B, T, K) normalized states. Returns (xn, surv) where surv[b] is the
    first index where |xn| exceeded thresh (or T if never); states after
    divergence are frozen at the last sane value to keep shapes."""
    bad = (xn.abs().max(dim=2).values > thresh) | ~torch.isfinite(xn).all(dim=2)
    B, T, K = xn.shape
    surv = torch.full((B,), T, dtype=torch.long)
    for b in range(B):
        nz = torch.nonzero(bad[b])
        if nz.numel():
            surv[b] = nz[0, 0].item()
    return surv


def pooled_iets(ev: torch.Tensor, dt: float, surv=None) -> np.ndarray:
    st = streams(ev)  # (B*K, T)
    B, T, K = ev.shape
    iets = []
    for i, row in enumerate(st):
        b = i // K
        end = int(surv[b]) if surv is not None else T
        idx = torch.nonzero(row[:end]).squeeze(-1)
        if idx.numel() >= 2:
            iets.append(np.diff(idx.numpy()) * dt)
    return np.concatenate(iets) if iets else np.array([])


def fano_curve(ev: torch.Tensor, dt: float, surv=None) -> dict:
    """Fano factor per window size, using only pre-divergence spans."""
    B, T, K = ev.shape
    out = {}
    evf = ev.float()
    for w_mtu in default_fano_windows(dt):
        w = int(round(w_mtu / dt))
        if w < 1:
            out[w_mtu] = np.nan
            continue
        counts = []
        for b in range(B):
            end = min(int(surv[b]) if surv is not None else T, T)
            nwin = end // w
            if nwin >= 1:
                c = evf[b, :nwin * w].reshape(nwin, w, K).sum(dim=1)
                counts.append(c.reshape(-1))
        if counts:
            c = torch.cat(counts)
            m = c.mean().item()
            out[w_mtu] = float(c.var(unbiased=True).item() / m) if m > 0 else np.nan
        else:
            out[w_mtu] = np.nan
    return out


def return_periods(x: torch.Tensor, thresholds: list, kind: str, dt: float,
                   surv=None) -> list:
    """Pooled return period (MTU per upcrossing per stream) for each threshold."""
    B, T, K = x.shape
    rps = []
    for u in thresholds:
        ev = hard_events(x, u, kind)
        n, time_tot = 0, 0.0
        for b in range(B):
            end = int(surv[b]) if surv is not None else T
            n += int(ev[b, : max(end - 1, 0)].sum())
            time_tot += max(end - 1, 0) * dt * K
        rps.append(time_tot / n if n > 0 else np.inf)
    return rps


def psd_mean(x: torch.Tensor, nseg: int = 1024) -> np.ndarray:
    """Mean per-site power spectrum over non-overlapping segments."""
    B, T, K = x.shape
    nwin = T // nseg
    if nwin == 0:
        return np.array([])
    seg = x[:, :nwin * nseg].reshape(B, nwin, nseg, K)
    seg = seg - seg.mean(dim=2, keepdim=True)
    p = torch.fft.rfft(seg, dim=2).abs().pow(2).mean(dim=(0, 1, 3))
    return p.numpy()


def acf_mean(x: torch.Tensor, nlags: int = 100) -> np.ndarray:
    """Mean per-site autocorrelation up to nlags."""
    B, T, K = x.shape
    xc = x - x.mean(dim=1, keepdim=True)
    denom = (xc * xc).sum(dim=1)
    out = np.zeros(nlags + 1)
    for lag in range(nlags + 1):
        num = (xc[:, : T - lag] * xc[:, lag:]).sum(dim=1)
        out[lag] = (num / denom).mean().item()
    return out


def hazard_curve(iets: np.ndarray, dt: float, max_lag_steps: int = 60) -> np.ndarray:
    """Empirical discrete hazard h[k] = P(event at lag k | survived to k),
    lag in steps since the previous event. iets in MTU."""
    lags = np.round(iets / dt).astype(int)
    out = np.full(max_lag_steps, np.nan)
    for k in range(1, max_lag_steps + 1):
        at_risk = (lags >= k).sum()
        if at_risk > 50:
            out[k - 1] = (lags == k).sum() / at_risk
    return out


def crps_ensemble(ens: torch.Tensor, obs: torch.Tensor) -> float:
    """ens: (m, B, K), obs: (B, K). Standard fair-CRPS estimator."""
    m = ens.shape[0]
    t1 = (ens - obs[None]).abs().mean()
    t2 = (ens[:, None] - ens[None, :]).abs().mean() * (m / (2 * (m - 1))) \
        if m > 1 else torch.tensor(0.0)
    return float(t1 - t2)


def event_ll_under(tpp, ev: torch.Tensor, dt: float, warmup: int,
                   device: str = "cuda", chunk: int = 256) -> float:
    """LL/step of hard events under a fitted TPP (rollout self-history warmup).
    Chunked over streams to bound GRU memory."""
    st = streams(ev).float()
    vals, ns = [], []
    with torch.no_grad():
        for i in range(0, st.shape[0], chunk):
            w = st[i:i + chunk].to(device)
            vals.append(tpp.log_likelihood(w, dt, warmup=warmup).item())
            ns.append(w.shape[0])
    return float(np.average(vals, weights=ns))


def summarize_events(x_phys: torch.Tensor, u: float, kind: str, dt: float,
                     surv=None) -> dict:
    ev = hard_events(x_phys, u, kind)
    iets = pooled_iets(ev, dt, surv)
    n = len(iets)
    res = dict(n_iets=n)
    if n > 10:
        res.update(rate=float(ev.float().mean().item() / dt),
                   iet_mean=float(iets.mean()), iet_cv=float(iets.std() / iets.mean()))
    res["fano"] = fano_curve(ev, dt, surv)
    return res, iets, ev
