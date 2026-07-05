"""E0 — estimator validation against closed forms (GATE V).

(a) Random-phase-sinusoid Gaussian processes with KNOWN spectral moments:
    Rice prediction vs dense-grid hard counts at multiple levels, 3 seeds,
    target <= 3% relative error.
(b) OU process (AR(1) exact discretization): NOT mean-square differentiable,
    so continuous Rice diverges as dt -> 0; the sampled sequence instead
    validates the exact discrete-time Gaussian predictor (Owen's-T form) and
    quantifies the continuous-Rice-from-FD-moments bias vs lag-1 correlation
    (the regime real activation traces live in).
(c) Deterministic multisine: exact dense-grid counts vs segment estimator.
(d) eps sweep of the smoothed estimators (segment + midpoint) at PRACTICAL
    trace lengths (B=256, T=100): bias-variance, plus the split-half noise
    floor of hard counts at that length.
(e) Send-on-delta event rates vs the TV/theta ladder bound on several
    synthetic processes: the correction curve gamma(theta / sigma_delta),
    saved for reuse as the analytic event predictor in E1/E2.

Writes results/exp0_*.json.  GATE V check printed at the end.
"""

import math

import numpy as np
import torch

from common import DEVICE, RESULTS, log, save_json

from eventrice.delta import delta_encode_trace
from eventrice.estimator import (crossing_count_hard, crossing_rate_midpoint,
                                 crossing_rate_segment)
from eventrice.rice import (gaussian_discrete_rate, gaussian_tv_rate,
                            rice_rate, sod_rate_bound, spectral_moments)


def gp_sinusoids(seed, m=80, w_lo=4.0, w_hi=40.0, t_max=400.0, n=1_600_001):
    rng = np.random.default_rng(seed)
    w = rng.uniform(w_lo, w_hi, m)
    a = rng.normal(0, 1, m) * 0.2
    ph = rng.uniform(0, 2 * math.pi, m)
    t = np.linspace(0, t_max, n)
    tr = np.zeros_like(t)
    for ai, wi, pi in zip(a, w, ph):
        tr += ai * np.sin(wi * t + pi)
    lam0 = float(np.sum(a ** 2) / 2)
    lam2 = float(np.sum(a ** 2 * w ** 2) / 2)
    return t, tr, lam0, lam2


def part_a_rice_gp(seeds=(0, 1, 2)):
    out = []
    for seed in seeds:
        t, tr, lam0, lam2 = gp_sinusoids(seed)
        dt = t[1] - t[0]
        sig = math.sqrt(lam0)
        levels = np.array([0.0, 0.5 * sig, 1.0 * sig, 1.5 * sig])
        pred = rice_rate(levels, 0.0, lam0, lam2)
        meas = crossing_count_hard(torch.from_numpy(tr).unsqueeze(0),
                                   torch.from_numpy(levels), dt=dt).numpy()
        rel = np.abs(pred - meas) / meas
        out.append(dict(seed=seed, lam0=lam0, lam2=lam2,
                        levels_sigma=[0.0, 0.5, 1.0, 1.5],
                        rice=pred.tolist(), measured=meas.tolist(),
                        rel_err=rel.tolist()))
        log(f"[A] seed {seed}: max rel err {rel.max():.4f}")
    return out


def part_b_ou(seeds=(0, 1, 2), rhos=(0.98, 0.9, 0.7, 0.5, 0.3, 0.1),
              n=4_000_000):
    """AR(1) x_{t+1} = rho x_t + sqrt(1-rho^2) e_t (exact OU discretization,
    stationary variance 1)."""
    out = []
    for seed in seeds:
        rng = np.random.default_rng(100 + seed)
        for rho in rhos:
            e = rng.normal(0, math.sqrt(1 - rho ** 2), n).astype(np.float64)
            x = np.empty(n)
            x[0] = rng.normal()
            for i in range(1, n):
                x[i] = rho * x[i - 1] + e[i]
            levels = np.array([0.0, 0.5, 1.0, 1.5])
            meas = crossing_count_hard(torch.from_numpy(x).unsqueeze(0),
                                       torch.from_numpy(levels)).numpy()
            exact = gaussian_discrete_rate(levels, 0.0, 1.0, rho)
            mom = spectral_moments(x[None, :], dt=1.0)
            rice_fd = rice_rate(levels, float(mom["mean"][0]),
                                float(mom["lam0"][0]), float(mom["lam2"][0]))
            out.append(dict(seed=seed, rho=rho, levels=levels.tolist(),
                            measured=meas.tolist(), exact_discrete=exact.tolist(),
                            rice_fd=rice_fd.tolist(),
                            rel_err_discrete=(np.abs(exact - meas) / meas).tolist(),
                            rel_err_rice_fd=(np.abs(rice_fd - meas) / meas).tolist()))
        log(f"[B] seed {seed} done")
    return out


def part_c_multisine():
    t = np.linspace(0, 20, 20001)
    dt = t[1] - t[0]
    tr = (np.sin(2 * math.pi * 1.3 * t) + 0.6 * np.sin(2 * math.pi * 3.7 * t + 1.0)
          + 0.3 * np.sin(2 * math.pi * 0.4 * t + 2.0))
    dense_t = np.linspace(0, 20, 2_000_001)
    dense = (np.sin(2 * math.pi * 1.3 * dense_t)
             + 0.6 * np.sin(2 * math.pi * 3.7 * dense_t + 1.0)
             + 0.3 * np.sin(2 * math.pi * 0.4 * dense_t + 2.0))
    levels = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
    exact = crossing_count_hard(torch.from_numpy(dense).unsqueeze(0),
                                torch.from_numpy(levels),
                                dt=dense_t[1] - dense_t[0]).numpy()
    seg = crossing_rate_segment(torch.from_numpy(tr).unsqueeze(0),
                                torch.from_numpy(levels), eps=0.005, dt=dt).numpy()
    rel = np.abs(seg - exact) / exact
    log(f"[C] multisine max rel err {rel.max():.4f}")
    return dict(levels=levels.tolist(), exact=exact.tolist(),
                segment=seg.tolist(), rel_err=rel.tolist())


def part_d_eps_sweep(seed=0, B=256, T=100):
    """Practical-length traces: GP sinusoids sampled at working resolution.
    Reference = hard counts on the same traces; bias/std over 32 batch splits.
    Split-half floor: |c(half1) - c(half2)| / c(all) on the hard counts."""
    rng = np.random.default_rng(seed)
    m, w_lo, w_hi = 40, 0.05, 0.6  # freqs vs dt=1 sampling
    tgrid = np.arange(T)
    trs = np.zeros((B, T))
    for b in range(B):
        w = rng.uniform(w_lo, w_hi, m)
        a = np.abs(rng.normal(0, 0.3, m))
        ph = rng.uniform(0, 2 * math.pi, m)
        trs[b] = sum(ai * np.sin(wi * tgrid + pi) for ai, wi, pi in zip(a, w, ph))
    trs_t = torch.from_numpy(trs)
    sig = trs.std()
    levels = torch.tensor([0.0, 0.5 * sig, 1.0 * sig])
    hard = crossing_count_hard(trs_t, levels).numpy()
    eps_grid = [0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0]
    rows = []
    for eps_rel in eps_grid:
        eps = eps_rel * sig
        seg = crossing_rate_segment(trs_t, levels, eps=eps).numpy()
        mid = crossing_rate_midpoint(trs_t, levels, eps=eps).numpy()
        # std over 32 disjoint batches of 8 traces
        seg_b, mid_b = [], []
        for i in range(0, B, 8):
            seg_b.append(crossing_rate_segment(trs_t[i:i + 8], levels, eps=eps).numpy())
            mid_b.append(crossing_rate_midpoint(trs_t[i:i + 8], levels, eps=eps).numpy())
        rows.append(dict(eps_rel=eps_rel,
                         seg_bias=((seg - hard) / hard).tolist(),
                         mid_bias=((mid - hard) / hard).tolist(),
                         seg_std=(np.std(seg_b, axis=0) / hard).tolist(),
                         mid_std=(np.std(mid_b, axis=0) / hard).tolist()))
    h1 = crossing_count_hard(trs_t[: B // 2], levels).numpy()
    h2 = crossing_count_hard(trs_t[B // 2:], levels).numpy()
    floor = (np.abs(h1 - h2) / hard).tolist()
    log(f"[D] split-half floor {floor}")
    return dict(levels_sigma=[0.0, 0.5, 1.0], hard=hard.tolist(),
                eps_sweep=rows, split_half_floor=floor)


def part_e_sod_gamma(seeds=(0, 1, 2)):
    """gamma(x) = measured SOD event rate / (TV-rate / theta) bound, with
    x = theta / sigma_delta. Several processes to test universality."""
    thetas_rel = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0,
                           3.0, 5.0, 8.0])
    processes = {}
    T, B = 2000, 64
    for seed in seeds:
        rng = np.random.default_rng(200 + seed)
        # smooth GP (well-resolved), rough GP, and OU at two correlations
        for name, gen in {
            "gp_smooth": lambda: _gp_batch(rng, B, T, 0.02, 0.15),
            "gp_rough": lambda: _gp_batch(rng, B, T, 0.1, 0.9),
            "ou_rho0.95": lambda: _ar1_batch(rng, B, T, 0.95),
            "ou_rho0.7": lambda: _ar1_batch(rng, B, T, 0.7),
        }.items():
            tr = gen()
            mom = spectral_moments(tr.numpy(), dt=1.0)
            sd = float(np.mean(mom["sigma_delta"]))
            tv = float(np.mean(mom["tv_rate"]))
            tr_dev = tr.unsqueeze(-1).to(DEVICE).float()
            rows = []
            for x in thetas_rel:
                theta = x * sd
                _, ev = delta_encode_trace(tr_dev, theta)
                meas = ev.float().mean().item()
                bound = float(sod_rate_bound(theta, tv))
                gauss_bound = float(sod_rate_bound(
                    theta, gaussian_tv_rate(np.mean(mom["lam2"]))))
                rows.append(dict(x=float(x), theta=float(theta), measured=meas,
                                 bound=bound, gamma=meas / bound,
                                 bound_gauss_tv=gauss_bound))
            processes.setdefault(name, []).append(dict(seed=seed, rows=rows))
        log(f"[E] seed {seed} done")
    # pooled correction curve: mean gamma per x over processes and seeds
    xs = thetas_rel.tolist()
    gam = {name: np.mean([[r["gamma"] for r in run["rows"]] for run in runs],
                         axis=0).tolist()
           for name, runs in processes.items()}
    pooled = np.mean(list(gam.values()), axis=0).tolist()
    spread = np.std(list(gam.values()), axis=0).tolist()
    return dict(x_grid=xs, per_process_gamma=gam, pooled_gamma=pooled,
                process_spread=spread, runs=processes)


def _gp_batch(rng, B, T, w_lo, w_hi, m=40):
    tgrid = np.arange(T)
    trs = np.zeros((B, T))
    for b in range(B):
        w = rng.uniform(w_lo * 2 * math.pi, w_hi * 2 * math.pi, m)
        a = np.abs(rng.normal(0, 1.0 / math.sqrt(m), m))
        ph = rng.uniform(0, 2 * math.pi, m)
        trs[b] = sum(ai * np.sin(wi * tgrid + pi) for ai, wi, pi in zip(a, w, ph))
    return torch.from_numpy(trs)


def _ar1_batch(rng, B, T, rho):
    e = rng.normal(0, math.sqrt(1 - rho ** 2), (B, T))
    x = np.empty((B, T))
    x[:, 0] = rng.normal(size=B)
    for i in range(1, T):
        x[:, i] = rho * x[:, i - 1] + e[:, i]
    return torch.from_numpy(x)


def main():
    a = part_a_rice_gp()
    save_json(a, RESULTS / "exp0_rice_gp.json")
    b = part_b_ou()
    save_json(b, RESULTS / "exp0_ou.json")
    c = part_c_multisine()
    save_json(c, RESULTS / "exp0_multisine.json")
    d = part_d_eps_sweep()
    save_json(d, RESULTS / "exp0_eps_sweep.json")
    e = part_e_sod_gamma()
    save_json(e, RESULTS / "exp0_sod_gamma.json")

    # ---- GATE V ----
    max_a = max(max(r["rel_err"]) for r in a)
    max_b = max(max(r["rel_err_discrete"]) for r in b)
    max_c = max(c["rel_err"])
    gate = dict(
        rice_gp_max_rel_err=max_a,
        ou_discrete_max_rel_err=max_b,
        multisine_max_rel_err=max_c,
        passed=bool(max_a <= 0.03 and max_b <= 0.03 and max_c <= 0.03),
    )
    save_json(gate, RESULTS / "exp0_gate_v.json")
    log(f"GATE V: {'PASS' if gate['passed'] else 'FAIL'}  "
        f"(A {max_a:.4f}, B {max_b:.4f}, C {max_c:.4f}; threshold 0.03)")


if __name__ == "__main__":
    main()
