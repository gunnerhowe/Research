"""GENERATE: a Doob-h-transform bridge that steers the base diffusion toward the
recoverable collar of the censored complement, stated honestly as a BIASED
GUIDED APPROXIMATION (not exact conditioning).

Reverse VP dynamics get an added drift g^2 grad_x log h(x,t) with
h(x,t)=E_base[r(x_0)|x_t] approximated by reconstruction/Tweedie guidance
h ~ r(x_hat_0(x_t,t)). The reward is the ESTIMATED SELECTION PROPENSITY on the
manifold collar:

    r(x_0) = (1 - s_hat(x_0)) * 1[p_hat>=tau] * 1[u(s_hat)<=u_max] * 1[prox ok].

This "guidance signal = estimated selection function" is the unoccupied move.
Off-manifold guards (all mandatory): (i) density gate inside the reward;
(ii) LOW-t-ONLY guidance; (iii) HARD density veto; (iv) guidance-to-score norm
clip <=1; (v) independent post-hoc validation (validate.py).

The sharp control is a MISDIRECTED selector at MATCHED guidance strength: the
identical bridge with s_hat precomposed with a fixed rotation (decoy="rotate")
or negated (decoy="negate"). It holds guidance machinery and manifold gates
fixed and varies ONLY whether guidance points at the real complement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RewardConfig:
    gamma: float = 4.0             # guidance scale
    tau_log: float = -6.0          # soft density-gate level (log p)
    k_p: float = 4.0               # density-gate sharpness
    veto_log: float = -8.0         # HARD density veto level (log p)
    u_max: float = 0.5             # epistemic-uncertainty gate (log-odds std)
    d_max: float = 0.6             # proximity gate (max dist to observed support)
    sigma_guide_max: float = 0.6   # LOW-t-only: guide when sqrt(1-abar)<=this
    cap_ratio: float = 1.0         # clip ||guidance eps|| / ||base eps|| <= this
    rotate_deg: float = 90.0       # decoy rotation angle


@dataclass
class BridgeDiag:
    guided_frac: float = 0.0
    mean_guide_norm: float = 0.0
    mean_base_norm: float = 0.0
    frac_veto: float = 0.0
    frac_unc_gate: float = 0.0
    frac_prox_gate: float = 0.0
    frac_capped: float = 0.0
    steps_guided: int = 0
    extra: dict = field(default_factory=dict)


def _rot(deg, device):
    a = np.deg2rad(deg)
    return torch.tensor([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]],
                        dtype=torch.float32, device=device)


def guided_sample(diffusion, selector, gate_kde, y_class, n, cfg: RewardConfig,
                  decoy=None, seed=0):
    """Run the guided bridge for one class. Returns (X_data, BridgeDiag).

    decoy: None (real selection guidance), "rotate", or "negate".
    """
    dev = diffusion.device
    g = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(n, 2, device=dev, generator=g)
    y = torch.full((n,), y_class, device=dev, dtype=torch.long)
    mu = torch.tensor(selector._mu, dtype=torch.float32, device=dev)
    R = _rot(cfg.rotate_deg, dev)

    diag = BridgeDiag()
    gn_acc, bn_acc, cap_acc = [], [], 0.0

    for i in reversed(range(diffusion.T)):
        with torch.no_grad():
            eps = diffusion.eps(z, i, y)
        abar = diffusion.abar[i]
        sigma_t = float(torch.sqrt(1 - abar).item())
        base_norm = eps.norm(dim=1)

        do_guide = (cfg.gamma > 0) and (sigma_t <= cfg.sigma_guide_max)
        if do_guide:
            delta, gate = _guidance_eps(diffusion, selector, gate_kde, z, eps,
                                        i, mu, R, cfg, decoy)
            # guidance-to-score norm clip (iv)
            gnorm = delta.norm(dim=1) + 1e-12
            scale = torch.clamp(cfg.cap_ratio * base_norm / gnorm, max=1.0)
            capped = (scale < 1.0).float().mean().item()
            delta = delta * scale[:, None]
            eps_g = eps - delta
            gn_acc.append(delta.norm(dim=1).mean().item())
            bn_acc.append(base_norm.mean().item())
            cap_acc += capped
            diag.steps_guided += 1
            diag.frac_veto += gate["veto"]
            diag.frac_unc_gate += gate["unc"]
            diag.frac_prox_gate += gate["prox"]
        else:
            eps_g = eps

        z = diffusion._ddpm_step(z, i, eps_g, g)

    X = diffusion.denorm(z).detach().cpu().numpy()
    ng = max(diag.steps_guided, 1)
    diag.guided_frac = diag.steps_guided / diffusion.T
    diag.mean_guide_norm = float(np.mean(gn_acc)) if gn_acc else 0.0
    diag.mean_base_norm = float(np.mean(bn_acc)) if bn_acc else 0.0
    diag.frac_veto /= ng
    diag.frac_unc_gate /= ng
    diag.frac_prox_gate /= ng
    diag.frac_capped = cap_acc / ng
    return X, diag


def _guidance_eps(diffusion, selector, gate_kde, z, eps, i, mu, R, cfg, decoy):
    """grad_z log r(x_hat_0) mapped to an eps-shift, with the hard collar gates
    applied as a per-sample mask. Tweedie reconstruction with eps detached
    (the '~free' reconstruction-guidance approximation)."""
    dev = z.device
    abar = diffusion.abar[i]
    z_g = z.detach().requires_grad_(True)
    x0n = (z_g - torch.sqrt(1 - abar) * eps.detach()) / torch.sqrt(abar)
    x0 = diffusion.denorm(x0n)                      # data space, diff'able in z_g

    # --- selection term (the decoyable direction) ---
    x_sel = x0
    if decoy == "rotate":
        x_sel = (x0 - mu) @ R.T + mu
    s_hat = selector.s_hat_torch(x_sel).clamp(1e-4, 1 - 1e-4)
    sel_term = s_hat if decoy == "negate" else (1 - s_hat)
    log_sel = torch.log(sel_term + 1e-6)

    # --- soft density gate on the TRUE manifold ---
    logp = gate_kde.log_p(x0)
    log_dens_gate = torch.nn.functional.logsigmoid(cfg.k_p * (logp - cfg.tau_log))
    log_r = log_sel + log_dens_gate

    grad = torch.autograd.grad(log_r.sum(), z_g)[0]     # grad_z log r
    delta = cfg.gamma * torch.sqrt(1 - abar) * grad     # eps-shift

    # --- hard collar gates (mask guidance; never steer by untrustworthy s_hat) ---
    with torch.no_grad():
        x0_np = x0.detach().cpu().numpy()
        x_sel_np = x_sel.detach().cpu().numpy()
        u = torch.tensor(selector.uncertainty(x_sel_np), device=dev)
        prox = torch.tensor(selector.proximity(x0_np), device=dev)
        veto = (logp < cfg.veto_log)
        unc_gate = (u > cfg.u_max)
        prox_gate = (prox > cfg.d_max)
        mask = (veto | unc_gate | prox_gate)
        delta = delta.clone()
        delta[mask] = 0.0
        gate = {"veto": float(veto.float().mean()),
                "unc": float(unc_gate.float().mean()),
                "prox": float(prox_gate.float().mean())}
    return delta, gate


def generate_labeled(diffusion, selector, gate_kde, n_per_class, cfg,
                     decoy=None, seed=0, n_classes=2):
    """Class-conditional generation of whole labeled units in the collar."""
    Xs, ys, diags = [], [], {}
    for c in range(n_classes):
        Xc, dg = guided_sample(diffusion, selector, gate_kde, c, n_per_class,
                               cfg, decoy=decoy, seed=seed + c)
        Xs.append(Xc)
        ys.append(np.full(len(Xc), c))
        diags[c] = dg
    return np.concatenate(Xs), np.concatenate(ys).astype(int), diags
