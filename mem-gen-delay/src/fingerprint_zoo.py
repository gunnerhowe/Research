"""P8 Mythos zoo: capture fingerprints from a FROZEN specimen, and read proximity inline.

All captures are single-forward-pass / one-shot-fit reads of frozen weights on a FIXED
probe bank P_k (NO training trajectory) — the requirement the Mythos idea needs and that
our induction precursor (watched forming) did NOT meet.

Two fingerprint kinds in the pilot:
  ATTENTION-ALIGNMENT (M0,M1,M2,M3): the run_probes gather idiom (train_lm.py) with the
    alignment index set to the episode's known target. Fingerprint = home (layer,head),
    its mean alignment mass m0, and the full per-head mass vector v0. Watch reads BOTH the
    frozen-home-head mass (relocation-FRAGILE) and the max-over-heads mass (relocation-
    ROBUST) against the ABSOLUTE m0 — never a running max (that firewall exposed the ESN
    data-mix reader).
  SUBSPACE (M6 Dyck): rank-r ridge subspace of the interior residual predicting running
    bracket depth; specimen-idiosyncratic. Watch reads energy in the FROZEN subspace plus
    a re-fit-R2 DIAGNOSTIC (never the alarm).

BEHAVIORAL metric (uniform): top-1 answer accuracy on P_k at the graded answer position.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def behavioral_metric(model, pb, device="cpu"):
    """Answer-span accuracy: fraction of graded positions whose argmax next-token equals the
    target, averaged over the span and the batch."""
    ids = pb["ids"].to(device)
    sp = pb["span_pos"].to(device)                    # (P,)
    logits, _ = model(ids)
    pred = logits[:, sp, :].argmax(-1)                # (N, P)
    return float((pred == pb["span_tok"].to(device)).float().mean())


@torch.no_grad()
def _alignment_mass(model, pb, device="cpu"):
    """Per-(layer,head) mean attention from each graded span position to its alignment
    target, averaged over the span. Returns (n_layers, n_heads)."""
    ids = pb["ids"].to(device)
    sp = pb["span_pos"].to(device)                    # (P,)
    tg = pb["span_tgt"].to(device)                    # (N, P)
    _, atts = model(ids, need_attn=True)
    N, P = tg.shape
    valid = (tg >= 0)
    tgc = tg.clamp(min=0)
    out = []
    for a in atts:                                    # a: (N, H, T, T)
        aq = a[:, :, sp, :]                           # (N, H, P, T)  from each span pos
        mass = aq.gather(3, tgc.view(N, 1, P, 1).expand(N, aq.shape[1], P, 1)).squeeze(3)
        mass = (mass * valid.view(N, 1, P)).sum((0, 2)) / valid.sum().clamp(min=1)  # (H,)
        out.append(mass)
    return torch.stack(out)                            # (L, H)


@torch.no_grad()
def capture_attention_fp(model, pb, device="cpu"):
    """Attention-alignment fingerprint from the frozen specimen: home (layer,head), m0, v0."""
    mass = _alignment_mass(model, pb, device)          # (L, H)
    flat = mass.flatten()
    idx = int(flat.argmax())
    L, H = mass.shape
    return dict(kind="attn", home_layer=idx // H, home_head=idx % H,
                m0=float(flat[idx]), v0=mass.flatten().tolist(), n_layers=L, n_heads=H)


@torch.no_grad()
def capture_subspace_fp(model, pb, r=4, ridge=1.0, layer=1, device="cpu"):
    """Dyck fingerprint: rank-r ridge subspace of the block[`layer`] interior residual
    predicting running bracket depth (labels in pb['depth'], -100 = ignore)."""
    ids = pb["ids"].to(device)
    _, _, hiddens = model(ids, need_attn=False, need_hidden=True)
    h = hiddens[layer]                                  # (N, T, D)
    d = pb["depth"].to(device)
    m = d != -100
    X = h[m].float()                                    # (M, D)
    y = d[m].float()                                    # (M,)
    Xc = X - X.mean(0, keepdim=True)
    yc = y - y.mean()
    A = Xc.T @ Xc + ridge * torch.eye(Xc.shape[1], device=device)
    w = torch.linalg.solve(A, Xc.T @ yc)                # ridge depth-decoder direction
    # rank-r subspace = top-r right singular vectors of the depth-correlated projection
    U, S, Vt = torch.linalg.svd(Xc * (Xc @ w).unsqueeze(1), full_matrices=False)
    basis = Vt[:r]                                      # (r, D) orthonormal-ish
    q, _ = torch.linalg.qr(basis.T)                     # (D, r) orthonormal
    pred = (Xc @ w)
    ss_res = float(((yc - pred) ** 2).sum())
    ss_tot = float((yc ** 2).sum() + 1e-9)
    return dict(kind="subspace", layer=layer, basis=q.T.tolist(), w=w.tolist(),
                r2=1 - ss_res / ss_tot, mu=X.mean(0).tolist())


@torch.no_grad()
def proximity(model, pb, fp, device="cpu"):
    """Inline watch read against the ABSOLUTE specimen fingerprint. Returns a dict of the
    structural scalars the composed anchor consumes, plus the behavioral accuracy."""
    acc = behavioral_metric(model, pb, device)
    if fp["kind"] == "attn":
        mass = _alignment_mass(model, pb, device).flatten()   # (L*H,)
        home = mass[fp["home_layer"] * fp["n_heads"] + fp["home_head"]]
        v0 = torch.tensor(fp["v0"], device=mass.device)
        cos = float(torch.dot(mass, v0) / (mass.norm() * v0.norm() + 1e-9))
        return dict(acc=acc, home_mass=float(home), max_mass=float(mass.max()), cos=cos,
                    home_frac=float(home / (fp["m0"] + 1e-9)),
                    max_frac=float(mass.max() / (fp["m0"] + 1e-9)))
    # subspace (Dyck): energy of the interior residual in the FROZEN specimen subspace
    ids = pb["ids"].to(device)
    _, _, hiddens = model(ids, need_attn=False, need_hidden=True)
    h = hiddens[fp["layer"]].float()
    d = pb["depth"].to(device)
    m = d != -100
    X = h[m] - torch.tensor(fp["mu"], device=h.device)
    B = torch.tensor(fp["basis"], device=h.device)             # (r, D)
    proj = X @ B.T                                              # (M, r)
    energy = float((proj.pow(2).sum(1) / (X.pow(2).sum(1) + 1e-9)).mean())
    return dict(acc=acc, subspace_energy=energy, home_mass=energy, max_mass=energy,
                home_frac=energy, max_frac=energy, cos=energy)
