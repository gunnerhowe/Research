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
    """Top-1 answer accuracy on a probe bank (fraction of episodes whose argmax next-token
    at the graded answer position equals the correct answer)."""
    ids = pb["ids"].to(device)
    logits, _ = model(ids)
    pred = logits[torch.arange(len(ids)), pb["ans_pos"].to(device)].argmax(-1)
    return float((pred == pb["ans_tok"].to(device)).float().mean())


@torch.no_grad()
def _alignment_mass(model, pb, device="cpu"):
    """Per-(layer,head) mean attention from each episode's answer position to its alignment
    target. Returns a (n_layers, n_heads) tensor — the run_probes gather with a per-episode
    target index."""
    ids = pb["ids"].to(device)
    ap = pb["ans_pos"].to(device)
    tg = pb["align_tgt"].to(device)
    _, atts = model(ids, need_attn=True)
    N = len(ids)
    rows = torch.arange(N, device=device)
    out = []
    for a in atts:                                   # a: (N, H, T, T)
        aq = a[rows, :, ap, :]                        # (N, H, T) attention FROM answer pos
        mass = aq.gather(2, tg.view(N, 1, 1).expand(N, aq.shape[1], 1)).squeeze(2)  # (N,H)
        out.append(mass.mean(0))                       # (H,)
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
