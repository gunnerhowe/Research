"""Representation probes: Fourier concentration, class clustering, logit stats."""
import torch


@torch.no_grad()
def fourier_concentration(embed_weight: torch.Tensor, p: int):
    """Power spectrum of the number-token embedding over the token axis.
    Post-grok, power concentrates in a handful of key frequencies (Nanda 2023)."""
    W = embed_weight[:p].detach().float()
    spec = torch.fft.rfft(W, dim=0)
    power = spec.abs().pow(2).sum(1)[1:]  # drop DC
    power = power / power.sum().clamp_min(1e-12)
    top8 = torch.sort(power, descending=True).values[:8].sum().item()
    sp = torch.sort(power).values
    n = sp.numel()
    ranks = torch.arange(1, n + 1, device=sp.device, dtype=sp.dtype)
    gini = (((2 * ranks - n - 1) * sp).sum() / (n * sp.sum().clamp_min(1e-12))).item()
    return top8, gini


@torch.no_grad()
def class_cluster_metrics(h: torch.Tensor, y: torch.Tensor, p: int):
    """(1) Fisher ratio: between-class / within-class variance of reps.
    (2) Cosine gap: mean within-class pairwise cosine sim minus between-class."""
    h = h.detach().float()
    n, d = h.shape
    counts = torch.zeros(p, device=h.device).index_add_(0, y, torch.ones(n, device=h.device))
    sums = torch.zeros(p, d, device=h.device).index_add_(0, y, h)
    present = counts > 0
    mu_c = sums[present] / counts[present, None]
    cnt = counts[present]
    mu = h.mean(0)
    between = (cnt * (mu_c - mu).pow(2).sum(1)).sum() / n
    within = (h - (sums / counts.clamp_min(1)[:, None])[y]).pow(2).sum() / n
    fisher = (between / within.clamp_min(1e-12)).item()

    z = torch.nn.functional.normalize(h, dim=1)
    s_c = torch.zeros(p, d, device=h.device).index_add_(0, y, z)
    s_all = z.sum(0)
    per_class_sq = s_c.pow(2).sum(1)
    n_c = counts
    within_pairs = (n_c * (n_c - 1)).sum()
    within_sim = ((per_class_sq - n_c).sum() / within_pairs.clamp_min(1)).item()
    between_pairs = n * n - n_c.pow(2).sum()
    between_sim = ((s_all.pow(2).sum() - per_class_sq.sum()) / between_pairs.clamp_min(1)).item()
    return fisher, within_sim - between_sim


@torch.no_grad()
def logit_stats(logits: torch.Tensor):
    """Logit scale + softmax saturation, to address logit-scale mediation (2606.18465)."""
    scale = logits.norm(dim=1).mean().item()
    conf = torch.softmax(logits, dim=1).max(1).values.mean().item()
    return scale, conf


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two rep matrices (n, d1), (n, d2)."""
    X = X.float() - X.float().mean(0)
    Y = Y.float() - Y.float().mean(0)
    hsic = (X.T @ Y).pow(2).sum()
    nx = (X.T @ X).pow(2).sum().sqrt()
    ny = (Y.T @ Y).pow(2).sum().sqrt()
    return (hsic / (nx * ny).clamp_min(1e-12)).item()
