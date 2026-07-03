"""Supervised contrastive loss (Khosla et al. 2020, L_out variant)."""
import torch
import torch.nn.functional as F


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    z = F.normalize(z, dim=1)
    sim = z @ z.T / tau
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos_mask = (labels[:, None] == labels[None, :]) & ~eye
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_count = pos_mask.sum(1)
    valid = pos_count > 0
    loss = -(log_prob.masked_fill(~pos_mask, 0.0)).sum(1)[valid] / pos_count[valid]
    return loss.mean()
