"""Baseline losses: Sobolev/H1 gradient matching and Focal Frequency Loss.

FocalFrequencyLoss follows Jiang et al., ICCV 2021 (arXiv:2012.12821): L2 distance in
the DFT domain, weighted per frequency by w = |F_pred - F_gt|^alpha normalized to
max 1 (weights detached). This local implementation supports 1D and 2D signals; it is
numerically checked against the official `focal-frequency-loss` package in the tests.
"""

import torch
import torch.nn as nn


class SobolevLoss(nn.Module):
    """H1 seminorm matching: mean |grad f_pred - grad f_gt|^2 at sample points.

    The closest spatial-domain competitor: also pointwise and mesh-free, but it
    matches gradients *pointwise* rather than matching a distributional statistic.

    normalize=True divides by the GT gradient power, making the loss scale-free
    (O(1) at init regardless of the signal's frequency content) so a single
    weight transfers across tasks.
    """

    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, pred_grad, gt_grad):
        err = (pred_grad - gt_grad).pow(2).sum(dim=-1).mean()
        if self.normalize:
            err = err / gt_grad.detach().pow(2).sum(dim=-1).mean().clamp_min(1e-12)
        return err


class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss for 1D (N, C, L) or 2D (N, C, H, W) signals."""

    def __init__(self, loss_weight=1.0, alpha=1.0, log_matrix=False, ave_spectrum=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.log_matrix = log_matrix
        self.ave_spectrum = ave_spectrum

    def _spectrum(self, x):
        if x.dim() == 3:  # (N, C, L)
            f = torch.fft.fft(x, dim=-1, norm="ortho")
        elif x.dim() == 4:  # (N, C, H, W)
            f = torch.fft.fft2(x, norm="ortho")
        else:
            raise ValueError(f"expected 3D or 4D tensor, got {x.dim()}D")
        return torch.stack([f.real, f.imag], dim=-1)

    def forward(self, pred, target):
        pf, tf = self._spectrum(pred), self._spectrum(target)
        if self.ave_spectrum:
            pf, tf = pf.mean(0, keepdim=True), tf.mean(0, keepdim=True)

        dist2 = (pf - tf).pow(2).sum(dim=-1)  # squared spectrum distance per bin

        with torch.no_grad():
            w = dist2.sqrt().pow(self.alpha)
            if self.log_matrix:
                w = torch.log(w + 1.0)
            # official impl normalizes per (N, C) slice: max over spatial dims only
            spatial = tuple(range(2, w.dim()))
            w = w / w.amax(dim=spatial, keepdim=True).clamp_min(1e-12)
            w = torch.nan_to_num(w, nan=0.0)
            w = w.clamp(0.0, 1.0)

        return (w * dist2).mean() * self.loss_weight
