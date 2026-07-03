"""Evaluation metrics: PSNR, high-frequency PSNR, radial per-band spectral error."""

import numpy as np
import torch


def psnr(pred, gt, data_range=1.0):
    mse = (pred - gt).pow(2).mean().item()
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range**2 / mse)


def highpass(img, sigma=4.0):
    """img minus Gaussian blur (scipy), i.e. the high-frequency residual."""
    from scipy.ndimage import gaussian_filter

    arr = img.detach().cpu().numpy()
    return img - torch.from_numpy(gaussian_filter(arr, sigma)).to(img)


def hf_psnr(pred, gt, sigma=4.0):
    """PSNR restricted to high-frequency residuals; range from GT residual."""
    hp_p, hp_g = highpass(pred, sigma), highpass(gt, sigma)
    rng = (hp_g.max() - hp_g.min()).item()
    return psnr(hp_p, hp_g, data_range=max(rng, 1e-8))


def ssim(pred, gt, data_range=1.0, window=11, sigma=1.5):
    """Standard single-scale SSIM (Wang et al. 2004) for (H, W) tensors."""
    import torch.nn.functional as F

    x = pred.float().unsqueeze(0).unsqueeze(0)
    y = gt.float().unsqueeze(0).unsqueeze(0)
    half = window // 2
    g = torch.exp(
        -((torch.arange(window, device=pred.device, dtype=torch.float32) - half) ** 2)
        / (2 * sigma**2)
    )
    g = (g / g.sum()).view(1, 1, -1)
    kern = (g.unsqueeze(-1) * g.unsqueeze(-2)).view(1, 1, window, window)

    def filt(t):
        return F.conv2d(t, kern, padding=half)

    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mx, my = filt(x), filt(y)
    vx = filt(x * x) - mx * mx
    vy = filt(y * y) - my * my
    cxy = filt(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / (
        (mx * mx + my * my + c1) * (vx + vy + c2)
    )
    return s.mean().item()


def radial_band_error(pred, gt, n_bands=8):
    """Per-band relative spectral error between two (H, W) images.

    Bins 2D FFT magnitudes into n_bands radial-frequency bands; returns an
    (n_bands,) array of  mean| |F_pred| - |F_gt| | / mean|F_gt|  per band.
    Band 0 is DC/low frequency; the last band is Nyquist-adjacent.
    """
    fp = torch.fft.fftshift(torch.fft.fft2(pred.float(), norm="ortho")).abs()
    fg = torch.fft.fftshift(torch.fft.fft2(gt.float(), norm="ortho")).abs()
    h, w = pred.shape
    yy = torch.arange(h, device=pred.device) - h // 2
    xx = torch.arange(w, device=pred.device) - w // 2
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    r = torch.sqrt((gy / (h / 2)) ** 2 + (gx / (w / 2)) ** 2)  # 0..~1.41
    r = (r / r.max()).clamp(max=1 - 1e-6)
    band = (r * n_bands).long()

    errs = []
    for b in range(n_bands):
        m = band == b
        denom = fg[m].mean().clamp_min(1e-12)
        errs.append(((fp[m] - fg[m]).abs().mean() / denom).item())
    return np.array(errs)


def spectrum_1d_error(pred, gt, n_bands=8):
    """Per-band relative spectral error for 1D signals (same idea as 2D)."""
    fp = torch.fft.rfft(pred.float(), norm="ortho").abs()
    fg = torch.fft.rfft(gt.float(), norm="ortho").abs()
    n = fp.shape[0]
    edges = np.linspace(0, n, n_bands + 1).astype(int)
    errs = []
    for b in range(n_bands):
        s = slice(edges[b], max(edges[b + 1], edges[b] + 1))
        denom = fg[s].mean().clamp_min(1e-12)
        errs.append(((fp[s] - fg[s]).abs().mean() / denom).item())
    return np.array(errs)
