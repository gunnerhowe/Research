"""Signals, images, coordinate sampling, and ground-truth gradients.

Conventions: coordinates live in [-1, 1]^d. Images are (H, W) float tensors in [0, 1]
(grayscale); image gradients are with respect to the *normalized* coordinates, i.e.
central differences scaled by (n-1)/2, so they are consistent with autograd gradients
of an INR taking [-1, 1]^2 inputs. Bilinear interpolation uses align_corners=True,
matching the same coordinate convention.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- 1D signals

def multisine(freqs=(2, 5, 11, 23, 47), amps=None, phases=None, seed=0):
    """Sum of sinusoids on [-1, 1] with analytic derivative.

    Returns (f, df) callables mapping (N,) or (N,1) coords -> (N,) values.
    """
    rng = np.random.default_rng(seed)
    freqs = np.asarray(freqs, dtype=np.float64)
    if amps is None:
        amps = 1.0 / np.sqrt(np.arange(1, len(freqs) + 1))
    amps = np.asarray(amps, dtype=np.float64)
    if phases is None:
        phases = rng.uniform(0, 2 * np.pi, len(freqs))
    phases = np.asarray(phases, dtype=np.float64)

    def to_flat(x):
        return x.reshape(-1) if x.dim() > 1 else x

    def f(x):
        x = to_flat(x)
        out = torch.zeros_like(x)
        for a, k, p in zip(amps, freqs, phases):
            out = out + a * torch.sin(math.pi * k * x + p)
        return out

    def df(x):
        x = to_flat(x)
        out = torch.zeros_like(x)
        for a, k, p in zip(amps, freqs, phases):
            out = out + a * math.pi * k * torch.cos(math.pi * k * x + p)
        return out

    return f, df


# ---------------------------------------------------------------- 2D images

def load_image(name="camera", size=256):
    """Load a standard test image as (H, W) float tensor in [0, 1].

    Tries imageio's bundled standard images; falls back to a synthetic
    multi-scale texture so experiments never block on a download.
    """
    img = None
    try:
        import imageio.v3 as iio

        arr = iio.imread(f"imageio:{name}.png")
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        img = torch.from_numpy(arr.astype(np.float32) / 255.0)
    except Exception:
        img = synthetic_texture(size)
    if img.shape != (size, size):
        img = F.interpolate(
            img[None, None], size=(size, size), mode="bicubic", align_corners=True
        )[0, 0].clamp(0, 1)
    return img


def synthetic_texture(size=256, seed=0):
    """Multi-scale synthetic image: smooth blobs + oriented high-frequency texture."""
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(
        np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing="ij"
    )
    img = np.zeros((size, size))
    for _ in range(6):  # low-frequency blobs
        cx, cy = rng.uniform(-0.8, 0.8, 2)
        s = rng.uniform(0.15, 0.4)
        img += rng.uniform(0.3, 1.0) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / s**2)
    for k in (17, 31, 53):  # oriented high-frequency gratings, spatially masked
        th = rng.uniform(0, np.pi)
        cx, cy = rng.uniform(-0.5, 0.5, 2)
        mask = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 0.35**2)
        img += 0.25 * mask * np.sin(np.pi * k * (x * np.cos(th) + y * np.sin(th)))
    img = (img - img.min()) / (img.max() - img.min())
    return torch.from_numpy(img.astype(np.float32))


def grid_coords(h, w=None, device="cpu"):
    """(H*W, 2) coordinates in [-1, 1]^2, xy order matching grid_sample."""
    w = w or h
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(-1, 2)


def image_gradients(img):
    """Central-difference gradient of (H, W) image w.r.t. normalized coords.

    Returns (H, W, 2) with [df/dx, df/dy]; pixel spacing is 2/(n-1) per axis.
    """
    h, w = img.shape
    gy, gx = torch.gradient(img)  # per-pixel differences
    return torch.stack([gx * (w - 1) / 2.0, gy * (h - 1) / 2.0], dim=-1)


def bilinear_sample(field, coords):
    """Sample an (H, W) or (H, W, C) field at (N, 2) coords in [-1, 1]^2 (xy)."""
    if field.dim() == 2:
        field = field.unsqueeze(-1)
    inp = field.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    grid = coords.view(1, 1, -1, 2)
    out = F.grid_sample(inp, grid, mode="bilinear", align_corners=True)
    out = out[0, :, 0, :].permute(1, 0)  # (N, C)
    return out.squeeze(-1) if out.shape[-1] == 1 else out


# ----------------------------------------------------- non-uniform sampling

def sample_coords(n, mode="uniform", seed=0, device="cpu"):
    """Draw (n, 2) sample coordinates in [-1, 1]^2.

    'uniform'  - homogeneous random (baseline off-grid case)
    'blobs'    - strongly non-uniform: 75% from Gaussian clusters, 25% uniform
                 floor so no region is entirely empty
    'ramp'     - density increases linearly left to right (smooth non-uniformity)
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "uniform":
        pts = torch.rand(n, 2, generator=g) * 2 - 1
    elif mode == "blobs":
        n_cl = int(0.75 * n)
        k = 5
        centers = torch.rand(k, 2, generator=g) * 1.6 - 0.8
        idx = torch.randint(k, (n_cl,), generator=g)
        pts_cl = centers[idx] + 0.12 * torch.randn(n_cl, 2, generator=g)
        pts_un = torch.rand(n - n_cl, 2, generator=g) * 2 - 1
        pts = torch.cat([pts_cl, pts_un]).clamp(-1, 1)
    elif mode == "ramp":
        # p(x) ~ 0.15 + 0.85*(x+1)/2 via rejection sampling on the x axis
        pts = []
        need = n
        while need > 0:
            cand = torch.rand(2 * need, 2, generator=g) * 2 - 1
            accept = torch.rand(2 * need, generator=g) < (
                0.15 + 0.85 * (cand[:, 0] + 1) / 2
            )
            pts.append(cand[accept][:need])
            need = n - sum(p.shape[0] for p in pts)
        pts = torch.cat(pts)
    else:
        raise ValueError(f"unknown sampling mode {mode!r}")
    return pts.to(device)


def scattered_to_grid(points, values, size, method="linear"):
    """Interpolate scattered samples onto a regular grid (what FFT losses require).

    Uses scipy griddata; linear inside the convex hull, nearest fill outside.
    Returns (size, size) tensor. This is the resampling step that pointwise losses
    (Kac-Rice, Sobolev) do not need.
    """
    from scipy.interpolate import griddata

    pts = points.detach().cpu().numpy()
    val = values.detach().cpu().numpy()
    ys = np.linspace(-1, 1, size)
    gy, gx = np.meshgrid(ys, ys, indexing="ij")
    grid = griddata(pts, val, (gx, gy), method=method)
    if np.isnan(grid).any():
        near = griddata(pts, val, (gx, gy), method="nearest")
        grid = np.where(np.isnan(grid), near, grid)
    return torch.from_numpy(grid.astype(np.float32))
