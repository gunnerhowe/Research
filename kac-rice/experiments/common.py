"""Shared experiment utilities: config table, plotting, result saving."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def save_history(histories, path):
    """histories: {config_name: [record dicts]} -> JSON."""
    ser = {
        k: [{kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
             for kk, vv in r.items()} for r in v]
        for k, v in histories.items()
    }
    Path(path).write_text(json.dumps(ser, indent=1))


def plot_psnr_curves(histories, path, key="psnr", title="PSNR vs iteration"):
    plt.figure(figsize=(7, 4.5))
    for name, hist in histories.items():
        it = [r["iter"] for r in hist if key in r]
        v = [r[key] for r in hist if key in r]
        plt.plot(it, v, label=name, lw=1.8)
    plt.xlabel("iteration")
    plt.ylabel(key)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_band_errors(band_errors, path, title="Per-band relative spectral error"):
    """band_errors: {config_name: (n_bands,) array} — lower is better; log scale."""
    plt.figure(figsize=(7, 4.5))
    n = len(next(iter(band_errors.values())))
    x = np.arange(n)
    width = 0.8 / len(band_errors)
    for i, (name, err) in enumerate(band_errors.items()):
        plt.bar(x + i * width, err, width, label=name)
    plt.xticks(x + 0.4, [f"band {b}" for b in x], fontsize=8)
    plt.ylabel("relative |spectrum| error")
    plt.yscale("log")
    plt.title(title + "  (band 0 = low freq)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_recons(images, gt, path, title=""):
    """images: {name: (H,W) tensor}. Renders GT + reconstructions + HF residuals."""
    import torch
    from kacrice.metrics import highpass

    n = len(images) + 1
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.4))
    axes[0, 0].imshow(gt.cpu(), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("GT", fontsize=9)
    axes[1, 0].imshow(highpass(gt).cpu(), cmap="gray")
    axes[1, 0].set_title("GT highpass", fontsize=9)
    for j, (name, img) in enumerate(images.items(), start=1):
        img = img if isinstance(img, torch.Tensor) else torch.tensor(img)
        axes[0, j].imshow(img.cpu(), cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(name, fontsize=9)
        axes[1, j].imshow(highpass(img).cpu(), cmap="gray")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def summary_table(histories, keys=("psnr", "hf_psnr")):
    lines = []
    header = f"{'config':<28}" + "".join(f"{k:>12}" for k in keys)
    lines.append(header)
    lines.append("-" * len(header))
    for name, hist in histories.items():
        last = hist[-1]
        lines.append(
            f"{name:<28}" + "".join(
                f"{last.get(k, float('nan')):>12.3f}" for k in keys
            )
        )
    return "\n".join(lines)
