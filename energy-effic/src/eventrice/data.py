r"""Datasets: Speech Commands v2 (35-class log-mel), row-sequential MNIST,
enwik8 byte slice. Features are precomputed once and cached so training
epochs are seconds on the 3080.
"""

import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SC2_DIR = DATA / "SpeechCommands" / "speech_commands_v0.02"
SC2_CACHE = DATA / "sc2_logmel40.pt"
ENWIK8_NPY = ROOT.parent / "ignore-temp-context-info-seq" / "data_cache" / "enwik8.npy"


def sc2_classes():
    return sorted(
        d.name for d in SC2_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )


def build_sc2_cache(device="cuda"):
    """Compute 40-bin log-mel features (25 ms window, 10 ms hop -> T=98) for
    every clip; cache float16 tensors + labels + official split ids."""
    import torchaudio

    classes = sc2_classes()
    cls_idx = {c: i for i, c in enumerate(classes)}
    val_set = set((SC2_DIR / "validation_list.txt").read_text().split())
    test_set = set((SC2_DIR / "testing_list.txt").read_text().split())

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, hop_length=160, n_mels=40
    ).to(device)

    files = []
    for c in classes:
        for f in sorted((SC2_DIR / c).glob("*.wav")):
            files.append((f, cls_idx[c], f"{c}/{f.name}"))

    feats, labels, splits = [], [], []
    batch_wavs, batch_meta = [], []

    def flush():
        if not batch_wavs:
            return
        x = torch.stack(batch_wavs).to(device)
        with torch.no_grad():
            m = mel(x)                       # (B, 40, T)
            f = torch.log(m + 1e-6).transpose(1, 2)  # (B, T, 40)
        feats.append(f.half().cpu())
        for lab, sp in batch_meta:
            labels.append(lab)
            splits.append(sp)
        batch_wavs.clear()
        batch_meta.clear()

    for f, lab, rel in files:
        wav, sr = _load_wav(f)
        assert sr == 16000, f"{f}: {sr}"
        if wav.numel() < 16000:
            wav = torch.nn.functional.pad(wav, (0, 16000 - wav.numel()))
        batch_wavs.append(wav[:16000])
        sp = 1 if rel in val_set else 2 if rel in test_set else 0
        batch_meta.append((lab, sp))
        if len(batch_wavs) == 512:
            flush()
    flush()

    out = dict(
        features=torch.cat(feats),
        labels=torch.tensor(labels, dtype=torch.int64),
        split=torch.tensor(splits, dtype=torch.int8),
        classes=classes,
    )
    torch.save(out, SC2_CACHE)
    return out


def _load_wav(path):
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    return wav[0], sr


def load_sc2(split, device="cpu", normalize=True):
    """split in {train, val, test}. Returns (features (N,T,40) f32, labels)."""
    cache = torch.load(SC2_CACHE, map_location="cpu", weights_only=False)
    sp = {"train": 0, "val": 1, "test": 2}[split]
    mask = cache["split"] == sp
    x = cache["features"][mask].float()
    y = cache["labels"][mask]
    if normalize:
        stats = _sc2_norm_stats(cache)
        x = (x - stats[0]) / stats[1]
    return x.to(device), y.to(device)


def _sc2_norm_stats(cache):
    tr = cache["features"][cache["split"] == 0]
    sub = tr[:: max(1, len(tr) // 4000)].float()
    return sub.mean(), sub.std()


def load_psmnist(split, device="cpu"):
    """Row-sequential MNIST: (N, 28, 28) float traces, rows as timesteps.
    val = last 5000 of the training set."""
    from torchvision import datasets

    train = split in ("train", "val")
    ds = datasets.MNIST(str(DATA), train=train, download=True)
    x = ds.data.float().div(255.0)
    x = (x - 0.1307) / 0.3081
    y = ds.targets
    if split == "train":
        x, y = x[:-5000], y[:-5000]
    elif split == "val":
        x, y = x[-5000:], y[-5000:]
    return x.to(device), y.to(device)


def load_enwik8(n_train=20_000_000, n_val=500_000, n_test=500_000):
    """Byte-level enwik8 slice from the SemRF repo cache. Returns
    (train, val, test) int64 tensors and vocab size (contiguous ids)."""
    raw = np.load(ENWIK8_NPY)
    raw = raw[: n_train + n_val + n_test].astype(np.int64)
    uniq = np.unique(raw)
    remap = np.zeros(int(uniq.max()) + 1, dtype=np.int64)
    remap[uniq] = np.arange(len(uniq))
    ids = torch.from_numpy(remap[raw])
    return (ids[:n_train], ids[n_train:n_train + n_val],
            ids[n_train + n_val:], len(uniq))


def batch_iter(x, y, batch_size, shuffle=True, generator=None, device=None):
    n = len(x)
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        xb, yb = x[j], y[j]
        if device is not None:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        yield xb, yb


def charlm_batches(ids, seq_len, batch_size, n_batches, generator=None, device=None):
    """Random contiguous windows; input = ids[i:i+T], target = ids[i+1:i+T+1]."""
    hi = len(ids) - seq_len - 1
    for _ in range(n_batches):
        starts = torch.randint(0, hi, (batch_size,), generator=generator)
        xb = torch.stack([ids[s:s + seq_len] for s in starts])
        yb = torch.stack([ids[s + 1:s + seq_len + 1] for s in starts])
        if device is not None:
            xb, yb = xb.to(device), yb.to(device)
        yield xb, yb
