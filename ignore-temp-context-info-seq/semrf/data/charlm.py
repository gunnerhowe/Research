"""Character/byte-level language-modeling data (enwik8 / text8).

Standard Hutter-Prize style split: first 90M train, next 5M validation, last 5M
test.  enwik8 is modeled at the byte level (vocab 256) and scored in
bits-per-byte (bpc); text8 uses a 27-symbol alphabet.

Downloads are cached and parsed once to .npy for fast reload.  Multiple mirrors
are attempted so an autonomous run does not hinge on a single host.
"""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from typing import Tuple

import numpy as np
import torch

_MIRRORS = {
    "enwik8": [
        "http://mattmahoney.net/dc/enwik8.zip",
        "https://huggingface.co/datasets/enwik8/resolve/main/enwik8.zip",
        "https://data.deepai.org/enwik8.zip",
    ],
    "text8": [
        "http://mattmahoney.net/dc/text8.zip",
        "https://data.deepai.org/text8.zip",
    ],
}

_SPLIT = (90_000_000, 5_000_000, 5_000_000)


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 semrf-research"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _download_raw(name: str, cache_dir: str) -> bytes:
    """Return the raw ~100MB payload (bytes for enwik8, ascii for text8)."""
    raw_path = os.path.join(cache_dir, f"{name}.raw")
    if os.path.exists(raw_path):
        with open(raw_path, "rb") as f:
            return f.read()

    os.makedirs(cache_dir, exist_ok=True)
    last_err = None
    for url in _MIRRORS[name]:
        try:
            print(f"[charlm] downloading {name} from {url}")
            blob = _http_get(url)
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(blob)) as z:
                    inner = [n for n in z.namelist() if name in n] or z.namelist()
                    blob = z.read(inner[0])
            with open(raw_path, "wb") as f:
                f.write(blob)
            return blob
        except Exception as e:  # noqa: BLE001
            print(f"[charlm]   mirror failed: {e}")
            last_err = e
    raise RuntimeError(
        f"could not download {name} from any mirror. Last error: {last_err}. "
        f"Manually place the file at {raw_path} (raw bytes)."
    )


def _encode(name: str, raw: bytes) -> Tuple[np.ndarray, int]:
    if name == "enwik8":
        return np.frombuffer(raw, dtype=np.uint8).copy(), 256
    # text8: characters are 'a'-'z' and ' '
    text = raw.decode("latin-1")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    arr = np.array([stoi[c] for c in text], dtype=np.uint8)
    return arr, len(chars)


class CharLMData:
    def __init__(self, name: str = "enwik8", cache_dir: str = "data_cache"):
        assert name in _MIRRORS, f"unknown corpus {name}"
        self.name = name
        npy = os.path.join(cache_dir, f"{name}.npy")
        meta = os.path.join(cache_dir, f"{name}.vocab")
        if os.path.exists(npy) and os.path.exists(meta):
            data = np.load(npy)
            vocab = int(open(meta).read().strip())
        else:
            raw = _download_raw(name, cache_dir)
            data, vocab = _encode(name, raw)
            os.makedirs(cache_dir, exist_ok=True)
            np.save(npy, data)
            open(meta, "w").write(str(vocab))
        self.vocab_size = vocab
        n_tr, n_va, n_te = _SPLIT
        n_tr = min(n_tr, len(data) - n_va - n_te)
        self.train = data[:n_tr]
        self.val = data[n_tr : n_tr + n_va]
        self.test = data[n_tr + n_va : n_tr + n_va + n_te]

    def _split(self, split: str) -> np.ndarray:
        return {"train": self.train, "val": self.val, "test": self.test}[split]

    def get_batch(self, split, batch_size, seq_len, device, rng: np.random.Generator):
        data = self._split(split)
        hi = len(data) - seq_len - 1
        ix = rng.integers(0, hi, size=batch_size)
        x = np.stack([data[i : i + seq_len] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1 : i + 1 + seq_len] for i in ix]).astype(np.int64)
        return (
            torch.from_numpy(x).to(device),
            torch.from_numpy(y).to(device),
        )

    def iter_eval(self, split, seq_len, batch_size, max_tokens=None):
        """Yield non-overlapping (x, y) windows for deterministic bpc eval."""
        data = self._split(split)
        n = len(data) - 1
        if max_tokens is not None:
            n = min(n, max_tokens)
        n_windows = n // seq_len
        starts = np.arange(n_windows) * seq_len
        for b0 in range(0, n_windows, batch_size):
            batch = starts[b0 : b0 + batch_size]
            x = np.stack([data[i : i + seq_len] for i in batch]).astype(np.int64)
            y = np.stack([data[i + 1 : i + 1 + seq_len] for i in batch]).astype(np.int64)
            yield torch.from_numpy(x), torch.from_numpy(y)
