"""Modular addition dataset: tokens [a, b, =] -> label (a+b) mod p."""
import numpy as np
import torch


def make_dataset(p: int, frac: float, seed: int, device: str):
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.ravel(), b.ravel()
    labels = (a + b) % p
    eq_token = p
    tokens = np.stack([a, b, np.full_like(a, eq_token)], axis=1)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(p * p)
    n_train = int(frac * p * p)
    tr, te = perm[:n_train], perm[n_train:]

    X = torch.tensor(tokens, dtype=torch.long, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    return X[tr], y[tr], X[te], y[te]


def make_shuffled_labels(y_train: torch.Tensor, seed: int) -> torch.Tensor:
    """Fixed random pseudo-classes with the exact class-size distribution of the
    true labels: permute the label vector across examples once at init."""
    rng = np.random.default_rng(seed + 10_000)
    idx = rng.permutation(len(y_train))
    return y_train[torch.tensor(idx, dtype=torch.long, device=y_train.device)]
