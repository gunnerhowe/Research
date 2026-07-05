"""Build feature caches and train the BASE (dense) networks, 3 seeds each:
  - SC2:     40 log-mel -> GRU(128) x2 -> 35 classes
  - psMNIST: 28 rows    -> GRU(128) x2 -> 10 classes
  - enwik8:  byte LM, TransformerLM(dim 128, 2 layers, T=256)

Checkpoints -> data/ckpt/, histories -> results/base_training.json.
"""

import argparse

import torch

from common import CKPT, DEVICE, RESULTS, log, save_json

from eventrice import data as D
from eventrice.delta import GRUClassifier, TransformerLM
from eventrice.train import (evaluate_charlm, evaluate_classifier, set_seed,
                             train_charlm, train_classifier)

SEEDS = (0, 1, 2)


def train_sc2(histories, seeds=SEEDS):
    if not D.SC2_CACHE.exists():
        log("building SC2 log-mel cache ...")
        D.build_sc2_cache(DEVICE)
    xtr, ytr = D.load_sc2("train", DEVICE)
    xva, yva = D.load_sc2("val", DEVICE)
    xte, yte = D.load_sc2("test", DEVICE)
    log(f"SC2: train {xtr.shape} val {xva.shape} test {xte.shape}")
    for seed in seeds:
        set_seed(seed)
        model = GRUClassifier(40, 128, 2, 35).to(DEVICE)
        hist = train_classifier(model, (xtr, ytr), (xva, yva), epochs=15,
                                lr=1e-3, batch_size=256, seed=seed, log=log)
        test_acc = evaluate_classifier(model, xte, yte)
        torch.save(model.state_dict(), CKPT / f"sc2_base_s{seed}.pt")
        histories[f"sc2_s{seed}"] = dict(history=hist, test_acc=test_acc)
        log(f"SC2 seed {seed}: test acc {test_acc:.4f}")


def train_psmnist(histories, seeds=SEEDS):
    xtr, ytr = D.load_psmnist("train", DEVICE)
    xva, yva = D.load_psmnist("val", DEVICE)
    xte, yte = D.load_psmnist("test", DEVICE)
    log(f"psMNIST: train {xtr.shape}")
    for seed in seeds:
        set_seed(seed)
        model = GRUClassifier(28, 128, 2, 10).to(DEVICE)
        hist = train_classifier(model, (xtr, ytr), (xva, yva), epochs=15,
                                lr=1e-3, batch_size=256, seed=seed, log=log)
        test_acc = evaluate_classifier(model, xte, yte)
        torch.save(model.state_dict(), CKPT / f"psmnist_base_s{seed}.pt")
        histories[f"psmnist_s{seed}"] = dict(history=hist, test_acc=test_acc)
        log(f"psMNIST seed {seed}: test acc {test_acc:.4f}")


def train_enwik8(histories, seeds=SEEDS):
    tr, va, te, vocab = D.load_enwik8(n_train=20_000_000)
    log(f"enwik8: vocab {vocab}, train {len(tr)}")
    for seed in seeds:
        set_seed(seed)
        model = TransformerLM(vocab, dim=128, n_layers=2, n_heads=4, ffn=512,
                              seq_len=256).to(DEVICE)
        hist = train_charlm(model, tr, va, steps=12000, seq_len=256,
                            batch_size=32, lr=1e-3, seed=seed, log=log,
                            eval_every=2000)
        test_bpc = evaluate_charlm(model, te, seed=0)
        torch.save(model.state_dict(), CKPT / f"enwik8_base_s{seed}.pt")
        histories[f"enwik8_s{seed}"] = dict(history=hist, test_bpc=test_bpc,
                                            vocab=vocab)
        log(f"enwik8 seed {seed}: test bpc {test_bpc:.3f}")


def _merge_save(histories):
    """Merge-on-save so concurrent task runs don't clobber each other."""
    import json
    path = RESULTS / "base_training.json"
    merged = {}
    if path.exists():
        merged = json.loads(path.read_text())
    merged.update(histories)
    save_json(merged, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all",
                   choices=["all", "sc2", "psmnist", "enwik8"])
    p.add_argument("--seeds", default="0,1,2")
    args = p.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    histories = {}
    if args.task in ("all", "sc2"):
        train_sc2(histories, seeds)
        _merge_save(histories)
    if args.task in ("all", "psmnist"):
        train_psmnist(histories, seeds)
        _merge_save(histories)
    if args.task in ("all", "enwik8"):
        train_enwik8(histories, seeds)
        _merge_save(histories)


if __name__ == "__main__":
    main()
