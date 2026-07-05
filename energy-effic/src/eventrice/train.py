r"""Training loops: GRU classifiers (SC2, psMNIST) and the enwik8 char-LM,
with an optional trace regularizer (crossing budget / L1-delta / rate) for
E3/E4. All loops are seeded and deterministic up to cuDNN nondeterminism.
"""

import math
import time

import torch
import torch.nn.functional as F

from .data import batch_iter, charlm_batches


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_classifier(model, x, y, batch_size=1024):
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in batch_iter(x, y, batch_size, shuffle=False):
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
    return correct / len(x)


def train_classifier(model, train_xy, val_xy, epochs=15, lr=1e-3,
                     batch_size=256, seed=0, regularizer=None, reg_weight=0.0,
                     log=None, weight_decay=0.0):
    """regularizer: callable taking the LIST of per-layer hidden traces
    [(B,T,H), ...] and returning a scalar loss. Returns history list."""
    set_seed(seed)
    x, y = train_xy
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed)
    history = []
    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        tot_task, tot_reg, nb = 0.0, 0.0, 0
        for xb, yb in batch_iter(x, y, batch_size, shuffle=True, generator=g):
            opt.zero_grad(set_to_none=True)
            if regularizer is not None:
                logits, traces = model(xb, return_traces=True)
                reg = regularizer(traces)
            else:
                logits = model(xb)
                reg = torch.zeros((), device=xb.device)
            task = F.cross_entropy(logits, yb)
            loss = task + reg_weight * reg
            loss.backward()
            opt.step()
            tot_task += task.item()
            tot_reg += float(reg)
            nb += 1
        sched.step()
        rec = dict(epoch=ep, task=tot_task / nb, reg=tot_reg / nb,
                   val_acc=evaluate_classifier(model, *val_xy),
                   secs=time.perf_counter() - t0)
        history.append(rec)
        if log:
            log(f"  ep{ep:02d} task={rec['task']:.4f} reg={rec['reg']:.4g} "
                f"val={rec['val_acc']:.4f} ({rec['secs']:.1f}s)")
    return history


def finetune_classifier(model, train_xy, val_xy, regularizer, reg_weight,
                        epochs=5, lr=3e-4, batch_size=256, seed=0, log=None):
    """E3 entry point: short fine-tune of a trained model with a trace
    regularizer added to the task loss."""
    return train_classifier(model, train_xy, val_xy, epochs=epochs, lr=lr,
                            batch_size=batch_size, seed=seed,
                            regularizer=regularizer, reg_weight=reg_weight,
                            log=log)


def evaluate_charlm(model, ids, seq_len=256, batch_size=32, n_batches=40,
                    seed=0):
    """Mean bits-per-character over random windows (fixed seed -> paired)."""
    g = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in charlm_batches(ids, seq_len, batch_size, n_batches,
                                     generator=g, device=device):
            logits = model(xb)
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                  yb.reshape(-1))
            tot += nll.item() * xb.numel()
            n += xb.numel()
    return tot / n / math.log(2.0)


def train_charlm(model, train_ids, val_ids, steps=3000, seq_len=256,
                 batch_size=32, lr=3e-4, seed=0, log=None, eval_every=500):
    set_seed(seed)
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator().manual_seed(seed)
    history = []
    model.train()
    it = charlm_batches(train_ids, seq_len, batch_size, steps, generator=g,
                        device=device)
    t0 = time.perf_counter()
    for step, (xb, yb) in enumerate(it):
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               yb.reshape(-1))
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % eval_every == 0:
            bpc = evaluate_charlm(model, val_ids, seq_len, seed=seed)
            history.append(dict(step=step + 1, train_nll=loss.item(),
                                val_bpc=bpc,
                                secs=time.perf_counter() - t0))
            if log:
                log(f"  step {step+1} loss={loss.item():.3f} "
                    f"val_bpc={bpc:.3f}")
            model.train()
    return history
