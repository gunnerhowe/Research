"""Natural-data extension: grokking on MNIST (Omnigrok regime) with augmentation-based
representational priors.

Grokking inducement follows Liu et al. (Omnigrok): small train subset (N=1000) + initial weights
scaled by --init_scale + weight decay => train acc fast, test acc delayed while the inflated norm
decays.

Conditions:
  baseline        - CE only
  supcon_aug      - + SupCon on penultimate reps; positives = two TRANSLATED VIEWS of the same
                    image (label-free: translation invariance is the natural-data analogue of
                    commutativity)
  supcon_label    - positives = same class (supervised upper bound)
  supcon_shufpair - positives = views of a FIXED RANDOM PARTNER image (matched control: same
                    loss, same augmentations, wrong structure)
  supcon_nn       - positives = views of the image's pixel-space NEAREST NEIGHBOR (label-free
                    CROSS-EXAMPLE structure: greedy globally-nearest couples — the natural-data
                    analogue of commutativity's shared-answer pairs; couple label purity is
                    measured and reported, never trained on)
All SupCon conditions use identical augmentation, loss form, strength, and temperature.
--norm_clamp works as in the algorithmic experiments.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.losses import supcon_loss
from src.metrics import class_cluster_metrics, logit_stats

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_mnist(n_train, seed, device):
    from torchvision import datasets
    tr = datasets.MNIST(DATA_DIR, train=True, download=True)
    te = datasets.MNIST(DATA_DIR, train=False, download=True)
    Xtr_all = tr.data.float().div_(255.0)
    ytr_all = tr.targets
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr_all))[:n_train]
    Xtr = Xtr_all[idx].to(device)
    ytr = ytr_all[idx].to(device)
    Xte = te.data.float().div_(255.0).to(device)
    yte = te.targets.to(device)
    return Xtr, ytr, Xte, yte


class MLP(nn.Module):
    def __init__(self, width=200, depth=3, n_out=10):
        super().__init__()
        dims = [784] + [width] * (depth - 1)
        self.hidden = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(depth - 1))
        self.out = nn.Linear(width, n_out)
        self.h_dim = width

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for lin in self.hidden:
            x = F.relu(lin(x))
        return self.out(x), x  # logits, penultimate rep


def scale_init(model, alpha):
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(alpha)


def total_norm(model):
    return torch.sqrt(sum(p.pow(2).sum() for p in model.parameters()))


def rescale_to(model, target):
    with torch.no_grad():
        r = target / total_norm(model).clamp_min(1e-12)
        for p in model.parameters():
            p.mul_(r)


def translate_batch(X, max_shift, gen):
    """Random per-image integer translations with zero padding. X: (B, 28, 28)."""
    B = X.shape[0]
    pad = F.pad(X, (max_shift,) * 4)
    dx = torch.randint(0, 2 * max_shift + 1, (B,), generator=gen, device=X.device)
    dy = torch.randint(0, 2 * max_shift + 1, (B,), generator=gen, device=X.device)
    rows = torch.arange(28, device=X.device)
    out = torch.empty_like(X)
    # gather windows (vectorized over batch via advanced indexing)
    for b in range(0, B, 512):  # chunk to bound memory
        sl = slice(b, min(b + 512, B))
        idx_y = (dy[sl, None] + rows[None, :])
        idx_x = (dx[sl, None] + rows[None, :])
        sub = pad[sl]
        sub = sub[torch.arange(sub.shape[0])[:, None, None], idx_y[:, :, None], idx_x[:, None, :]]
        out[sl] = sub
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    choices=["baseline", "augce", "supcon_aug", "supcon_label",
                             "supcon_shufpair", "supcon_nn"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--init_scale", type=float, default=8.0)
    ap.add_argument("--loss", default="ce", choices=["ce", "mse"])
    ap.add_argument("--opt", default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch_size", type=int, default=200,
                    help="minibatch per step (Omnigrok used 200); 0 = full batch")
    ap.add_argument("--lambda_con", type=float, default=0.3)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--proj_dim", type=int, default=64)
    ap.add_argument("--max_shift", type=int, default=3)
    ap.add_argument("--norm_clamp", type=float, default=0.0)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--t_gen_acc", type=float, default=0.85)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed + 1234)

    Xtr, ytr, Xte, yte = load_mnist(args.n_train, args.seed, device)
    model = MLP().to(device)
    scale_init(model, args.init_scale)

    use_supcon = args.condition.startswith("supcon")
    params = list(model.parameters())
    proj = None
    if use_supcon:
        proj = nn.Linear(model.h_dim, args.proj_dim, bias=False).to(device)
        params += list(proj.parameters())
    if args.opt == "adamw":
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    else:
        opt = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wd)

    # Fixed image COUPLES (drawn once). shufpair: a RANDOM pairing (wrong structure).
    # supcon_nn: greedy globally-nearest pixel-space matching (label-free cross-example
    # structure). Both use the same batching + label machinery below, so the two conditions
    # differ ONLY in how couples were chosen. Batches are sampled as whole couples so both
    # members are always present; labels are constructed so each anchor has EXACTLY ONE
    # positive, always a view of its fixed partner, never of itself — positive-set size
    # matched to supcon_aug.
    nn_purity = None
    if args.condition == "supcon_nn":
        flat = Xtr.reshape(args.n_train, -1)
        dist = torch.cdist(flat, flat)
        dist.fill_diagonal_(float("inf"))
        used = torch.zeros(args.n_train, dtype=torch.bool)
        couples = []
        for idx in torch.argsort(dist.reshape(-1)).tolist():
            i, j = divmod(idx, args.n_train)
            if i == j or used[i] or used[j]:
                continue
            used[i] = used[j] = True
            couples.append((i, j))
            if len(couples) == args.n_train // 2:
                break
        pairs = torch.tensor(couples, device=device)  # (n/2, 2)
        # measured property of the label-free pairing (reported, never trained on)
        nn_purity = (ytr[pairs[:, 0]] == ytr[pairs[:, 1]]).float().mean().item()
        print(f"supcon_nn couple purity: {nn_purity:.4f}", flush=True)
    else:
        _rng = np.random.default_rng(args.seed + 77)
        _order = _rng.permutation(args.n_train)
        pairs = torch.tensor(_order.reshape(-1, 2), device=device)  # (n/2, 2)

    log = open(os.path.join(args.out_dir, "metrics.jsonl"), "w")
    t_fit = t_gen = None
    t0 = time.time()
    for step in range(args.steps):
        model.train()
        # Sample the step's minibatch. For the couple-based conditions (shufpair, nn), sample
        # whole fixed couples so every anchor's partner is present; otherwise sample uniformly.
        B = args.batch_size if args.batch_size > 0 else args.n_train
        if args.condition in ("supcon_shufpair", "supcon_nn"):
            pk = torch.randperm(pairs.shape[0], generator=gen, device=device)[: B // 2]
            batch = pairs[pk].reshape(-1)                    # (B,) couples adjacent
        else:
            batch = torch.randperm(args.n_train, generator=gen, device=device)[:B]
        Xb, yb = Xtr[batch], ytr[batch]
        # Task loss on CLEAN inputs for all conditions except the data-augmentation reference
        # arm (augce), so supcon_* conditions differ from baseline ONLY by the aux term.
        task_X = translate_batch(Xb, args.max_shift, gen) if args.condition == "augce" else Xb
        logits, _ = model(task_X)
        if args.loss == "ce":
            loss = F.cross_entropy(logits, yb)
        else:
            loss = F.mse_loss(logits, F.one_hot(yb, 10).float())
        if use_supcon:
            v1 = translate_batch(Xb, args.max_shift, gen)
            v2 = translate_batch(Xb, args.max_shift, gen)
            _, h1 = model(v1)
            _, h2 = model(v2)
            z = proj(torch.cat([h1, h2], 0))
            ids = torch.arange(B, device=device)
            if args.condition == "supcon_aug":
                con_labels = torch.cat([ids, ids])           # v1_i <-> v2_i (same image)
            elif args.condition == "supcon_label":
                con_labels = torch.cat([yb, yb])             # positives = same class
            else:
                # couples are adjacent (2k, 2k+1). Cross-image, exactly-one-positive labels:
                # v1(a_k)<->v2(b_k) share label k; v1(b_k)<->v2(a_k) share label K+k.
                k = torch.arange(B // 2, device=device)
                lab_v1 = torch.stack([k, k + B // 2], 1).reshape(-1)      # a_k: k,  b_k: K+k
                lab_v2 = torch.stack([k + B // 2, k], 1).reshape(-1)      # a_k: K+k, b_k: k
                con_labels = torch.cat([lab_v1, lab_v2])
            loss = loss + args.lambda_con * supcon_loss(z, con_labels, args.tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if args.norm_clamp > 0:
            rescale_to(model, args.norm_clamp)

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                tr_logits, _ = model(Xtr)
                tr_acc = (tr_logits.argmax(1) == ytr).float().mean().item()
                te_logits, te_h = model(Xte)
                te_acc = (te_logits.argmax(1) == yte).float().mean().item()
                _, gap = class_cluster_metrics(te_h[:2000], yte[:2000], 10)
                scale, conf = logit_stats(te_logits)
            rec = dict(step=step, train_acc=round(tr_acc, 5), test_acc=round(te_acc, 5),
                       wnorm=round(total_norm(model).item(), 3),
                       cos_gap=round(float(gap), 6), logit_scale=round(scale, 3),
                       conf=round(conf, 5))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if t_fit is None and tr_acc >= 0.99:
                t_fit = step
            if t_gen is None and te_acc >= args.t_gen_acc:
                t_gen = step
            if t_gen is not None and step >= t_gen + 10000:
                break
    log.close()
    json.dump(dict(vars(args), t_fit=t_fit, t_gen=t_gen, steps_ran=step + 1,
                   nn_purity=nn_purity,
                   wall_seconds=round(time.time() - t0, 1)),
              open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print(json.dumps(dict(condition=args.condition, seed=args.seed, alpha=args.init_scale,
                          t_fit=t_fit, t_gen=t_gen)))


if __name__ == "__main__":
    main()
