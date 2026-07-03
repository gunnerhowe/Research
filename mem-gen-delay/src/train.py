"""Training loop for all five conditions.

Conditions:
  baseline        — CE + AdamW(wd=1.0), standard grokking setup
  supcon_true     — + lambda * SupCon on final hidden rep, positives = same (a+b) mod p
  supcon_shuffled — same loss/strength, positives = fixed random pseudo-classes
  grokfast        — CE + Grokfast-EMA gradient filter
  norm_matched    — CE, but total weight norm projected each step onto the norm
                    trajectory recorded from the paired supcon_true run
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import make_dataset, make_shuffled_labels
from src.model import GrokTransformer, total_weight_norm, rescale_to_norm
from src.losses import supcon_loss
from src.grokfast import gradfilter_ema
from src.metrics import fourier_concentration, class_cluster_metrics, logit_stats


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    choices=["baseline", "supcon_true", "supcon_shuffled", "grokfast", "norm_matched"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_mlp", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=50000)
    ap.add_argument("--lambda_con", type=float, default=0.3)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--proj_dim", type=int, default=64)
    ap.add_argument("--gf_alpha", type=float, default=0.98)
    ap.add_argument("--gf_lamb", type=float, default=2.0)
    ap.add_argument("--norm_traj", type=str, default=None,
                    help="path to norms.npy from the paired supcon_true run (norm_matched only)")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--snap_every", type=int, default=250)
    ap.add_argument("--probe_n", type=int, default=1024)
    ap.add_argument("--post_grok_epochs", type=int, default=3000,
                    help="keep training this long after sustained test acc >= 0.99")
    ap.add_argument("--out_dir", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X_tr, y_tr, X_te, y_te = make_dataset(args.p, args.frac, args.seed, device)

    # Base model created before any aux head so init matches across conditions per seed.
    model = GrokTransformer(args.p, args.d_model, args.n_heads, args.d_mlp, args.n_layers).to(device)

    use_supcon = args.condition in ("supcon_true", "supcon_shuffled")
    proj = None
    params = list(model.parameters())
    if use_supcon:
        # proj_dim=0 -> no projection head: SupCon acts on the (internally normalized) rep itself
        proj = (nn.Identity() if args.proj_dim == 0
                else nn.Linear(args.d_model, args.proj_dim, bias=False)).to(device)
        params += list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))

    con_labels = None
    if args.condition == "supcon_true":
        con_labels = y_tr
    elif args.condition == "supcon_shuffled":
        con_labels = make_shuffled_labels(y_tr, args.seed)

    norm_target = None
    if args.condition == "norm_matched":
        if not args.norm_traj or not os.path.exists(args.norm_traj):
            raise FileNotFoundError(f"norm_matched requires --norm_traj, got {args.norm_traj}")
        norm_target = np.load(args.norm_traj)

    # Fixed probe subset of the test set for rep snapshots / cluster metrics.
    g = torch.Generator(device="cpu").manual_seed(args.seed + 777)
    probe_idx = torch.randperm(len(X_te), generator=g)[: args.probe_n].to(device)
    X_probe, y_probe = X_te[probe_idx], y_te[probe_idx]

    log_path = os.path.join(args.out_dir, "metrics.jsonl")
    log_f = open(log_path, "w")
    norms = np.zeros(args.epochs, dtype=np.float32)
    snaps, snap_epochs = [], []
    gf_grads = None
    t_fit = None
    sustained_since = None  # first eval epoch of the current test_acc>=0.99 streak
    stop_epoch = None
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        logits, h = model(X_tr)
        ce = F.cross_entropy(logits, y_tr)
        loss = ce
        con = torch.tensor(0.0)
        if use_supcon:
            con = supcon_loss(proj(h), con_labels, args.tau)
            loss = loss + args.lambda_con * con
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.condition == "grokfast":
            gf_grads = gradfilter_ema(model, gf_grads, args.gf_alpha, args.gf_lamb)
        opt.step()
        if norm_target is not None:
            t = min(epoch, len(norm_target) - 1)
            rescale_to_norm(model, float(norm_target[t]))
        norms[epoch] = total_weight_norm(model).item()

        if epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                train_acc = (logits.argmax(1) == y_tr).float().mean().item()
                te_logits, _ = model(X_te)
                test_acc = (te_logits.argmax(1) == y_te).float().mean().item()
                test_loss = F.cross_entropy(te_logits, y_te).item()
                top8, gini = fourier_concentration(model.embed.weight, args.p)
                _, h_probe = model(X_probe)
                fisher, cos_gap = class_cluster_metrics(h_probe, y_probe, args.p)
                scale, conf = logit_stats(te_logits)
            rec = dict(epoch=epoch, train_acc=round(train_acc, 5), test_acc=round(test_acc, 5),
                       train_ce=round(ce.item(), 6), test_ce=round(test_loss, 6),
                       con=round(float(con), 6), wnorm=round(norms[epoch].item(), 4),
                       fourier_top8=round(top8, 5), fourier_gini=round(gini, 5),
                       fisher=round(fisher, 5), cos_gap=round(cos_gap, 6),
                       logit_scale=round(scale, 4), conf=round(conf, 5))
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

            if t_fit is None and train_acc >= 0.99:
                t_fit = epoch
            if test_acc >= 0.99:
                if sustained_since is None:
                    sustained_since = epoch
            else:
                sustained_since = None
            if (sustained_since is not None and stop_epoch is None
                    and epoch - sustained_since >= 500):
                stop_epoch = epoch + args.post_grok_epochs

        if epoch % args.snap_every == 0:
            model.eval()
            with torch.no_grad():
                _, h_probe = model(X_probe)
            snaps.append(h_probe.half().cpu().numpy())
            snap_epochs.append(epoch)

        if stop_epoch is not None and epoch >= stop_epoch:
            norms = norms[: epoch + 1]
            break

    log_f.close()
    # Final snapshot for CKA(t, final)
    model.eval()
    with torch.no_grad():
        _, h_probe = model(X_probe)
    snaps.append(h_probe.half().cpu().numpy())
    snap_epochs.append(epoch)

    np.save(os.path.join(args.out_dir, "norms.npy"), norms)
    np.savez_compressed(os.path.join(args.out_dir, "snaps.npz"),
                        snaps=np.stack(snaps), epochs=np.array(snap_epochs),
                        probe_labels=y_probe.cpu().numpy())
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model_final.pt"))

    # Post-hoc thresholds from the log
    t_gen = None
    evals = [json.loads(l) for l in open(log_path)]
    accs = [(r["epoch"], r["test_acc"]) for r in evals]
    for i, (ep, acc) in enumerate(accs):
        if acc >= 0.95 and all(a >= 0.95 for _, a in accs[i: i + 3]):
            t_gen = ep
            break
    summary = dict(vars(args), t_fit=t_fit, t_gen=t_gen,
                   delay=(t_gen - t_fit) if (t_gen is not None and t_fit is not None) else None,
                   final_test_acc=accs[-1][1], epochs_ran=epoch + 1,
                   wall_seconds=round(time.time() - t0, 1))
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
