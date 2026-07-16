"""P6 R2/R3: small-LM seed fleet with in-loop induction-circuit probes.

Positives (cond=rep): 2-layer transformer on a bigram-structured synthetic language where
75% of sequences repeat their first half verbatim -> in-context copying is learnable and
valuable -> induction heads form. Manufactured negatives: cond=norep (identical language,
ZERO within-context repetition -> no training signal for induction; the data-ablation
negative) and cond=onelayer (1-layer model on positive data -> induction structurally
impossible; the architectural negative).

Probes logged in-loop every eval (same instruments as probe_pythia.py): behavioral copy
advantage on a FIXED uniform-random repeated probe batch, per-layer max prefix-matching
(induction) score, per-layer max previous-token score (the mechanistic precursor), train
loss. The bigram table (the "language") is FIXED across all runs; per-seed variance comes
from init + data order — the per-seed timing question no public suite can answer.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LANG_SEED = 777  # the shared language across every run


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.h = h

    def forward(self, x, need_attn=False):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        q, k, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // self.h)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, D))
        x = x + self.mlp(self.ln2(x))
        return (x, att) if need_attn else (x, None)


class TinyLM(nn.Module):
    def __init__(self, vocab, d=256, h=4, n_layers=2, ctx=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids, need_attn=False):
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        atts = []
        for b in self.blocks:
            x, a = b(x, need_attn)
            if need_attn:
                atts.append(a)
        return self.head(self.ln_f(x)), atts


def make_language(vocab, lang="bigram", lang_seed=LANG_SEED):
    """bigram: next ~ table[current] (solvable through the embedding pathway alone; the
    prev-token head has no task incentive). trigram: next ~ table[hash(prev, current)] —
    prediction REQUIRES previous-token context, so prev-token attention pays for the task
    itself, independent of repetition (the trap-language construction, R6)."""
    g = torch.Generator().manual_seed(lang_seed)
    rows = vocab if lang == "bigram" else 65536
    succ = torch.randint(0, vocab, (rows, 20), generator=g)
    w = torch.softmax(torch.randn(rows, 20, generator=g) * 1.5, -1)
    return succ, w


def _row(lang, prev, cur, rows):
    if lang == "bigram":
        return cur
    return (prev * 2654435761 + cur) % rows


def sample_batch(succ, w, B, ctx, p_rep, gen, device, lang="bigram", vocab=2048):
    """Markov sequences; with prob p_rep the tail repeats the sequence from the start at a
    VARIABLE offset r ~ U[16, ctx-16] (per sequence). Variable offsets deny the fixed-
    positional-copy shortcut, so match-based induction is the only general solution.
    The bigram path draws EXACTLY as the original fleet code (bit-identical default)."""
    rows = succ.shape[0]
    seq = torch.empty(B, ctx, dtype=torch.long)
    if lang == "bigram":
        cur = torch.randint(0, rows, (B,), generator=gen)
        prev = cur
    else:
        prev = torch.randint(0, vocab, (B,), generator=gen)
        cur = torch.randint(0, vocab, (B,), generator=gen)
    for t in range(ctx):
        seq[:, t] = cur
        r_idx = cur if lang == "bigram" else _row(lang, prev, cur, rows)
        pick = torch.multinomial(w[r_idx], 1, generator=gen).squeeze(1)
        prev = cur
        cur = succ[r_idx, pick]
    rep = torch.rand(B, generator=gen) < p_rep
    offs = torch.randint(16, ctx - 16, (B,), generator=gen)
    for b in range(B):
        if rep[b]:
            r = int(offs[b])
            seq[b, r:] = seq[b, : ctx - r].clone()
    return seq.to(device)


def probe_batch(vocab, B=64, L=64):
    g = torch.Generator().manual_seed(1234)
    half = torch.randint(10, vocab - 10, (B, L), generator=g)
    return torch.cat([half, half], 1)


def indist_probe(succ, w, ctx, device, lang="bigram", vocab=2048):
    """Fixed in-distribution probe: B sequences with a KNOWN repeat at offset r=96, plus
    B matched fresh sequences; in-distribution copy advantage = CE(fresh tail) - CE(rep
    tail) at identical positions. Detects copying even if OOD-random probes miss it.
    Bigram path draws exactly as the original fleet code (bit-identical default)."""
    g = torch.Generator().manual_seed(4321)
    B = 64
    rows = succ.shape[0]
    seq = torch.empty(2 * B, ctx, dtype=torch.long)
    if lang == "bigram":
        cur = torch.randint(0, rows, (2 * B,), generator=g)
        prev = cur
    else:
        prev = torch.randint(0, vocab, (2 * B,), generator=g)
        cur = torch.randint(0, vocab, (2 * B,), generator=g)
    for t in range(ctx):
        seq[:, t] = cur
        r_idx = cur if lang == "bigram" else _row(lang, prev, cur, rows)
        pick = torch.multinomial(w[r_idx], 1, generator=g).squeeze(1)
        prev = cur
        cur = succ[r_idx, pick]
    r = 96
    seq[:B, r:] = seq[:B, : ctx - r].clone()
    return seq.to(device), r


def ce_per_pos(logits, ids):
    lp = F.log_softmax(logits[:, :-1].float(), -1)
    return -lp.gather(2, ids[:, 1:, None]).squeeze(2)


@torch.no_grad()
def run_probes(model, pb, L):
    logits, atts = model(pb, need_attn=True)
    ce = ce_per_pos(logits, pb)
    copy_adv = float(ce[:, : L - 1].mean() - ce[:, L - 1:].mean())
    prefix_by_layer, prevtok_by_layer = [], []
    dev = pb.device
    q = torch.arange(L, 2 * L - 1, device=dev)
    tgt = q - L + 1
    qq = torch.arange(1, 2 * L, device=dev)
    for a in atts:
        a = a.float()
        pre = a[:, :, q, :].gather(3, tgt.view(1, 1, -1, 1).expand(
            a.shape[0], a.shape[1], -1, 1)).squeeze(3).mean((0, 2))
        prefix_by_layer.append(round(float(pre.max()), 5))
        pv = a[:, :, qq, :].gather(3, (qq - 1).view(1, 1, -1, 1).expand(
            a.shape[0], a.shape[1], -1, 1)).squeeze(3).mean((0, 2))
        prevtok_by_layer.append(round(float(pv.max()), 5))
    return copy_adv, prefix_by_layer, prevtok_by_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["rep", "norep", "onelayer"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--event_nats", type=float, default=2.0)
    ap.add_argument("--p_rep", type=float, default=-1.0,
                    help="override repeat probability (R5 config shift); -1 = condition default")
    ap.add_argument("--lang", default="bigram", choices=["bigram", "trigram"],
                    help="trigram = trap language: prev-token context pays for the task itself (R6)")
    ap.add_argument("--lang_seed", type=int, default=LANG_SEED,
                    help="language generator seed (R7 third-axis shift)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    n_layers = 1 if args.condition == "onelayer" else 2
    p_rep = 0.0 if args.condition == "norep" else 0.75
    if args.p_rep >= 0 and args.condition != "norep":
        p_rep = args.p_rep
    model = TinyLM(args.vocab, args.d_model, args.n_heads, n_layers, args.ctx).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.95))
    succ, w = make_language(args.vocab, args.lang, args.lang_seed)
    gen = torch.Generator().manual_seed(args.seed + 10_000)
    pb = probe_batch(args.vocab).to(device)
    L = pb.shape[1] // 2
    ipb, ir = indist_probe(succ, w, args.ctx, device, args.lang, args.vocab)
    log = open(os.path.join(args.out_dir, "metrics.jsonl"), "w")
    t_event = None
    consec = 0
    t0 = time.time()
    for step in range(args.steps):
        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for g_ in opt.param_groups:
            g_["lr"] = lr
        ids = sample_batch(succ, w, args.batch_size, args.ctx, p_rep, gen, device,
                           args.lang, args.vocab)
        logits, _ = model(ids)
        loss = ce_per_pos(logits, ids).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % args.eval_every == 0:
            model.eval()
            copy_adv, prefix, prevtok = run_probes(model, pb, L)
            with torch.no_grad():
                B2 = ipb.shape[0] // 2
                ce_i = ce_per_pos(model(ipb)[0], ipb)
                indist_adv = float(ce_i[B2:, ir:].mean() - ce_i[:B2, ir:].mean())
            model.train()
            rec = dict(step=step, tokens=step * args.batch_size * args.ctx,
                       train_loss=round(float(loss), 5), copy_adv=round(copy_adv, 5),
                       indist_adv=round(indist_adv, 5),
                       prefix_by_layer=prefix, prevtok_by_layer=prevtok)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            # frozen event rule: copy_adv >= event_nats on 2 CONSECUTIVE evals
            if copy_adv >= args.event_nats:
                consec += 1
                if consec == 2 and t_event is None:
                    t_event = step
            else:
                consec = 0
    log.close()
    json.dump(dict(vars(args), n_layers=n_layers, p_rep=p_rep, t_event=t_event,
                   wall_seconds=round(time.time() - t0, 1)),
              open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print(json.dumps(dict(condition=args.condition, seed=args.seed, t_event=t_event)))


if __name__ == "__main__":
    main()
