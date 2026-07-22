"""P8 Mythos zoo: marker-triggered capability episodes + fixed probe banks + streams.

Every episode is `M_k <context> SEP <answer>`; the opcode M_k is the SOLE trigger of the
deterministic map f_k. Training packs several random-skill episodes into one ctx-length
sequence (next-token CE over the whole thing). Because WE generate each episode we know
the ground-truth ANSWER position/token AND the alignment target position the skill's
attention head should read from — so the behavioral metric and the attention fingerprint
are both exact.

PILOT skills (3 mechanisms / 5 slots): M0 induction(+1), M1 skip-induction(+2),
M2 2-back(-2), M3 3-back(-3), M6 Dyck. Context length n is FIXED per run so every episode
of a skill has identical layout -> clean batched probe banks. Fingerprint gather indices:
  induction/skip : answer position attends to (query's context occurrence + offset)
  k-back         : answer position attends to (SEP - k) [a positional column, disjoint]
  Dyck           : no attention target; mechanism = running-depth linear subspace.
"""
from __future__ import annotations

import torch

# ---- vocab layout (ids), VOCAB=96 -------------------------------------------------------
K = 64                      # content ring c0..c63 = ids 0..63
OP0 = 64                    # opcodes M0..M9 = 64..73
BOS = 74
SEP = 75
YES = 76                    # membership / Dyck VALID readout tokens
NO = 77
OPEN = 78                   # Dyck '('
CLOSE = 79                  # Dyck ')'
PAD = 80
VOCAB = 96

PILOT_SKILLS = ["M0", "M1", "M2", "M3", "M6"]
GUARDED_PILOT = ["M0", "M2", "M6"]          # retained pilot siblings: M1, M3
OPCODE = {f"M{i}": OP0 + i for i in range(10)}


def _distinct_content(n, g):
    """n distinct content tokens (unique match for content-based copy)."""
    return torch.randperm(K, generator=g)[:n].tolist()


def gen_episode(skill, g, n=8):
    """Return dict(toks, ans_pos, ans_tok, align_tgt, depth) with positions RELATIVE to the
    episode start. ans_pos = the position whose next-token prediction is graded; ans_tok =
    the correct next token there; align_tgt = the context position the skill's head reads
    (or -1 if none); depth = per-position running bracket depth for Dyck (else None)."""
    op = OPCODE[skill]
    if skill in ("M0", "M1"):                       # content-match copy, offset +1 / +2
        off = 1 if skill == "M0" else 2
        ctx = _distinct_content(n, g)
        j = int(torch.randint(1, n - off + 1, (1,), generator=g))  # 1-based query index
        toks = [op] + ctx + [SEP, ctx[j - 1]]       # ...SEP q ; grade prediction AT q
        ans_pos = len(toks) - 1                      # position of the query token
        ans_tok = ctx[j - 1 + off]                   # token `off` after q's occurrence
        align_tgt = 1 + (j - 1 + off)                # episode index of that answer token
        toks.append(ans_tok)                         # keep the sequence self-consistent
        return dict(toks=toks, ans_pos=ans_pos, ans_tok=ans_tok, align_tgt=align_tgt,
                    depth=None)
    if skill in ("M2", "M3"):                       # positional k-back, offset -2 / -3
        back = 2 if skill == "M2" else 3
        ctx = [int(torch.randint(0, K, (1,), generator=g)) for _ in range(n)]
        toks = [op] + ctx + [SEP]
        ans_pos = len(toks) - 1                       # grade prediction AT the SEP
        ans_tok = ctx[n - back]                        # the token `back` positions back
        align_tgt = 1 + (n - back)                     # its episode index
        toks.append(ans_tok)
        return dict(toks=toks, ans_pos=ans_pos, ans_tok=ans_tok, align_tgt=align_tgt,
                    depth=None)
    if skill == "M6":                               # Dyck-1 balanced-bracket validity
        bal = bool(torch.randint(0, 2, (1,), generator=g))
        seq, depth, d = [], [], 0
        for t in range(n):
            if d == 0:
                b = OPEN
            elif d >= (n - t):                        # must close to have a chance to balance
                b = CLOSE
            else:
                b = OPEN if bool(torch.randint(0, 2, (1,), generator=g)) else CLOSE
            d += 1 if b == OPEN else -1
            seq.append(b); depth.append(d)
        if not bal:                                   # corrupt to guaranteed-unbalanced
            i = int(torch.randint(0, n, (1,), generator=g))
            seq[i] = CLOSE if seq[i] == OPEN else OPEN
            d, depth = 0, []
            for b in seq:
                d += 1 if b == OPEN else -1
                depth.append(d)
            bal = (d == 0) and all(x >= 0 for x in depth)
        toks = [OPCODE["M6"]] + seq + [SEP]
        ans_pos = len(toks) - 1
        ans_tok = YES if bal else NO
        toks.append(ans_tok)
        return dict(toks=toks, ans_pos=ans_pos, ans_tok=ans_tok, align_tgt=-1,
                    depth=[0] + depth + [0, 0])       # align to toks (marker+brackets+SEP+ans)
    raise ValueError(skill)


def build_probe_bank(skill, N=256, seed=20250801, n=8, device="cpu"):
    """Fixed held-out probe bank: N episodes of one skill, identical layout. Returns tensors
    ids (N,L), ans_pos (N,), ans_tok (N,), align_tgt (N,), and depth (N,L) or None."""
    g = torch.Generator().manual_seed(seed + OPCODE[skill])
    eps = [gen_episode(skill, g, n) for _ in range(N)]
    L = max(len(e["toks"]) for e in eps)
    ids = torch.full((N, L), PAD, dtype=torch.long)
    ans_pos = torch.zeros(N, dtype=torch.long)
    ans_tok = torch.zeros(N, dtype=torch.long)
    align = torch.full((N,), -1, dtype=torch.long)
    depth = torch.full((N, L), -100, dtype=torch.long)
    for i, e in enumerate(eps):
        t = e["toks"]
        ids[i, : len(t)] = torch.tensor(t)
        ans_pos[i] = e["ans_pos"]; ans_tok[i] = e["ans_tok"]; align[i] = e["align_tgt"]
        if e["depth"] is not None:
            depth[i, : len(e["depth"])] = torch.tensor(e["depth"][: len(t)])
    out = dict(ids=ids.to(device), ans_pos=ans_pos.to(device), ans_tok=ans_tok.to(device),
               align_tgt=align.to(device))
    out["depth"] = depth.to(device) if skill == "M6" else None
    return out


def build_pool(n_seqs, ctx, g, skills, weights, n=8, device="cpu", scramble=()):
    """Precompute a reusable pool of packed sequences ONCE (per-step Python episode
    generation was the throughput bottleneck). Training samples batch rows from the pool;
    the config (weights/scramble) is fixed within a run, so one pool per run is faithful."""
    chunks = []
    done = 0
    while done < n_seqs:
        b = min(512, n_seqs - done)
        chunks.append(pack_batch(b, ctx, g, skills, weights, n, "cpu", scramble))
        done += b
    return torch.cat(chunks, 0).to(device)


def pack_batch(B, ctx, g, skills, weights, n=8, device="cpu", scramble=()):
    """Training / stream batch: pack random-skill episodes (BOS-separated) into ctx-length
    sequences. `weights` = per-skill sampling probability (0 omits a skill -> the N1
    omission / guard-ablation lever). `scramble` = skills whose ANSWER token is resampled
    uniformly (the N2 marker-scramble negative: marker + input dist identical, rule
    unlearnable). Returns ids (B, ctx)."""
    wsum = sum(weights[s] for s in skills)
    probs = [weights[s] / wsum for s in skills]
    ids = torch.full((B, ctx), PAD, dtype=torch.long)
    for b in range(B):
        pos, out = 0, []
        while pos < ctx:
            si = int(torch.multinomial(torch.tensor(probs), 1, generator=g))
            sk = skills[si]
            e = gen_episode(sk, g, n)
            toks = list(e["toks"])
            if sk in scramble:
                toks[-1] = int(torch.randint(0, K, (1,), generator=g))  # break the rule
            seg = [BOS] + toks
            if pos + len(seg) > ctx:
                break
            out.extend(seg); pos += len(seg)
        if out:
            ids[b, : len(out)] = torch.tensor(out[:ctx])
    return ids.to(device)
