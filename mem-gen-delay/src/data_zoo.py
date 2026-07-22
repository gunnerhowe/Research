"""P8 Mythos zoo (dense-supervision redesign): marker-triggered capability episodes with
ANSWER SPANS (multiple graded next-token targets per episode) so the plain LM loss
densely supervises each skill — the pilot showed a single end-of-episode answer token is
too weak a gradient (Dyck collapsed to a constant; k-back never formed).

Library (3 distinct mechanisms + 1 sibling):
  M0 INDUCTION (+1)     content-match copy; attention to matched token's successor
  M1 SKIP-INDUCTION(+2) sibling of M0 (offset +2)
  M4 KV-RECALL          indexed retrieval: match query to a KEY block, copy the VALUE at
                        the same index in a separate value block (distinct from induction)
  M6 DEPTH              running bracket-depth counting, output the depth sequence (dense);
                        fingerprint = a linear depth subspace of the interior residual
Guard {M0, M6}; retain {M1, M4}. Every episode reports its graded span positions, target
tokens, and (for attention skills) the alignment-target position the skill's head reads.
"""
from __future__ import annotations

import torch

# ---- vocab layout (ids), VOCAB=96 -------------------------------------------------------
K = 64                      # content ring c0..c63 = ids 0..63
OP0 = 64                    # opcodes M0..M9 = 64..73
BOS = 74
SEP = 75
EQ = 76                     # M4 key/value block separator '='
OPEN = 77                   # bracket '('
CLOSE = 78                  # bracket ')'
PAD = 79
DEPTH0 = 80                 # depth tokens 0..8 -> ids 80..88
VOCAB = 96

# Pilot library = 2 genuinely-distinct, NON-interfering mechanisms (+ induction's sibling):
# content-match copy (attention-alignment fp) + depth-counting (subspace fp). Multiple
# attention-MATCHING skills interfere at this scale (M4 retrieval degraded induction even
# at 2x capacity — a disclosed pilot finding); M4 deferred to future breadth work.
PILOT_SKILLS = ["M0", "M1", "M6"]
GUARDED_PILOT = ["M0", "M6"]                  # retained pilot sibling: M1
OPCODE = {f"M{i}": OP0 + i for i in range(10)}
SPAN_LEN = {"M0": 1, "M1": 1, "M4": 3, "M6": 8}     # graded positions per skill
ATTN_SKILLS = ["M0", "M1", "M4"]                     # subspace skill: M6


def _distinct(k, n, g):
    return torch.randperm(k, generator=g)[:n].tolist()


def gen_episode(skill, g, n=8):
    """Return dict(toks, span, depth). span = list of (pos, tok, align_tgt) graded
    positions (episode-relative). depth = per-token running depth for M6 (else None)."""
    op = OPCODE[skill]
    if skill in ("M0", "M1"):                        # content-match copy, offset +1/+2
        off = 1 if skill == "M0" else 2               # SINGLE query: multi-query duplicates
        ctx = _distinct(K, n, g)                       # a context token as an answer, making
        j = int(torch.randint(0, n - off, (1,), generator=g))  # later matches ambiguous and
        toks = [op] + ctx + [SEP, ctx[j], ctx[j + off]]        # breaking induction (bug found
        qpos = len(toks) - 2                                   # in the multi-query variant)
        return dict(toks=toks, span=[(qpos, ctx[j + off], 1 + j + off)], depth=None)
    if skill == "M4":                                 # indexed key->value retrieval
        keys = _distinct(K, 4, g)
        vals = torch.randint(0, K, (4,), generator=g).tolist()
        qi = torch.randint(0, 4, (3,), generator=g).tolist()          # 3 queries (dense)
        toks = [op] + keys + [EQ] + vals + [SEP]
        vbase = 1 + 4 + 1                             # values block starts here
        span = []
        for i in qi:
            qpos = len(toks)                          # position of this query token
            toks += [keys[i], vals[i]]                # query then its answer
            span.append((qpos, vals[i], vbase + i))   # head reads the value at matched index
        return dict(toks=toks, span=span, depth=None)
    if skill == "M6":                                 # running-depth counting (dense)
        draws = torch.randint(0, 2, (n,), generator=g).tolist()
        seq, dep, d = [], [], 0
        for t in range(n):
            b = OPEN if (d == 0 or draws[t]) else CLOSE          # always-valid prefix: d>=0
            d += 1 if b == OPEN else -1
            seq.append(b); dep.append(d)
        toks = [op] + seq + [SEP]
        sep = len(toks) - 1
        span = []
        for k in range(n):                            # predict d1..dn across the answer span
            toks.append(DEPTH0 + dep[k])
            span.append((sep + k, DEPTH0 + dep[k], -1))
        depth = [-100] + dep + [-100] * (len(toks) - 1 - n)         # depth at bracket positions
        return dict(toks=toks, span=span, depth=depth)
    raise ValueError(skill)


def build_probe_bank(skill, N=256, seed=20250801, n=8, device="cpu"):
    """Fixed held-out probe bank. Returns ids (N,L); span_pos (P,) fixed positions;
    span_tok (N,P); span_tgt (N,P) alignment targets; depth (N,L) or None."""
    g = torch.Generator().manual_seed(seed + OPCODE[skill])
    eps = [gen_episode(skill, g, n) for _ in range(N)]
    L = max(len(e["toks"]) for e in eps)
    P = SPAN_LEN[skill]
    ids = torch.full((N, L), PAD, dtype=torch.long)
    span_pos = torch.tensor([p for p, _, _ in eps[0]["span"]])         # identical across eps
    span_tok = torch.zeros(N, P, dtype=torch.long)
    span_tgt = torch.full((N, P), -1, dtype=torch.long)
    depth = torch.full((N, L), -100, dtype=torch.long)
    for i, e in enumerate(eps):
        t = e["toks"]
        ids[i, : len(t)] = torch.tensor(t)
        for k, (_, tok, tgt) in enumerate(e["span"]):
            span_tok[i, k] = tok; span_tgt[i, k] = tgt
        if e["depth"] is not None:
            depth[i, : len(e["depth"])] = torch.tensor(e["depth"][: len(t)])
    return dict(ids=ids.to(device), span_pos=span_pos.to(device),
                span_tok=span_tok.to(device), span_tgt=span_tgt.to(device),
                depth=depth.to(device) if skill == "M6" else None)


def pack_batch(B, ctx, g, skills, weights, n=8, device="cpu", scramble=()):
    """Pack random-skill episodes (BOS-separated) into ctx-length sequences. weights=0
    omits a skill (N1/guard); `scramble` resamples a skill's ANSWER tokens (N2). Returns
    (ids, ansmask): ansmask[b,t]=True iff position t's NEXT token is a graded skill answer,
    so training can focus the loss on the skill signal (the ~8 unpredictable context tokens
    per episode otherwise drown out the 1-per-episode answer gradient)."""
    wsum = sum(weights[s] for s in skills)
    probs = [weights[s] / wsum for s in skills]
    ids = torch.full((B, ctx), PAD, dtype=torch.long)
    ansmask = torch.zeros((B, ctx), dtype=torch.bool)
    for b in range(B):
        pos, out, spans = 0, [], []
        while pos < ctx:
            sk = skills[int(torch.multinomial(torch.tensor(probs), 1, generator=g))]
            e = gen_episode(sk, g, n)
            toks = list(e["toks"])
            if sk in scramble:
                for (p, _, _) in e["span"]:
                    toks[p + 1] = int(torch.randint(0, K, (1,), generator=g))  # break rule
            seg = [BOS] + toks
            if pos + len(seg) > ctx:
                break
            for (p, _, _) in e["span"]:
                spans.append(pos + 1 + p)             # abs position whose next-token = answer
            out.extend(seg); pos += len(seg)
        if out:
            row = out[:ctx]
            ids[b, : len(row)] = torch.tensor(row)
            for aq in spans:
                if aq < len(row):
                    ansmask[b, aq] = True
    return ids.to(device), ansmask.to(device)


def build_pool(n_seqs, ctx, g, skills, weights, n=8, device="cpu", scramble=()):
    ids_c, m_c, done = [], [], 0
    while done < n_seqs:
        b = min(512, n_seqs - done)
        i, m = pack_batch(b, ctx, g, skills, weights, n, "cpu", scramble)
        ids_c.append(i); m_c.append(m); done += b
    return torch.cat(ids_c, 0).to(device), torch.cat(m_c, 0).to(device)
