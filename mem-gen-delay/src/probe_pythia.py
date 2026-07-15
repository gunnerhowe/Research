"""P6 R0: probe public Pythia checkpoints for induction-head formation + precursors.

Per checkpoint (revision=stepN), computes:
  copy_adv     behavioral induction bump: CE(1st half) - CE(2nd half) on repeated random
               sequences (positive once in-context copying works)
  prefix_by_layer  per-layer MAX head prefix-matching score (attn from 2nd-occurrence
               query to previous-occurrence+1 — the induction pattern)
  prevtok_by_layer per-layer MAX head previous-token score (the mechanistic precursor)
  text_loss    CE on a fixed, local, deterministic text batch (course/*.md)
Idempotent: skips steps already present in the output jsonl. HF cache pinned to E:.
"""
import argparse
import glob
import json
import os

os.environ.setdefault("HF_HOME", "E:/GitHub/Research/hf_cache")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# valid Pythia revisions: step0..step512 in powers of 2, then every 1000 to step143000
STEPS_DEFAULT = [0, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000,
                 6000, 8000, 12000, 16000, 32000, 64000, 110000, 143000]
TOKENS_PER_STEP = 1024 * 2048  # Pythia batch size in tokens


def fixed_text_batch(tok, n_seq=32, seq_len=512):
    files = sorted(glob.glob(os.path.join(ROOT, "course", "*.md")))
    text = "\n\n".join(open(f, encoding="utf-8").read() for f in files)
    ids = tok(text, return_tensors="pt").input_ids[0]
    need = n_seq * seq_len
    ids = ids[:need].reshape(n_seq, seq_len)
    return ids


def repeated_batch(vocab_size, B=64, L=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    half = torch.randint(1000, min(vocab_size, 45000), (B, L), generator=g)
    return torch.cat([half, half], dim=1)  # (B, 2L)


def ce_per_pos(logits, ids):
    lp = F.log_softmax(logits[:, :-1].float(), -1)
    return -lp.gather(2, ids[:, 1:, None]).squeeze(2)  # (B, T-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m")
    ap.add_argument("--steps", type=int, nargs="+", default=STEPS_DEFAULT)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["step"] for l in open(args.out) if l.strip()}
    tok = AutoTokenizer.from_pretrained(args.model)
    text_ids = None
    rep = None
    L = 64
    for step in args.steps:
        if step in done:
            print(f"skip step{step}", flush=True)
            continue
        print(f">> {args.model} step{step}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=f"step{step}", torch_dtype=torch.float32,
            attn_implementation="eager").to(device).eval()
        if text_ids is None:
            text_ids = fixed_text_batch(tok).to(device)
            rep = repeated_batch(model.config.vocab_size).to(device)
        with torch.no_grad():
            # behavioral + attention probes on repeated sequences
            out = model(rep, output_attentions=True)
            ce = ce_per_pos(out.logits, rep)                       # (B, 2L-1)
            first = ce[:, : L - 1].mean().item()
            second = ce[:, L - 1:].mean().item()
            n_layers = len(out.attentions)
            prefix_by_layer, prevtok_by_layer = [], []
            for att in out.attentions:                             # (B, H, T, T)
                a = att.float()
                q = torch.arange(L, 2 * L - 1, device=device)      # queries in 2nd half
                tgt = q - L + 1                                    # prev occurrence + 1
                prefix = a[:, :, q, :].gather(
                    3, tgt.view(1, 1, -1, 1).expand(a.shape[0], a.shape[1], -1, 1)
                ).squeeze(3).mean((0, 2))                          # (H,)
                prefix_by_layer.append(round(float(prefix.max()), 5))
                qq = torch.arange(1, 2 * L, device=device)
                prev = a[:, :, qq, :].gather(
                    3, (qq - 1).view(1, 1, -1, 1).expand(a.shape[0], a.shape[1], -1, 1)
                ).squeeze(3).mean((0, 2))
                prevtok_by_layer.append(round(float(prev.max()), 5))
            # fixed-text loss (chunked)
            losses = []
            for i in range(0, text_ids.shape[0], 8):
                lo = model(text_ids[i:i + 8]).logits
                losses.append(ce_per_pos(lo, text_ids[i:i + 8]).mean().item())
        rec = dict(model=args.model, step=step, tokens=step * TOKENS_PER_STEP,
                   copy_adv=round(first - second, 5),
                   ce_first=round(first, 5), ce_second=round(second, 5),
                   prefix_max=max(prefix_by_layer),
                   prefix_by_layer=prefix_by_layer,
                   prevtok_by_layer=prevtok_by_layer,
                   text_loss=round(float(np.mean(losses)), 5))
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in
                          ("step", "copy_adv", "prefix_max", "text_loss")}), flush=True)
        del model, out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
