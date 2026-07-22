"""P8 Mythos zoo trainer/driver (prereg plan_p8.md, commit 204a124).

Modes:
  specimen : train TinyLM (3L/8H, ctx128) on the full skill mixture -> checkpoint + metrics
  capture  : read the frozen specimen's fingerprint library on fixed probe banks -> json
  guard    : continue-train with guarded skills omitted (data-ablation) -> guarded ckpt
  watch    : continue-train a guarded ckpt on a stream, logging per-skill proximity INLINE
             every eval on the FIXED probe banks (never the streaming batch)

Levers (compose into named streams via scripts/run_zoo_pilot.py):
  --omit s,s      zero those skills' episode weight (guard data-ablation / N1 omission)
  --scramble s,s  resample those skills' answers (N2 marker-scramble negative)
  --reteach s     reintroduce one guarded skill (watch positive)
  --burn_home     D-RELOC: pin the reteught skill's specimen home head to a useless
                  pattern (attach_scaffold sink) so the skill must relocate
Metrics per eval: per-skill behavioral accuracy + (watch) proximity scalars vs the frozen fp.
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from train_lm import TinyLM, attach_scaffold
import data_zoo as dz
import fingerprint_zoo as fz


def build_model(seed, n_layers, n_heads, d, ctx, device):
    torch.manual_seed(seed)
    return TinyLM(dz.VOCAB, d=d, h=n_heads, n_layers=n_layers, ctx=ctx).to(device)


def probe_banks(skills, n, device):
    return {s: dz.build_probe_bank(s, N=256, n=n, device=device) for s in skills}


def evaluate(model, banks, fp_lib, device):
    model.eval()
    rec = {}
    for s, pb in banks.items():
        if fp_lib and s in fp_lib:
            rec[s] = fz.proximity(model, pb, fp_lib[s], device)
        else:
            rec[s] = dict(acc=fz.behavioral_metric(model, pb, device))
    model.train()
    return rec


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    skills = args.skills.split(",")
    omit = set(args.omit.split(",")) if args.omit else set()
    scramble = tuple(args.scramble.split(",")) if args.scramble else ()
    boost = {}
    if args.weights:                                   # e.g. "M2:3,M3:3,M6:2" (disclosed
        for kv in args.weights.split(","):             # specimen mix rebalance so the hard
            k, v = kv.split(":"); boost[k] = float(v)  # skills aren't crowded out)
    weights = {s: (0.0 if s in omit else boost.get(s, 1.0)) for s in skills}
    if args.reteach:                                   # watch positive: turn a guarded skill on
        weights[args.reteach] = boost.get(args.reteach, 1.0)

    model = build_model(args.seed, args.n_layers, args.n_heads, args.d, args.ctx, device)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location=device))
    if args.burn_home and args.fp_lib and args.reteach:
        fp = json.load(open(args.fp_lib))[args.reteach]
        if fp["kind"] == "attn":                        # D-RELOC: pin the home head useless
            attach_scaffold(model, "sink", 8.0, layer=fp["home_layer"], head=fp["home_head"])
            model.to(device)
    if args.erase_subspace and args.fp_lib and args.reteach:  # D-IDIO: erase the captured
        fp = json.load(open(args.fp_lib))[args.reteach]        # depth subspace every forward
        B = torch.tensor(fp["basis"], device=device)           # (r, D), so the return must
        lyr = fp["layer"]                                       # re-form in an orthogonal dir

        def _erase(mod, inp, out):
            x, a = out
            return (x - (x @ B.T) @ B, a)
        model.blocks[lyr].register_forward_hook(_erase)

    fp_lib = json.load(open(args.fp_lib)) if (args.fp_lib and args.mode == "watch") else None
    banks = probe_banks(skills, args.n, device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": nodecay, "weight_decay": 0.0}], lr=args.lr)
    g = torch.Generator().manual_seed(args.seed + 1)
    os.makedirs(args.out_dir, exist_ok=True)
    mf = open(os.path.join(args.out_dir, "metrics.jsonl"), "w")
    pool, poolmask = dz.build_pool(args.pool, args.ctx, g, skills, weights, n=args.n,
                                   device=device, scramble=scramble)   # one pool per run
    bg = torch.Generator().manual_seed(args.seed + 2)

    import math
    floor = 0.1 * args.lr
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            rec = evaluate(model, banks, fp_lib, device)
            mf.write(json.dumps(dict(step=step, skills=rec)) + "\n"); mf.flush()
        if step == args.steps:
            break
        if step > 0 and step % args.pool_refresh == 0:   # fresh data (avoid memorizing a pool)
            pool, poolmask = dz.build_pool(args.pool, args.ctx, g, skills, weights, n=args.n,
                                           device=device, scramble=scramble)
        if step < args.warmup:                            # warmup then cosine decay to a floor
            lr = args.lr * (step + 1) / args.warmup
        else:
            prog = (step - args.warmup) / max(1, args.steps - args.warmup)
            lr = floor + 0.5 * (args.lr - floor) * (1 + math.cos(math.pi * prog))
        for gp in opt.param_groups:
            gp["lr"] = lr
        idx = torch.randint(0, pool.shape[0], (args.batch,), generator=bg)
        ids = pool[idx]
        logits, _ = model(ids)
        tgt = ids[:, 1:]
        m = poolmask[idx][:, :-1] & (tgt != dz.PAD)     # focus loss on the skill-answer tokens
        loss = F.cross_entropy(logits[:, :-1][m], tgt[m])
        opt.zero_grad(); loss.backward(); opt.step()

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
    json.dump(dict(mode=args.mode, skills=skills, omit=list(omit), scramble=list(scramble),
                   reteach=args.reteach, burn_home=args.burn_home, seed=args.seed,
                   steps=args.steps), open(os.path.join(args.out_dir, "summary.json"), "w"))
    mf.close()


def capture(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    skills = args.skills.split(",")
    model = build_model(args.seed, args.n_layers, args.n_heads, args.d, args.ctx, device)
    model.load_state_dict(torch.load(args.init_from, map_location=device))
    model.eval()
    lib = {}
    for s in skills:
        pb = dz.build_probe_bank(s, N=256, n=args.n, device=device)
        if s == "M6":
            lib[s] = fz.capture_subspace_fp(model, pb, layer=args.n_layers // 2, device=device)
        else:
            lib[s] = fz.capture_attention_fp(model, pb, device=device)
        lib[s]["tau_acc"] = None                         # pinned by the scorer prereg step
    json.dump(lib, open(args.out_lib, "w"))
    print("captured fingerprint library ->", args.out_lib)
    for s in skills:
        f = lib[s]
        print(f"  {s}: {f['kind']}", (f"home=({f['home_layer']},{f['home_head']}) m0={f['m0']:.3f}"
              if f["kind"] == "attn" else f"r2={f['r2']:.3f}"))

    # separability confusion (G-P2, Fork 1): mass at skill f's HOME head read on skill p's
    # bank (p's own alignment target). Diagonal dominance => offset-specific home heads.
    attn = [s for s in skills if lib[s]["kind"] == "attn"]
    cs = {p: {} for p in skills}
    for p in attn:
        pb = dz.build_probe_bank(p, N=256, n=args.n, device=device)
        mass = fz._alignment_mass(model, pb, device).flatten()   # (L*H,)
        for f in attn:
            fl = lib[f]
            cs[p][f] = float(mass[fl["home_layer"] * fl["n_heads"] + fl["home_head"]])
    if args.conf_out:
        json.dump(cs, open(args.conf_out, "w"))
        print("wrote confusion ->", args.conf_out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["specimen", "guard", "watch", "capture"])
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--out_lib", default="")
    ap.add_argument("--conf_out", default="")
    ap.add_argument("--init_from", default="")
    ap.add_argument("--fp_lib", default="")
    ap.add_argument("--skills", default=",".join(dz.PILOT_SKILLS))
    ap.add_argument("--omit", default="")
    ap.add_argument("--scramble", default="")
    ap.add_argument("--reteach", default="")
    ap.add_argument("--weights", default="")
    ap.add_argument("--burn_home", action="store_true")       # D-RELOC
    ap.add_argument("--erase_subspace", action="store_true")  # D-IDIO
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pool", type=int, default=16384)
    ap.add_argument("--pool_refresh", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.mode == "capture":
        capture(args)
    else:
        train(args)
