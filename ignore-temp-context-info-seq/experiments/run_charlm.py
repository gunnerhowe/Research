"""Driver for the character-level language-modeling experiments (enwik8 / text8).

Trains a small decoder-only LM per (variant, seed), reports test bits-per-char,
and runs the length-extrapolation protocol (train at T=512, evaluate bpc at
increasing context lengths without retraining).  One JSON per run; resumable.

Usage:
    python -m experiments.run_charlm --steps 10000 --seeds 0 1
    python -m experiments.run_charlm --corpus enwik8 --variants rope alibi semrf
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from semrf.config import ModelConfig, TrainConfig
from semrf.model import TransformerLM
from semrf.data.charlm import CharLMData
from semrf.train import run_training
from semrf.eval import evaluate_bpc, extrapolation_bpc
from semrf.utils import set_seed, count_params, save_json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "charlm")

VARIANTS = {
    "nope": ("nope", {}),
    "sinusoidal": ("sinusoidal", {}),
    "learned": ("learned", {}),
    "rope": ("rope", {}),
    "alibi": ("alibi", {}),
    "t5": ("t5", {}),
    "cable": ("cable", {}),
    "semrf": ("semrf", {}),
}
BASELINES = list(VARIANTS)

TRAIN_LEN = 512
EVAL_LENS = [256, 512, 1024, 2048, 4096]
EVAL_MAX_TOKENS = 500_000          # cap eval for speed / reproducibility


def build_model_cfg(vocab_size, position, extra):
    base = dict(vocab_size=vocab_size, d_model=384, n_layers=6, n_heads=6, d_ff=1536,
                dropout=0.1, max_seq_len=max(EVAL_LENS), position=position, tie_embeddings=True,
                semrf_num_anchors=32, semrf_res_dim=32)
    base.update(extra)
    return ModelConfig(**base)


def eval_batch_for_len(L):
    return max(1, 8192 // L)


def run_one(corpus, data, variant, seed, steps, batch_size, overwrite=False):
    out_path = os.path.join(RESULTS, corpus, f"{variant}__seed{seed}.json")
    if os.path.exists(out_path) and not overwrite:
        print(f"skip {out_path}")
        return

    position, extra = VARIANTS[variant]
    set_seed(seed)
    mcfg = build_model_cfg(data.vocab_size, position, extra)
    tcfg = TrainConfig(steps=steps, batch_size=batch_size, lr=6e-4, warmup_steps=max(200, steps // 25),
                       weight_decay=0.1, eval_every=max(1000, steps // 6), eval_batches=0,
                       device=DEVICE, log_every=200, seed=seed, amp=True, amp_dtype="bfloat16")
    model = TransformerLM(mcfg)
    n_params = count_params(model)
    rng = np.random.default_rng(seed)

    def tb(step):
        x, y = data.get_batch("train", batch_size, TRAIN_LEN, DEVICE, rng)
        return x, y, None

    def ev(step):
        bpc = evaluate_bpc(model, data, "val", TRAIN_LEN, eval_batch_for_len(TRAIN_LEN),
                           DEVICE, max_tokens=EVAL_MAX_TOKENS)
        return {"val_bpc": round(bpc, 4)}

    t0 = time.time()
    history = run_training(model, tcfg, tb, ev, log_prefix=f"[{corpus}:{variant}:s{seed}]")
    train_time = time.time() - t0

    # final test bpc at train length + length-extrapolation curve
    test_bpc = evaluate_bpc(model, data, "test", TRAIN_LEN, eval_batch_for_len(TRAIN_LEN),
                            DEVICE, max_tokens=EVAL_MAX_TOKENS)
    extrap = {}
    for L in EVAL_LENS:
        try:
            extrap[str(L)] = round(evaluate_bpc(model, data, "test", L, eval_batch_for_len(L),
                                                DEVICE, max_tokens=EVAL_MAX_TOKENS), 4)
        except Exception as e:  # e.g. learned-abs beyond table (shouldn't happen w/ max_seq_len)
            extrap[str(L)] = None
            print(f"  extrap L={L} failed: {e}")

    result = {
        "corpus": corpus, "variant": variant, "position": position, "seed": seed,
        "n_params": n_params, "train_time_s": train_time,
        "model_cfg": mcfg.to_dict(), "train_cfg": tcfg.to_dict(),
        "test_bpc": round(test_bpc, 4), "extrapolation_bpc": extrap,
        "history": history,
    }
    save_json(result, out_path)
    if variant == "semrf" and seed == 0:
        ckpt = os.path.join(RESULTS, corpus, "semrf__seed0.ckpt.pt")
        torch.save({"state_dict": model.state_dict(), "model_cfg": mcfg.to_dict(),
                    "vocab_size": data.vocab_size}, ckpt)
        print(f"  saved checkpoint {ckpt}")
    print(f"saved {out_path}  test_bpc={test_bpc:.4f}  ({train_time/60:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="enwik8", choices=["enwik8", "text8"])
    ap.add_argument("--variants", nargs="+", default=BASELINES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    print(f"loading {args.corpus} ...")
    data = CharLMData(args.corpus, cache_dir=os.path.join(ROOT, "data_cache"))
    print(f"vocab={data.vocab_size} train={len(data.train):,} val={len(data.val):,} test={len(data.test):,}")

    combos = [(v, s) for v in args.variants for s in args.seeds]
    print(f"{len(combos)} runs: variants={args.variants} seeds={args.seeds} steps={args.steps}")
    for i, (v, s) in enumerate(combos):
        print(f"\n--- run {i+1}/{len(combos)}: {v} / seed {s} ---")
        run_one(args.corpus, data, v, s, args.steps, args.batch_size, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
