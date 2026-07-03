"""Driver for the synthetic-task experiments.

For every (task, variant, seed) it trains a model, evaluates final accuracy,
collects accuracy-vs-distance, and runs the length-extrapolation protocol
(train short, evaluate at increasing lengths).  Results are written as one JSON
per run under results/synthetic/... and the driver is resumable (existing runs
are skipped unless --overwrite).

Usage:
    python -m experiments.run_synthetic --steps 4000 --seeds 0 1 2
    python -m experiments.run_synthetic --tasks assoc_recall --variants semrf rope
"""
import sys, os, argparse, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from semrf.config import ModelConfig, TrainConfig
from semrf.model import TransformerLM
from semrf.data import build_task
from semrf.train import run_training
from semrf.eval import evaluate_synthetic, collect_distance_correct, bucketize_accuracy
from semrf.utils import set_seed, count_params, save_json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "synthetic")

# --- task definitions: train setting + a ladder of eval lengths ------------- #
# Extrapolation grows the *span* while keeping the memory load fixed
# (assoc_recall: same #pairs, larger gap), so positional robustness is measured
# separately from recall capacity (cf. Zoology / MQAR capacity effects).
TASK_SPECS = {
    "assoc_recall": {
        # train with randomized gap 0-8 so BLANK filler is in-distribution;
        # eval grows the gap (span) with the memory load (8 pairs) fixed.
        # vocab 16/16 + gap<=8 + 15k steps: the regime where the induction
        # circuit reliably forms at d=256 (scripts/ar_frontier.py, P6). With
        # wider gap jitter (0-32) no baseline solves the training distribution
        # within budget (kept as results/synthetic/assoc_recall_hard).
        "train": dict(key_vocab=16, value_vocab=16, n_pairs=8, n_queries=8, gap=(0, 8)),
        "steps": 15000,
        "eval_ladder": [
            ("L1", dict(key_vocab=16, value_vocab=16, n_pairs=8, n_queries=8, gap=8)),
            ("L2", dict(key_vocab=16, value_vocab=16, n_pairs=8, n_queries=8, gap=64)),
            ("L3", dict(key_vocab=16, value_vocab=16, n_pairs=8, n_queries=8, gap=160)),
            ("L4", dict(key_vocab=16, value_vocab=16, n_pairs=8, n_queries=8, gap=352)),
        ],
    },
    "temporal_recency": {
        # memory load fixed (4 vars); longer streams add more reassignments and
        # longer retrieval spans -> tests time-sensitivity over growing lengths.
        # probe-validated: alibi/semrf solve this config at 10k steps; rope does not.
        "train": dict(n_vars=4, n_vals=16, seq_len=96, p_query=0.5),
        "steps": 10000,
        "eval_ladder": [
            ("L1", dict(n_vars=4, n_vals=16, seq_len=96, p_query=0.5)),
            ("L2", dict(n_vars=4, n_vals=16, seq_len=192, p_query=0.5)),
            ("L3", dict(n_vars=4, n_vals=16, seq_len=288, p_query=0.5)),
            ("L4", dict(n_vars=4, n_vals=16, seq_len=384, p_query=0.5)),
        ],
    },
    "selective_copy": {
        "train": dict(n_data=16, context_len=128, data_vocab=16),
        "steps": 5000,
        "eval_ladder": [
            ("L1", dict(n_data=16, context_len=128, data_vocab=16)),
            ("L2", dict(n_data=16, context_len=256, data_vocab=16)),
            ("L3", dict(n_data=16, context_len=384, data_vocab=16)),
            ("L4", dict(n_data=16, context_len=512, data_vocab=16)),
        ],
    },
}

# --- variants: 7 baselines + SemRF ablations -------------------------------- #
VARIANTS = {
    "nope": ("nope", {}),
    "sinusoidal": ("sinusoidal", {}),
    "learned": ("learned", {}),
    "rope": ("rope", {}),
    "alibi": ("alibi", {}),
    "t5": ("t5", {}),
    "cable": ("cable", {}),
    "semrf": ("semrf", {}),
    # ablations
    "semrf_no_time": ("semrf", dict(semrf_use_time=False)),
    "semrf_no_sem": ("semrf", dict(semrf_use_sem=False)),
    "semrf_no_res": ("semrf", dict(semrf_use_res=False)),
    "semrf_hard": ("semrf", dict(semrf_hard=True)),
    "semrf_K8": ("semrf", dict(semrf_num_anchors=8)),
    "semrf_K64": ("semrf", dict(semrf_num_anchors=64)),
}
BASELINES = ["nope", "sinusoidal", "learned", "rope", "alibi", "t5", "cable", "semrf"]


def build_model_cfg(vocab_size, position, extra, max_seq_len):
    # d=256/8 heads: the capacity regime where the induction/matching circuit
    # reliably forms (see scripts/ar_capacity.py); identical for all variants.
    base = dict(vocab_size=vocab_size, d_model=256, n_layers=4, n_heads=8, d_ff=1024,
                max_seq_len=max_seq_len, position=position,
                semrf_num_anchors=16, semrf_res_dim=32)
    base.update(extra)
    return ModelConfig(**base)


def run_one(task_name, variant, seed, steps, batch_size, overwrite=False):
    out_path = os.path.join(RESULTS, task_name, f"{variant}__seed{seed}.json")
    if os.path.exists(out_path) and not overwrite:
        print(f"skip {out_path}")
        return

    spec = TASK_SPECS[task_name]
    steps = spec.get("steps", steps)          # per-task budget overrides CLI
    position, extra = VARIANTS[variant]
    train_task = build_task(task_name, **spec["train"])
    vocab = train_task.vocab_size
    max_eval_len = max(build_task(task_name, **kw).seq_len for _, kw in spec["eval_ladder"]) + 2

    set_seed(seed)
    mcfg = build_model_cfg(vocab, position, extra, max_eval_len)
    tcfg = TrainConfig(steps=steps, batch_size=batch_size, lr=1e-3, warmup_steps=max(100, steps // 20),
                       eval_every=max(500, steps // 4), eval_batches=20, device=DEVICE,
                       log_every=max(500, steps // 4), seed=seed)
    model = TransformerLM(mcfg)
    n_params = count_params(model)
    rng = np.random.default_rng(seed)

    def tb(step):
        b = train_task.sample_batch(batch_size, DEVICE, rng)
        return b["input_ids"], b["targets"], b["loss_mask"]

    def ev(step):
        return evaluate_synthetic(model, train_task, 20, batch_size, DEVICE, np.random.default_rng(10_000 + seed))

    t0 = time.time()
    history = run_training(model, tcfg, tb, ev, log_prefix=f"[{task_name}:{variant}:s{seed}]")
    train_time = time.time() - t0

    # length-extrapolation ladder
    ladder = []
    for label, kw in spec["eval_ladder"]:
        etask = build_task(task_name, **kw)
        assert etask.vocab_size == vocab
        m = evaluate_synthetic(model, etask, 30, min(batch_size, 32), DEVICE,
                               np.random.default_rng(20_000 + seed))
        ladder.append({"label": label, "seq_len": etask.seq_len,
                        "token_acc": m["token_acc"], "seq_acc": m["seq_acc"]})

    # accuracy vs distance (train-length distribution)
    dist_buckets = []
    if task_name in ("assoc_recall", "temporal_recency"):
        d, c = collect_distance_correct(model, train_task, 40, batch_size, DEVICE,
                                        np.random.default_rng(30_000 + seed))
        dist_buckets = [{"distance": dd, "acc": aa, "n": nn} for dd, aa, nn in bucketize_accuracy(d, c, 12)]

    result = {
        "task": task_name, "variant": variant, "position": position, "seed": seed,
        "n_params": n_params, "train_time_s": train_time,
        "model_cfg": mcfg.to_dict(), "train_cfg": tcfg.to_dict(),
        "final": history[-1], "history": history,
        "extrapolation": ladder, "distance_buckets": dist_buckets,
    }
    save_json(result, out_path)
    if variant == "semrf" and seed == 0:
        ckpt = os.path.join(RESULTS, task_name, "semrf__seed0.ckpt.pt")
        torch.save({"state_dict": model.state_dict(), "model_cfg": mcfg.to_dict(),
                    "vocab_size": vocab, "task": task_name}, ckpt)
    print(f"saved {out_path}  final_acc={history[-1]['token_acc']:.3f}  ({train_time:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASK_SPECS))
    ap.add_argument("--variants", nargs="+", default=None, help="default: baselines only")
    ap.add_argument("--ablations", action="store_true", help="include SemRF ablations")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    variants = args.variants
    if variants is None:
        variants = list(VARIANTS) if args.ablations else BASELINES

    combos = [(t, v, s) for t in args.tasks for v in variants for s in args.seeds
              # ablation variants run on the first two seeds only (compute budget)
              if not (v not in BASELINES and s not in args.seeds[:2])]
    print(f"{len(combos)} runs on {DEVICE}: tasks={args.tasks} variants={variants} seeds={args.seeds} steps={args.steps}")
    for i, (t, v, s) in enumerate(combos):
        print(f"\n--- run {i+1}/{len(combos)}: {t} / {v} / seed {s} ---")
        run_one(t, v, s, args.steps, args.batch_size, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
