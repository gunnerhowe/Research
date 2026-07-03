"""Confirm the synthetic tasks are actually solvable and discriminative with a
modest budget. Trains a few variants ~1500 steps on each task and prints final
accuracy + throughput, so we can size the full sweep sensibly.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from semrf.config import ModelConfig, TrainConfig
from semrf.model import TransformerLM
from semrf.data import build_task
from semrf.train import run_training
from semrf.eval import evaluate_synthetic
from semrf.utils import set_seed, count_params, human

DEVICE = "cuda"

TASKS = {
    "assoc_recall": dict(key_vocab=64, value_vocab=64, n_pairs=16, n_queries=4),
    "temporal_recency": dict(n_vars=20, n_vals=48, seq_len=192, p_query=0.25),
    "selective_copy": dict(n_data=16, context_len=128, data_vocab=16),
}


def run(task_name, task_kwargs, variants, steps=1500):
    task = build_task(task_name, **task_kwargs)
    print(f"\n=== {task_name}  vocab={task.vocab_size} seq_len={task.seq_len} ===")
    for pos in variants:
        set_seed(0)
        mcfg = ModelConfig(vocab_size=task.vocab_size, d_model=128, n_layers=4, n_heads=4,
                           d_ff=512, max_seq_len=512, position=pos,
                           semrf_num_anchors=16, semrf_res_dim=16)
        tcfg = TrainConfig(steps=steps, batch_size=64, lr=5e-4, warmup_steps=100,
                           eval_every=steps, eval_batches=20, device=DEVICE, log_every=steps)
        model = TransformerLM(mcfg)
        rng = np.random.default_rng(0)

        def tb(step):
            b = task.sample_batch(tcfg.batch_size, DEVICE, rng)
            return b["input_ids"], b["targets"], b["loss_mask"]

        def ev(step):
            return evaluate_synthetic(model, task, 20, 64, DEVICE, np.random.default_rng(999))

        t0 = time.time()
        hist = run_training(model, tcfg, tb, ev, log_prefix=f"[{task_name}:{pos}]")
        dt = time.time() - t0
        h = hist[-1]
        print(f"  {pos:11s} params={human(count_params(model)):>7s}  "
              f"token_acc={h['token_acc']:.3f} seq_acc={h['seq_acc']:.3f}  "
              f"{dt:.0f}s  {steps*tcfg.batch_size/dt:.0f} seq/s")


if __name__ == "__main__":
    variants = ["nope", "rope", "alibi", "semrf"]
    for name, kw in TASKS.items():
        run(name, kw, variants)
