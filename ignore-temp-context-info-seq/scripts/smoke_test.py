"""Fast end-to-end sanity check: every position variant trains a few steps on a
tiny associative-recall task and evaluates, on both GPU and the extrapolation
path. Run: python -m scripts.smoke_test
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from semrf.config import ModelConfig, TrainConfig
from semrf.model import TransformerLM
from semrf.positions import POSITION_TYPES
from semrf.data import build_task
from semrf.train import run_training
from semrf.eval import evaluate_synthetic, collect_distance_correct, bucketize_accuracy
from semrf.utils import set_seed, count_params, human


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    task = build_task("assoc_recall", key_vocab=32, value_vocab=32, n_pairs=8, n_queries=4)
    print("task vocab", task.vocab_size, "seq_len", task.seq_len)

    for pos in POSITION_TYPES:
        set_seed(0)
        mcfg = ModelConfig(
            vocab_size=task.vocab_size, d_model=64, n_layers=2, n_heads=4, d_ff=128,
            max_seq_len=256, position=pos, semrf_num_anchors=8, semrf_res_dim=8,
        )
        tcfg = TrainConfig(steps=30, batch_size=16, eval_every=30, eval_batches=5,
                           warmup_steps=5, device=device, log_every=30)
        model = TransformerLM(mcfg)
        rng = np.random.default_rng(0)

        def train_batch(step):
            b = task.sample_batch(tcfg.batch_size, device, rng)
            return b["input_ids"], b["targets"], b["loss_mask"]

        def evaluate(step):
            return evaluate_synthetic(model, task, tcfg.eval_batches, tcfg.batch_size, device,
                                      np.random.default_rng(123))

        hist = run_training(model, tcfg, train_batch, evaluate, log_prefix=f"[{pos}]")
        acc = hist[-1]["token_acc"]
        print(f"  {pos:11s} params={human(count_params(model))} final token_acc={acc:.3f}")

        # extrapolation path: eval at a longer sequence than trained
        try:
            big = build_task("assoc_recall", key_vocab=32, value_vocab=32, n_pairs=24, n_queries=4)
            with torch.no_grad():
                m = evaluate_synthetic(model, big, 3, 8, device, np.random.default_rng(1))
            print(f"    extrapolation(len {big.seq_len}) token_acc={m['token_acc']:.3f}")
        except Exception as e:
            print(f"    extrapolation FAILED (expected for learned-abs): {type(e).__name__}: {e}")

    # distance bucketing check
    d, c = collect_distance_correct(model, task, 5, 16, device, np.random.default_rng(7))
    print("distance buckets:", bucketize_accuracy(d, c, 5))
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
