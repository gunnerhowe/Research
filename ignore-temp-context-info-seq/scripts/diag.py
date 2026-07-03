"""Focused gating diagnostics before the full sweep:
  - are assoc_recall / temporal_recency solvable with more steps?
  - does SemRF recover on selective_copy after the ALiBi-start init fix?
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from semrf.config import ModelConfig, TrainConfig
from semrf.model import TransformerLM
from semrf.data import build_task
from semrf.train import run_training
from semrf.eval import evaluate_synthetic
from semrf.utils import set_seed

DEVICE = "cuda"

def go(task_name, kw, pos, steps, lr=5e-4, d=128, L=4, H=4):
    task = build_task(task_name, **kw)
    set_seed(0)
    mcfg = ModelConfig(vocab_size=task.vocab_size, d_model=d, n_layers=L, n_heads=H, d_ff=4*d,
                       max_seq_len=1024, position=pos, semrf_num_anchors=16, semrf_res_dim=16)
    tcfg = TrainConfig(steps=steps, batch_size=64, lr=lr, warmup_steps=100,
                       eval_every=steps, eval_batches=20, device=DEVICE, log_every=max(1,steps//5))
    model = TransformerLM(mcfg)
    rng = np.random.default_rng(0)
    tb = lambda s: (lambda b: (b["input_ids"], b["targets"], b["loss_mask"]))(task.sample_batch(64, DEVICE, rng))
    ev = lambda s: evaluate_synthetic(model, task, 20, 64, DEVICE, np.random.default_rng(999))
    t0=time.time()
    h = run_training(model, tcfg, tb, ev, log_prefix=f"[{task_name}:{pos}]")
    print(f">> {task_name:16s} {pos:6s} steps={steps} token_acc={h[-1]['token_acc']:.3f} seq_acc={h[-1]['seq_acc']:.3f} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    # (1) selective_copy: does SemRF recover to ALiBi/RoPE level now?
    go("selective_copy", dict(n_data=16, context_len=128, data_vocab=16), "semrf", 2500)
    go("selective_copy", dict(n_data=16, context_len=128, data_vocab=16), "rope", 2500)
    # (2) assoc_recall: solvable with more steps?
    go("assoc_recall", dict(key_vocab=64, value_vocab=64, n_pairs=16, n_queries=4), "rope", 6000)
    go("assoc_recall", dict(key_vocab=64, value_vocab=64, n_pairs=16, n_queries=4), "semrf", 6000)
    # (3) temporal_recency: solvable?
    go("temporal_recency", dict(n_vars=20, n_vals=48, seq_len=192, p_query=0.25), "rope", 6000)
    go("temporal_recency", dict(n_vars=20, n_vals=48, seq_len=192, p_query=0.25), "semrf", 6000)
