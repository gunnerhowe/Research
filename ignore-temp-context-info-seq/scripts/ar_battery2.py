"""Second battery: which lever makes large-ish AR + temporal recency solvable?
  A. moderate vocab (32/32) + full-coverage queries
  B. weight decay 0 at vocab 64
  C. lr 2e-3 at vocab 64
  D. small-task temporal recency
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
DEVICE="cuda"

def trial(tag, task_name, task_kw, d=256, L=4, steps=8000, lr=1e-3, wd=0.1):
    task=build_task(task_name, **task_kw)
    set_seed(0)
    mcfg=ModelConfig(vocab_size=task.vocab_size,d_model=d,n_layers=L,n_heads=d//32,d_ff=4*d,
                     max_seq_len=512,position="rope")
    tcfg=TrainConfig(steps=steps,batch_size=64,lr=lr,weight_decay=wd,warmup_steps=250,
                     eval_every=steps,eval_batches=30,device=DEVICE,log_every=10**9)
    model=TransformerLM(mcfg); rng=np.random.default_rng(0)
    tb=lambda s:(lambda b:(b["input_ids"],b["targets"],b["loss_mask"]))(task.sample_batch(64,DEVICE,rng))
    ev=lambda s:evaluate_synthetic(model,task,30,64,DEVICE,np.random.default_rng(999))
    t0=time.time(); h=run_training(model,tcfg,tb,ev,log_prefix="")
    print(f">> {tag:24s} acc={h[-1]['token_acc']:.3f} seq={h[-1]['seq_acc']:.3f} ({time.time()-t0:.0f}s)", flush=True)

AR64 = dict(key_vocab=64,value_vocab=64,n_pairs=8,n_queries=8,gap=(0,8))
trial("A_vocab32",   "assoc_recall", dict(key_vocab=32,value_vocab=32,n_pairs=8,n_queries=8,gap=(0,8)))
trial("B_wd0",       "assoc_recall", AR64, wd=0.0)
trial("C_lr2e3",     "assoc_recall", AR64, lr=2e-3)
trial("D_TR_small",  "temporal_recency", dict(n_vars=8,n_vals=16,seq_len=128,p_query=0.4))
