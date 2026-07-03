"""Map the (vocab, n_pairs) solvability frontier for AR + a long-train TR probe.
Lock the paper's task configs to the hardest solvable points."""
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

def trial(tag, task_name, task_kw, d=256, steps=6000, lr=1e-3):
    task=build_task(task_name, **task_kw)
    set_seed(0)
    mcfg=ModelConfig(vocab_size=task.vocab_size,d_model=d,n_layers=4,n_heads=8,d_ff=4*d,
                     max_seq_len=512,position="rope")
    tcfg=TrainConfig(steps=steps,batch_size=64,lr=lr,warmup_steps=250,
                     eval_every=steps,eval_batches=30,device=DEVICE,log_every=10**9)
    model=TransformerLM(mcfg); rng=np.random.default_rng(0)
    tb=lambda s:(lambda b:(b["input_ids"],b["targets"],b["loss_mask"]))(task.sample_batch(64,DEVICE,rng))
    ev=lambda s:evaluate_synthetic(model,task,30,64,DEVICE,np.random.default_rng(999))
    t0=time.time(); h=run_training(model,tcfg,tb,ev,log_prefix="")
    print(f">> {tag:22s} acc={h[-1]['token_acc']:.3f} seq={h[-1]['seq_acc']:.3f} ({time.time()-t0:.0f}s)", flush=True)

trial("P1_v16_D4",  "assoc_recall", dict(key_vocab=16,value_vocab=16,n_pairs=4,n_queries=8,gap=(0,8)))
trial("P2_v16_D8",  "assoc_recall", dict(key_vocab=16,value_vocab=16,n_pairs=8,n_queries=8,gap=(0,8)))
trial("P3_v32_D2",  "assoc_recall", dict(key_vocab=32,value_vocab=32,n_pairs=2,n_queries=8,gap=(0,8)))
trial("P4_v8_D6",   "assoc_recall", dict(key_vocab=8,value_vocab=8,n_pairs=6,n_queries=8,gap=(0,8)))
trial("P5_TR_15k",  "temporal_recency", dict(n_vars=8,n_vals=16,seq_len=128,p_query=0.4), steps=15000)
trial("P6_v16D8_15k","assoc_recall", dict(key_vocab=16,value_vocab=16,n_pairs=8,n_queries=8,gap=(0,8)), steps=15000)
