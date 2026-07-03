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

def trial(tag, task_kw, layers, heads, d, lr, steps):
    task=build_task("assoc_recall", **task_kw)
    set_seed(0)
    mcfg=ModelConfig(vocab_size=task.vocab_size,d_model=d,n_layers=layers,n_heads=heads,d_ff=4*d,
                     max_seq_len=256,position="rope")
    tcfg=TrainConfig(steps=steps,batch_size=64,lr=lr,warmup_steps=200,eval_every=steps,
                     eval_batches=20,device=DEVICE,log_every=steps//3)
    model=TransformerLM(mcfg); rng=np.random.default_rng(0)
    tb=lambda s:(lambda b:(b["input_ids"],b["targets"],b["loss_mask"]))(task.sample_batch(64,DEVICE,rng))
    ev=lambda s:evaluate_synthetic(model,task,20,64,DEVICE,np.random.default_rng(999))
    t0=time.time(); h=run_training(model,tcfg,tb,ev,log_prefix=f"[{tag}]")
    print(f">> {tag:26s} L={layers} h={heads} d={d} lr={lr} steps={steps} "
          f"acc={h[-1]['token_acc']:.3f} seq={h[-1]['seq_acc']:.3f} ({time.time()-t0:.0f}s)", flush=True)

# recipe search on associative recall
trial("tinyAR_L2_lr1e3", dict(key_vocab=8,value_vocab=8,n_pairs=4,n_queries=2), 2,2,128,1e-3,6000)
trial("tinyAR_L4_lr1e3", dict(key_vocab=8,value_vocab=8,n_pairs=4,n_queries=2), 4,4,128,1e-3,6000)
trial("medAR_L4_lr1e3",  dict(key_vocab=32,value_vocab=32,n_pairs=8,n_queries=4), 4,4,128,1e-3,8000)
trial("medAR_L2_lr1e3",  dict(key_vocab=32,value_vocab=32,n_pairs=8,n_queries=4), 2,4,128,1e-3,8000)
