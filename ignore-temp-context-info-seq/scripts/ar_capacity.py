"""Find the (width, #pairs) regime where the baselines actually solve associative
recall and temporal recency, so the main sweep trains in a solvable regime."""
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

def trial(task_name, task_kw, d, L=4, steps=7000, lr=1e-3, pos="rope"):
    task=build_task(task_name, **task_kw)
    H=d//32
    set_seed(0)
    mcfg=ModelConfig(vocab_size=task.vocab_size,d_model=d,n_layers=L,n_heads=H,d_ff=4*d,
                     max_seq_len=512,position=pos)
    tcfg=TrainConfig(steps=steps,batch_size=64,lr=lr,warmup_steps=250,eval_every=steps,
                     eval_batches=30,device=DEVICE,log_every=10**9)
    model=TransformerLM(mcfg); rng=np.random.default_rng(0)
    tb=lambda s:(lambda b:(b["input_ids"],b["targets"],b["loss_mask"]))(task.sample_batch(64,DEVICE,rng))
    ev=lambda s:evaluate_synthetic(model,task,30,64,DEVICE,np.random.default_rng(999))
    t0=time.time(); h=run_training(model,tcfg,tb,ev,log_prefix="")
    tag=f"{task_name[:5]} {list(task_kw.values())} d={d} L={L}"
    print(f">> {tag:52s} pos={pos:5s} acc={h[-1]['token_acc']:.3f} seq={h[-1]['seq_acc']:.3f} ({time.time()-t0:.0f}s)", flush=True)

# associative recall: sweep width x pairs (n_queries=8 for dense supervision)
for D in [4, 8, 16]:
    for d in [256, 384]:
        trial("assoc_recall", dict(key_vocab=64,value_vocab=64,n_pairs=D,n_queries=8,gap=0), d)
# temporal recency at larger width
for d in [256, 384]:
    trial("temporal_recency", dict(n_vars=20,n_vals=48,seq_len=192,p_query=0.25), d, L=4)
