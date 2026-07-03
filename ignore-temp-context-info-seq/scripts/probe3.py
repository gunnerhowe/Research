"""Does decay help TR and hurt AR? alibi/semrf on TR-easy; alibi on locked AR."""
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

def trial(tag, task_name, task_kw, pos, d=256, steps=10000, lr=1e-3):
    task=build_task(task_name, **task_kw)
    set_seed(0)
    mcfg=ModelConfig(vocab_size=task.vocab_size,d_model=d,n_layers=4,n_heads=8,d_ff=4*d,
                     max_seq_len=512,position=pos,semrf_num_anchors=16,semrf_res_dim=32)
    tcfg=TrainConfig(steps=steps,batch_size=64,lr=lr,warmup_steps=250,
                     eval_every=steps//2,eval_batches=30,device=DEVICE,log_every=10**9)
    model=TransformerLM(mcfg); rng=np.random.default_rng(0)
    tb=lambda s:(lambda b:(b["input_ids"],b["targets"],b["loss_mask"]))(task.sample_batch(64,DEVICE,rng))
    ev=lambda s:evaluate_synthetic(model,task,30,64,DEVICE,np.random.default_rng(999))
    t0=time.time(); h=run_training(model,tcfg,tb,ev,log_prefix="")
    accs=" ".join(f"{x['token_acc']:.3f}" for x in h)
    print(f">> {tag:22s} pos={pos:6s} accs=[{accs}] ({time.time()-t0:.0f}s)", flush=True)

TRE = dict(n_vars=4,n_vals=16,seq_len=96,p_query=0.5)
ARL = dict(key_vocab=16,value_vocab=16,n_pairs=8,n_queries=8,gap=(0,32))
trial("TR_easy_alibi", "temporal_recency", TRE, "alibi")
trial("TR_easy_semrf", "temporal_recency", TRE, "semrf")
trial("AR_locked_alibi","assoc_recall",    ARL, "alibi", steps=15000)
