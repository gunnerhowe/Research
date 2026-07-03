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

def trial(tag, task_kw, pos="rope", steps=6000, d=128, L=4, H=4, lr=1e-3, amp=True):
    task=build_task("assoc_recall", **task_kw)
    set_seed(0)
    mcfg=ModelConfig(vocab_size=task.vocab_size,d_model=d,n_layers=L,n_heads=H,d_ff=4*d,
                     max_seq_len=256,position=pos)
    tcfg=TrainConfig(steps=steps,batch_size=64,lr=lr,warmup_steps=200,eval_every=steps,
                     eval_batches=30,device=DEVICE,log_every=10**9,amp=amp)
    model=TransformerLM(mcfg); rng=np.random.default_rng(0)
    tb=lambda s:(lambda b:(b["input_ids"],b["targets"],b["loss_mask"]))(task.sample_batch(64,DEVICE,rng))
    ev=lambda s:evaluate_synthetic(model,task,30,64,DEVICE,np.random.default_rng(999),amp=amp)
    t0=time.time(); h=run_training(model,tcfg,tb,ev,log_prefix="")
    print(f">> {tag:20s} acc={h[-1]['token_acc']:.3f} seq={h[-1]['seq_acc']:.3f} ({time.time()-t0:.0f}s)", flush=True)

D2 = dict(key_vocab=8,value_vocab=8,n_pairs=2,n_queries=1)
trial("D2_base_bf16",   D2)
trial("D2_fp32",        D2, amp=False)
trial("D2_d256h8",      D2, d=256, H=8)
trial("D2_long15k",     D2, steps=15000)
trial("D2_manyQ",       dict(key_vocab=8,value_vocab=8,n_pairs=2,n_queries=8))
