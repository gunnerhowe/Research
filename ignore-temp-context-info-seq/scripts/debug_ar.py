import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from semrf.config import ModelConfig, TrainConfig
from semrf.model import TransformerLM
from semrf.data import build_task
from semrf.train import run_training
from semrf.eval import evaluate_synthetic
from semrf.utils import set_seed

DEVICE="cuda"
# tiny AR: can the model solve it at all?
task = build_task("assoc_recall", key_vocab=8, value_vocab=8, n_pairs=4, n_queries=2)
print("vocab", task.vocab_size, "seq_len", task.seq_len)
set_seed(0)
mcfg=ModelConfig(vocab_size=task.vocab_size,d_model=128,n_layers=4,n_heads=4,d_ff=512,
                 max_seq_len=128,position="rope")
tcfg=TrainConfig(steps=3000,batch_size=64,lr=5e-4,warmup_steps=100,eval_every=3000,
                 eval_batches=20,device=DEVICE,log_every=1000)
model=TransformerLM(mcfg); rng=np.random.default_rng(0)
tb=lambda s:(lambda b:(b["input_ids"],b["targets"],b["loss_mask"]))(task.sample_batch(64,DEVICE,rng))
ev=lambda s:evaluate_synthetic(model,task,20,64,DEVICE,np.random.default_rng(999))
h=run_training(model,tcfg,tb,ev,log_prefix="[tinyAR]")
print("final", h[-1])

# inspect one batch
b=task.sample_batch(4,DEVICE,np.random.default_rng(7))
with torch.no_grad():
    logits,_=model(b["input_ids"])
pred=logits.argmax(-1)
inp=b["input_ids"].cpu().numpy(); tgt=b["targets"].cpu().numpy()
pr=pred.cpu().numpy(); mask=b["loss_mask"].cpu().numpy()
for i in range(2):
    print(f"\nseq {i}")
    print(" input :", inp[i].tolist())
    print(" target:", tgt[i].tolist())
    ans=np.where(mask[i]==1)[0]
    for p in ans:
        print(f"  pos {p}: query_key={inp[i][p]} -> target_val={tgt[i][p]} pred={pr[i][p]} {'OK' if pr[i][p]==tgt[i][p] else 'X'}")
