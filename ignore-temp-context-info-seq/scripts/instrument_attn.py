"""Train a tiny 2-pair associative-recall model, then visualize (in text) the
attention at the query position for each layer/head, to see whether an induction
head forms (query attends to the correct value position).
"""
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

task = build_task("assoc_recall", key_vocab=8, value_vocab=8, n_pairs=2, n_queries=1)
set_seed(0)
mcfg=ModelConfig(vocab_size=task.vocab_size,d_model=128,n_layers=4,n_heads=4,d_ff=512,
                 max_seq_len=64,position="rope")
tcfg=TrainConfig(steps=6000,batch_size=64,lr=1e-3,warmup_steps=200,eval_every=6000,
                 eval_batches=20,device=DEVICE,log_every=10**9)
model=TransformerLM(mcfg); rng=np.random.default_rng(0)
tb=lambda s:(lambda b:(b["input_ids"],b["targets"],b["loss_mask"]))(task.sample_batch(64,DEVICE,rng))
ev=lambda s:evaluate_synthetic(model,task,20,64,DEVICE,np.random.default_rng(999))
h=run_training(model,tcfg,tb,ev,log_prefix="[instr]")
print("final acc", h[-1]["token_acc"])

# enable capture
for blk in model.blocks:
    blk.attn.capture_attn = True
model.eval()
b = task.sample_batch(6, DEVICE, np.random.default_rng(3))
with torch.no_grad():
    logits,_ = model(b["input_ids"])
inp = b["input_ids"].cpu().numpy()
ans_pos = b["ans_pos"].cpu().numpy()   # (B,1)
pred = logits.float().argmax(-1).cpu().numpy()
tgt = b["targets"].cpu().numpy()

# sequence layout (context 2D=4, then query at pos4): [ka,va,kb,vb,q]
for i in range(4):
    qp = int(ans_pos[i,0])
    print(f"\nseq{i}: tokens={inp[i].tolist()}  query@{qp}={inp[i][qp]}  target={tgt[i][qp]} pred={pred[i][qp]}")
    # which context key matches the query, and where is its value?
    for L,blk in enumerate(model.blocks):
        A = blk.attn.last_attn[i]          # (H, T, T)
        row = A[:, qp, :].mean(0).cpu().numpy()   # avg over heads, attention from query pos
        top = np.argsort(row)[::-1][:4]
        print(f"  L{L} query-attn top: " + ", ".join(f"pos{p}({row[p]:.2f})" for p in top))
