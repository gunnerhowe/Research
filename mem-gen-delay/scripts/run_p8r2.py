"""P8-R2 fleet (prereg plan_p8.md commit b127fa1). Idempotent. Sequence per seed
501-503: specimen (rep, 16k, ckpt) -> guard (shufrep continue, 8k, ckpt; + norep guard
secondary) -> reteach-faithful (rep continue, 10k) -> reteach-burned (rep continue with
the guarded model's layer-0 argmax prevtok head sink-burned, 10k, log_heads) ->
guard-hold negative (shufrep continue, 10k).
"""
import json
import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, "runs", "grid8r2")
PY = sys.executable
sys.path.insert(0, os.path.join(ROOT, "src"))


def run_one(name, extra):
    out = os.path.join(GRID, name)
    if os.path.exists(os.path.join(out, "summary.json")):
        print(f"skip (done): {name}", flush=True)
        return
    cmd = [PY, os.path.join(ROOT, "src", "train_lm.py"), "--out_dir", out,
           "--lr", "1e-3"] + extra
    print(">>", name, flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def home_head(ckpt):
    """Layer-0 argmax prev-token head of a checkpoint (the surviving scaffold)."""
    from train_lm import TinyLM, probe_batch, run_probes
    m = TinyLM(2048, 256, 4, 2, 256)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    pb = probe_batch(2048)
    _, _, _, _, pvh = run_probes(m, pb, pb.shape[1] // 2, heads=True)
    heads = pvh[0]                                   # layer-0 per-head prevtok
    h = int(max(range(len(heads)), key=lambda i: heads[i]))
    print(f"   home head of {os.path.basename(os.path.dirname(ckpt))}: "
          f"L0H{h} (prevtok {heads[h]:.3f}; all {heads})", flush=True)
    return h


def main():
    os.makedirs(GRID, exist_ok=True)
    for s in (501, 502, 503):
        run_one(f"spec_s{s}", ["--condition", "rep", "--seed", str(s),
                               "--steps", "16000", "--save_ckpt"])
    for s in (501, 502, 503):
        spec = os.path.join(GRID, f"spec_s{s}", "model.pt")
        run_one(f"guard_s{s}", ["--condition", "shufrep", "--seed", str(s + 50),
                                "--steps", "8000", "--save_ckpt",
                                "--init_from", spec])
        run_one(f"guardnr_s{s}", ["--condition", "norep", "--seed", str(s + 60),
                                  "--steps", "8000", "--save_ckpt",
                                  "--init_from", spec])
    for s in (501, 502, 503):
        g = os.path.join(GRID, f"guard_s{s}", "model.pt")
        run_one(f"refaith_s{s}", ["--condition", "rep", "--seed", str(s + 100),
                                  "--steps", "10000", "--log_heads",
                                  "--init_from", g])
        h = home_head(g)
        run_one(f"reburn_s{s}", ["--condition", "rep", "--seed", str(s + 200),
                                 "--steps", "10000", "--log_heads",
                                 "--init_from", g, "--scaffold", "sink",
                                 "--scaffold_beta", "8", "--scaffold_layer", "0",
                                 "--scaffold_head", str(h)])
        run_one(f"hold_s{s}", ["--condition", "shufrep", "--seed", str(s + 300),
                               "--steps", "10000", "--init_from", g])
    print("P8-R2 fleet complete", flush=True)


if __name__ == "__main__":
    main()
