"""P8-R2c confirmation fleet (prereg plan_p8.md commit ca0312d). Seeds 504-510, full
chains, protocol verbatim from R2/R2b: specimen -> guard v2 -> refaith + reburn2(beta16,
carrier head per guard) + hold. Idempotent."""
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
    from train_lm import TinyLM, probe_batch, run_probes
    m = TinyLM(2048, 256, 4, 2, 256)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    pb = probe_batch(2048)
    _, _, _, _, pvh = run_probes(m, pb, pb.shape[1] // 2, heads=True)
    heads = pvh[0]
    h = int(max(range(len(heads)), key=lambda i: heads[i]))
    print(f"   carrier head: L0H{h} ({[round(x,3) for x in heads]})", flush=True)
    return h


def main():
    for s in range(504, 511):
        run_one(f"spec_s{s}", ["--condition", "rep", "--seed", str(s),
                               "--steps", "16000", "--save_ckpt"])
        spec = os.path.join(GRID, f"spec_s{s}", "model.pt")
        run_one(f"guard_s{s}", ["--condition", "shufrep", "--seed", str(s + 50),
                                "--steps", "12000", "--save_ckpt", "--p_rep", "1.0",
                                "--shuf_uniform", "0.25", "--init_from", spec])
        g = os.path.join(GRID, f"guard_s{s}", "model.pt")
        run_one(f"refaith_s{s}", ["--condition", "rep", "--seed", str(s + 100),
                                  "--steps", "10000", "--log_heads", "--init_from", g])
        h = home_head(g)
        run_one(f"reburn2_s{s}", ["--condition", "rep", "--seed", str(s + 400),
                                  "--steps", "10000", "--log_heads", "--init_from", g,
                                  "--scaffold", "sink", "--scaffold_beta", "16",
                                  "--scaffold_layer", "0", "--scaffold_head", str(h)])
        run_one(f"hold_s{s}", ["--condition", "shufrep", "--seed", str(s + 300),
                               "--steps", "10000", "--init_from", g])
    print("P8-R2c fleet complete", flush=True)


if __name__ == "__main__":
    main()
