"""Seed-0 calibration sweep for aux weights / lr (dev seed; not reported)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT = ROOT / "runs/dt02r03_base_s0/ckpt.pt"
COMMON = ["--phase", "ft", "--seed", "0", "--init", str(INIT),
          "--data_dir", "data/dt02r03", "--ref_dir", "data/dt02",
          "--steps", "800"]

CFGS = [
    ("cal_baseft_lr1e4", ["--condition", "base", "--lr", "1e-4"]),
    ("cal_tpp_w3_lr1e4", ["--condition", "tpp", "--aux_weight", "3", "--lr", "1e-4"]),
    ("cal_tpp_w10_lr1e4", ["--condition", "tpp", "--aux_weight", "10", "--lr", "1e-4"]),
    ("cal_tpp_w30_lr1e4", ["--condition", "tpp", "--aux_weight", "30", "--lr", "1e-4"]),
    ("cal_tpp_w10_lr3e4", ["--condition", "tpp", "--aux_weight", "10", "--lr", "3e-4"]),
    ("cal_marg_w1_lr1e4", ["--condition", "marg", "--marg_weight", "1", "--lr", "1e-4"]),
    ("cal_marg_w3_lr1e4", ["--condition", "marg", "--marg_weight", "3", "--lr", "1e-4"]),
]


def run(cmd):
    print(f">>> {' '.join(cmd)}", flush=True)
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT)
    assert r.returncode == 0, cmd


for name, extra in CFGS:
    out = ROOT / f"runs/{name}"
    if not (out / "ckpt.pt").exists():
        run([str(ROOT / "src/train_surrogate.py")] + COMMON +
            ["--out", str(out)] + extra)
    if not (out / "metrics.json").exists():
        run([str(ROOT / "src/eval_surrogate.py"), "--ckpt",
             str(out / "ckpt.pt"), "--data_dir", "data/dt02"])

keys = ["iet_ks", "iet_w1", "rate_ratio", "fano_logerr", "rl_logerr",
        "tpp_ll", "marg_w1", "psd_logdist", "acf_rmse", "div_frac"]
print("\n=== calibration summary (base_s0 first) ===")
base = json.load(open(ROOT / "runs/dt02r03_base_s0/metrics.json"))
print(f"{'base(no ft)':22s}", {k: round(float(base[k]), 4) for k in keys})
for name, _ in CFGS:
    m = json.load(open(ROOT / f"runs/{name}/metrics.json"))
    print(f"{name:22s}", {k: round(float(m[k]), 4) for k in keys})
