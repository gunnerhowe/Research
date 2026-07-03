"""KS calibration: fix the ratio-objective tracking failure on clustered
events. Sweep aux_weight x self_tpp_steps on seed 1 (KS dev seed)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT = ROOT / "runs/ks_grid_basephase_s1/ckpt.pt"
COMMON = ["--phase", "ft", "--condition", "tpp", "--seed", "1",
          "--init", str(INIT), "--data_dir", "data/ksr03",
          "--ref_dir", "data/ks", "--steps", "800", "--lr", "1e-4"]

CFGS = [
    ("kcal_w1_st25", ["--aux_weight", "1", "--self_tpp_steps", "25"]),
    ("kcal_w3_st25", ["--aux_weight", "3", "--self_tpp_steps", "25"]),
    ("kcal_w10_st25", ["--aux_weight", "10", "--self_tpp_steps", "25"]),
    ("kcal_w1_st5", ["--aux_weight", "1", "--self_tpp_steps", "5"]),
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
             str(out / "ckpt.pt"), "--data_dir", "data/ks"])

keys = ["iet_ks", "iet_w1", "rate_ratio", "fano_logerr", "rl_logerr",
        "tpp_ll", "marg_w1", "acf_rmse"]
print("\n=== KS calibration (base + gt ceiling -0.0501) ===")
base = json.load(open(ROOT / "runs/ks_grid_basephase_s1/metrics.json"))
print(f"{'base':14s}", {k: round(float(base[k]), 3) for k in keys})
for name, _ in CFGS:
    m = json.load(open(ROOT / f"runs/{name}/metrics.json"))
    print(f"{name:14s}", {k: round(float(m[k]), 3) for k in keys})
