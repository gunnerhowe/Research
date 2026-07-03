"""Scan base-surrogate event-timing deficit vs training-set size."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for n in [2, 4, 8]:
    out = ROOT / f"runs/scan_base_n{n}"
    ck = out / "ckpt.pt"
    if not ck.exists():
        r = subprocess.run([sys.executable, str(ROOT / "src/train_surrogate.py"),
                            "--phase", "base", "--condition", "base",
                            "--seed", "0", "--n_traj", str(n),
                            "--out", str(out)], cwd=ROOT)
        assert r.returncode == 0
    if not (out / "metrics.json").exists():
        r = subprocess.run([sys.executable, str(ROOT / "src/eval_surrogate.py"),
                            "--ckpt", str(ck)], cwd=ROOT)
        assert r.returncode == 0

print("\n=== deficit scan summary ===")
keys = ["div_frac", "rate_ratio", "iet_ks", "iet_w1", "fano_logerr",
        "rl_logerr", "tpp_ll", "marg_w1", "psd_logdist"]
full = json.load(open(ROOT / "runs/base_base_s0/metrics.json"))
print("n=64:", {k: round(full.get(k, float("nan")), 4) for k in keys})
for n in [2, 4, 8]:
    m = json.load(open(ROOT / f"runs/scan_base_n{n}/metrics.json"))
    print(f"n={n}: ", {k: round(m.get(k, float("nan")), 4) for k in keys})
