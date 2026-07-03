"""Three probes to fix the KS tpp collapse (dev seed 1).

A. Composition: marg+tpp (climatology repairs dynamics, catalog repairs
   event law) on ksr03.
B. Gentle-long: tpp w3 st25 at lr 3e-5 for 2400 steps on ksr03.
C. Milder deficit: r=0.2 noise variant, fresh base, tpp w3 st25.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    print(f">>> {' '.join(cmd)}", flush=True)
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT)
    assert r.returncode == 0, cmd


TS = str(ROOT / "src/train_surrogate.py")
EV = str(ROOT / "src/eval_surrogate.py")
INIT3 = str(ROOT / "runs/ks_grid_basephase_s1/ckpt.pt")

# C: milder noise (first: lightest, independent)
if not (ROOT / "data/ksr02/l96_train.npz").exists():
    run([str(ROOT / "scripts/make_noisy.py"), "--src", "data/ks", "--r", "0.2"])
base2 = ROOT / "runs/kfix_base_r02"
if not (base2 / "metrics.json").exists():
    run([TS, "--phase", "base", "--condition", "base", "--seed", "1",
         "--data_dir", "data/ksr02", "--ref_dir", "data/ks",
         "--out", str(base2)])
    run([EV, "--ckpt", str(base2 / "ckpt.pt"), "--data_dir", "data/ks"])
out = ROOT / "runs/kfix_tpp_r02"
if not (out / "metrics.json").exists():
    run([TS, "--phase", "ft", "--condition", "tpp", "--seed", "1",
         "--init", str(base2 / "ckpt.pt"), "--data_dir", "data/ksr02",
         "--ref_dir", "data/ks", "--steps", "800", "--lr", "1e-4",
         "--aux_weight", "3", "--self_tpp_steps", "25", "--out", str(out)])
    run([EV, "--ckpt", str(out / "ckpt.pt"), "--data_dir", "data/ks"])

# B: gentle-long
out = ROOT / "runs/kfix_gentle"
if not (out / "metrics.json").exists():
    run([TS, "--phase", "ft", "--condition", "tpp", "--seed", "1",
         "--init", INIT3, "--data_dir", "data/ksr03", "--ref_dir", "data/ks",
         "--steps", "2400", "--lr", "3e-5", "--aux_weight", "3",
         "--self_tpp_steps", "25", "--out", str(out)])
    run([EV, "--ckpt", str(out / "ckpt.pt"), "--data_dir", "data/ks"])

# A: composition (last; memory-heavy -> reduced rollout batch + Sinkhorn pts)
out = ROOT / "runs/kfix_margtpp"
if not (out / "metrics.json").exists():
    run([TS, "--phase", "ft", "--condition", "margtpp", "--seed", "1",
         "--init", INIT3, "--data_dir", "data/ksr03", "--ref_dir", "data/ks",
         "--steps", "800", "--lr", "1e-4", "--aux_weight", "3",
         "--self_tpp_steps", "25", "--marg_weight", "1", "--stats", "ks",
         "--b_roll", "8", "--sink_pts", "512",
         "--out", str(out)])
    run([EV, "--ckpt", str(out / "ckpt.pt"), "--data_dir", "data/ks"])

keys = ["iet_ks", "iet_w1", "rate_ratio", "fano_logerr", "rl_logerr",
        "tpp_ll", "marg_w1", "psd_logdist", "acf_rmse"]
print("\n=== KS fix probes (gt ceiling -0.0501) ===")
for name in ["ks_grid_basephase_s1", "kfix_margtpp", "kfix_gentle",
             "kfix_base_r02", "kfix_tpp_r02"]:
    p = ROOT / f"runs/{name}/metrics.json"
    if p.exists():
        m = json.load(open(p))
        print(f"{name:22s}", {k: round(float(m[k]), 3) for k in keys})
