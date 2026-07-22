"""P8 pilot orchestration (prereg plan_p8.md). Library = {M0 induction (attention fp),
M6 depth (subspace fp)} — the two distinct non-interfering mechanisms the substrate holds.
Idempotent (skips a run if its summary.json exists). Sequence:

  specimen -> capture -> guard(M0), guard(M6) -> watch streams per guarded skill:
    P-FAITHFUL  reteach the guarded skill as learned            (positive)
    D-RELOC     reteach M0 with the home head burned            (relocation disguise)
    D-IDIO      reteach M6 with the depth subspace erased       (subspace disguise)
    N1          continue-train, guarded skill absent            (omission negative)
    N2          reteach with scrambled answers                  (marker-scramble negative)

Watch runs log per-skill proximity INLINE every eval on the fixed probe banks (never the
stream), against the frozen specimen fingerprints. Scored one-shot by score_zoo_pilot.py.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, "runs", "zoo_pilot")
PY = sys.executable
SPEC = os.path.join(GRID, "specimen_s1")
LIB = os.path.join(GRID, "fp_lib.json")
MODEL = os.path.join(SPEC, "model.pt")
BIG = ["--n_layers", "6", "--d", "384", "--n_heads", "8"]
LIBSK = "M0,M6"
WEIGHTS = "M0:3,M6:0.4"


def run(cmd):
    print(">>", " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def train(out, extra):
    if os.path.exists(os.path.join(out, "summary.json")):
        print("skip (done):", os.path.basename(out), flush=True); return
    run([PY, os.path.join(ROOT, "src", "train_zoo.py"), "--mode", extra[0],
         "--out_dir", out, "--skills", LIBSK] + BIG + extra[1:])


def main():
    os.makedirs(GRID, exist_ok=True)
    # 1) specimen (skip if already trained by the standalone launch)
    train(SPEC, ["specimen", "--steps", "12000", "--eval_every", "500", "--seed", "1",
                 "--lr", "1e-3", "--weights", WEIGHTS, "--pool", "12288",
                 "--pool_refresh", "1500"])
    # 2) capture the fingerprint library
    if not os.path.exists(LIB):
        run([PY, os.path.join(ROOT, "src", "train_zoo.py"), "--mode", "capture",
             "--init_from", MODEL, "--skills", LIBSK, "--out_lib", LIB,
             "--conf_out", os.path.join(GRID, "fp_confusion.json")] + BIG)
    # 3) guard each skill by ACTIVE unlearning (scramble its answers -> drives behavior AND
    #    structure to chance; omission left the structure intact). Retained skill kept.
    guard_common = ["--init_from", MODEL, "--steps", "8000", "--eval_every", "250",
                    "--lr", "5e-4", "--pool", "12288", "--pool_refresh", "1500",
                    "--weights", "M0:1,M6:1"]
    for gk in ("M0", "M6"):
        train(os.path.join(GRID, f"guard_{gk}"),
              ["guard", "--scramble", gk] + guard_common)
    # 4) watch streams: reteach the guarded skill at a LOW weight (slow re-emergence -> lead
    #    time to measure); log proximity inline. RW = reteach weight per skill (M6 dense).
    RW = {"M0": "M0:0.4,M6:0.5", "M6": "M0:0.5,M6:0.15"}
    def watch(name, gk, extra, weights=None):
        gckpt = os.path.join(GRID, f"guard_{gk}", "model.pt")
        train(os.path.join(GRID, name),
              ["watch", "--init_from", gckpt, "--fp_lib", LIB, "--steps", "12000",
               "--eval_every", "200", "--lr", "5e-4", "--pool", "12288",
               "--pool_refresh", "1500", "--weights", weights or RW[gk]] + extra)
    # M0 (attention fp): faithful, relocation disguise, omission, marker-scramble
    watch("watch_M0_faithful", "M0", ["--reteach", "M0"])
    watch("watch_M0_reloc", "M0", ["--reteach", "M0", "--burn_home"])
    watch("watch_M0_N1", "M0", ["--omit", "M0"], weights="M6:0.5")
    watch("watch_M0_N2", "M0", ["--reteach", "M0", "--scramble", "M0"])
    # M6 (subspace fp): faithful, subspace-erase disguise, omission, marker-scramble
    watch("watch_M6_faithful", "M6", ["--reteach", "M6"])
    watch("watch_M6_idio", "M6", ["--reteach", "M6", "--erase_subspace"])
    watch("watch_M6_N1", "M6", ["--omit", "M6"], weights="M0:0.5")
    watch("watch_M6_N2", "M6", ["--reteach", "M6", "--scramble", "M6"])
    print("P8 pilot fleet complete", flush=True)


if __name__ == "__main__":
    main()
