"""Run the complete E0-E3 pipeline sequentially (single-GPU discipline).

Order:
  E0: capture M2 -> capture M3 -> capture M4 -> capture-pruned -> ablate -> analyze
  E1: causal interventions
  E2: routedness + EDMD
  E3: capture-valid, capture-natlin, capture-epochs, analyze

Each stage is a subprocess; failures abort the chain (except analyze stages, which are
retried at the end).  Progress: runs/pipeline_log.txt
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "runs" / "pipeline_log.txt"
PY = [sys.executable, "-X", "utf8", "-W", "ignore"]

STAGES = [
    ("E0 capture M2", PY + ["experiments/exp0_gate.py", "capture", "--model", "M2"]),
    ("E0 capture M3", PY + ["experiments/exp0_gate.py", "capture", "--model", "M3"]),
    ("E0 capture M4", PY + ["experiments/exp0_gate.py", "capture", "--model", "M4"]),
    ("E0 capture pruned", PY + ["experiments/exp0_gate.py", "capture-pruned"]),
    ("E0 ablate", PY + ["experiments/exp0_gate.py", "ablate"]),
    ("E0 analyze", PY + ["experiments/exp0_gate.py", "analyze"]),
    ("E1 causal", PY + ["experiments/exp1_causal.py"]),
    ("E2 koopman", PY + ["experiments/exp2_koopman.py"]),
    ("E3 capture valid", PY + ["experiments/exp3_robustness.py", "capture-valid"]),
    ("E3 capture natlin", PY + ["experiments/exp3_robustness.py", "capture-natlin"]),
    ("E3 capture epochs", PY + ["experiments/exp3_robustness.py", "capture-epochs"]),
    ("E3 analyze", PY + ["experiments/exp3_robustness.py", "analyze"]),
]


def main():
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as log:
        for name, cmd in STAGES:
            t0 = time.time()
            msg = f"\n===== {name} @ {time.strftime('%H:%M:%S')} =====\n"
            print(msg, flush=True)
            log.write(msg)
            log.flush()
            r = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            dt = (time.time() - t0) / 60
            status = "OK" if r.returncode == 0 else f"FAIL rc={r.returncode}"
            msg = f"===== {name}: {status} ({dt:.1f} min) =====\n"
            print(msg, flush=True)
            log.write(msg)
            log.flush()
            if r.returncode != 0 and "analyze" not in name.lower():
                print("aborting chain", flush=True)
                sys.exit(1)
    print("PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
