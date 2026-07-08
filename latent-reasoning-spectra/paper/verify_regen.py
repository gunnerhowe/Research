"""Verify every number in the paper is machine-generated: re-run
experiments/make_figures.py and confirm paper/numbers.tex is reproduced bit-for-bit
from results/*.json + runs/*.npz."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = ROOT / "paper" / "numbers.tex"


def main():
    before = NUM.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as td:
        shutil.copy(NUM, Path(td) / "numbers.tex.bak")
        r = subprocess.run([sys.executable, "-X", "utf8",
                            str(ROOT / "experiments" / "make_figures.py")],
                           cwd=ROOT, capture_output=True, text=True)
        after = NUM.read_text(encoding="utf-8")
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            sys.exit("make_figures.py failed")
        if before != after:
            sys.exit("numbers.tex CHANGED on regeneration -- stale numbers in paper!")
    print("verify_regen: OK (numbers.tex reproduces exactly)")


if __name__ == "__main__":
    main()
