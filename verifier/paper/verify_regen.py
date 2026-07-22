"""Byte-identity check: regenerate numbers.tex and diff against the committed
file. Fails (exit 1) if any paper number drifted from the committed results."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = ROOT / "paper" / "numbers.tex"

before = NUM.read_text(encoding="utf-8") if NUM.exists() else None
subprocess.run([sys.executable, str(ROOT / "paper" / "gen_numbers.py")], check=True)
after = NUM.read_text(encoding="utf-8")

if before is None:
    print("numbers.tex did not exist; generated fresh.")
elif before == after:
    print("OK: numbers.tex is byte-identical to a fresh regeneration.")
else:
    print("MISMATCH: numbers.tex differs from regeneration — commit the regen.")
    sys.exit(1)
