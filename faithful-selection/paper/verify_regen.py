"""House rule check: regenerate numbers.tex and confirm it is byte-identical
to the committed one (no hand-edited numbers). Exit nonzero on drift."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = ROOT / "paper" / "numbers.tex"

old = NUM.read_text()
subprocess.run([sys.executable, str(ROOT / "paper" / "gen_paper_numbers.py")],
               check=True)
new = NUM.read_text()

if old == new:
    print("OK: numbers.tex is reproducible (byte-identical on regeneration)")
    sys.exit(0)

print("DRIFT: numbers.tex changed on regeneration. Diff:")
old_lines = dict(l.split("}{", 1) for l in old.splitlines()
                 if l.startswith("\\newcommand"))
new_lines = dict(l.split("}{", 1) for l in new.splitlines()
                 if l.startswith("\\newcommand"))
for k in sorted(set(old_lines) | set(new_lines)):
    if old_lines.get(k) != new_lines.get(k):
        print(f"  {k}: {old_lines.get(k)} -> {new_lines.get(k)}")
sys.exit(1)
