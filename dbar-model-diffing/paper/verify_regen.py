"""Verify that numbers.tex regenerates identically from the committed results JSONs.
Run before submission: any diff means a hand-edited number or stale results."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
committed = (HERE / "numbers.tex").read_text()
subprocess.run([sys.executable, str(HERE / "gen_paper_numbers.py")], check=True)
regenerated = (HERE / "numbers.tex").read_text()
if committed != regenerated:
    print("MISMATCH: numbers.tex does not regenerate from results/ — investigate.")
    sys.exit(1)
print("OK: numbers.tex regenerates byte-identically from results/.")
