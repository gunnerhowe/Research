"""House rule 1 for the emergence-forecasting paper: regenerate numbers and byte-diff."""
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = subprocess.run([sys.executable, os.path.join(ROOT, "paper4", "gen_numbers.py")],
                   cwd=ROOT, stdout=subprocess.DEVNULL)
if r.returncode != 0:
    print("FAIL: gen_numbers.py errored")
    sys.exit(1)
d = subprocess.run(["git", "diff", "--exit-code", "--",
                    "paper4/numbers.tex", "paper4/numbers.json"], cwd=ROOT)
if d.returncode != 0:
    print("\nFAIL: committed numbers differ from regeneration.")
    sys.exit(1)
print("PASS: every cited number matches regeneration from the scored artifacts.")
