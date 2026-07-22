"""Byte-check: numbers.tex must equal a fresh regeneration from committed artifacts.
Run before every compile that matters. Exits nonzero on any drift or if R2c macros are
absent at finalization time (pass --allow-pending during drafting)."""
import subprocess
import sys
import os

here = os.path.dirname(os.path.abspath(__file__))
cur = open(os.path.join(here, "numbers.tex"), "rb").read()
subprocess.run([sys.executable, os.path.join(here, "gen_numbers.py")], check=True)
new = open(os.path.join(here, "numbers.tex"), "rb").read()
if cur != new:
    print("FAIL: numbers.tex drifted from regeneration")
    sys.exit(1)
if b"RtwoCready}{1}" not in new and "--allow-pending" not in sys.argv:
    print("FAIL: R2c macros not present — finalization blocked until the sealed "
          "confirmation exists (use --allow-pending during drafting)")
    sys.exit(1)
print("verify_regen: PASS" + (" (R2c pending)" if b"RtwoCready}{1}" not in new else ""))
