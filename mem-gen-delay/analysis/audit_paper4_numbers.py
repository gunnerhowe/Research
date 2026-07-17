"""Audit: list every prose line of paper4/main.tex containing a bare numeric literal,
so hand-typed measured values can be eyeballed against artifacts."""
import re

tex = open("paper4/main.tex").read()
tex = re.sub(r"(?<!\\)%.*", "", tex)
lines = tex.split("\n")
PROTOCOL = re.compile(
    r"^(0\.10|0\.05|0\.85|0\.80|2\.0|1\.5|1\.3|5|90|7/10|4/5|9/10|1\.11|1\.36|825|1125|"
    r"20\d\d|1|2|3|4|10|11|16|25|30|96|64|256|320|512)$")
for i, ln in enumerate(lines):
    if "\\newcommand" in ln or "includegraphics" in ln or "label{" in ln:
        continue
    nums = re.findall(r"\d[\d,]*\.?\d*", ln)
    interesting = [n for n in nums if not PROTOCOL.fullmatch(n)]
    if interesting:
        print(f"{i:4d}: {ln.strip()[:150]}")
