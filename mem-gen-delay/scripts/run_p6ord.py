"""P6 R-ORD: ordinal cross-family sweep (OLMo-1B-hf + OLMo-2-0425-1B).

Resolves each family's available checkpoint revisions from the Hub at runtime and picks
the one nearest each target token count (targets frozen in the prereg spirit: dense early
where transitions live, sparse late). Probes via src/probe_pythia.py with --purge so peak
disk stays ~one checkpoint. Idempotent via the prober's skip-by-revision.
"""
import os
import re
import subprocess
import sys

os.environ.setdefault("HF_HOME", "E:/GitHub/Research/hf_cache")
from huggingface_hub import list_repo_refs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS_B = [0, 1, 4, 8, 12, 16, 21, 42, 63, 105, 210, 420, 1000, 2000, 4000]


def resolve(repo, stage1_only=False):
    refs = [b.name for b in list_repo_refs(repo).branches]
    rows = []
    for r in refs:
        if stage1_only and "stage1" not in r:
            continue
        m = re.search(r"step(\d+)-tokens(\d+)B", r)
        if m:
            rows.append((int(m.group(2)), r))
    rows.sort()
    picked = []
    for t in TARGETS_B:
        best = min(rows, key=lambda x: abs(x[0] - t))
        if best[1] not in picked:
            picked.append(best[1])
    return picked


def main():
    jobs = [("allenai/OLMo-1B-hf", "olmo1", False),
            ("allenai/OLMo-2-0425-1B", "olmo2", True)]
    for repo, tag, s1 in jobs:
        revs = resolve(repo, s1)
        print(f"{repo}: {len(revs)} checkpoints -> {revs}", flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "src", "probe_pythia.py"),
                        "--model", repo, "--revisions", ",".join(revs), "--purge",
                        "--out", os.path.join(ROOT, "runs", "p6ord", f"{tag}.jsonl")],
                       check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
