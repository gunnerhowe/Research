"""Print aggregate results (mean ± std over seeds) for every suite/config.

Usage: python experiments/summarize.py [suite ...]
"""

import sys
from collections import defaultdict

import numpy as np

from make_figures import finals, load


def show(suite, group_keys=("image", "mode"), metrics=("psnr", "hf_psnr", "ssim")):
    recs = load(suite)
    if not recs:
        print(f"== {suite}: no data yet")
        return
    groups = defaultdict(list)
    for r in recs:
        key = tuple(r.get(k, "-") for k in group_keys)
        groups[key].append(r)
    print(f"== {suite} ({len(recs)} runs)")
    for key in sorted(groups):
        sub = groups[key]
        f = finals(sub, keys=metrics + ("hf_err",))
        label = "/".join(str(k) for k in key if k != "-")
        # count seeds per config
        seeds = defaultdict(set)
        for r in sub:
            seeds[r["config"]].add(r["seed"])
        if label:
            print(f"-- {label}")
        for cfg in sorted(f):
            vals = "  ".join(
                f"{m}={f[cfg][m][0]:.2f}±{f[cfg][m][1]:.2f}"
                for m in metrics + ("hf_err",) if m in f[cfg]
            )
            print(f"   {cfg:<22} [{len(seeds[cfg])} seeds] {vals}")
        # interp reference if present
        interp = [r["interp"]["psnr"] for r in sub if "interp" in r]
        if interp:
            ih = [r["interp"]["hf_psnr"] for r in sub if "interp" in r]
            print(f"   {'(griddata interp)':<22} psnr={np.mean(interp):.2f} "
                  f"hf_psnr={np.mean(ih):.2f}")


def show_ablations():
    recs = load("ablations")
    if not recs:
        print("== ablations: no data yet")
        return
    groups = defaultdict(list)
    for r in recs:
        groups[r["tag"]].append(r["history"][-1]["psnr"])
    print(f"== ablations ({len(recs)} runs)")
    for tag in sorted(groups):
        v = groups[tag]
        print(f"   {tag:<22} [{len(v)} runs] psnr={np.mean(v):.2f}±{np.std(v):.2f}")


if __name__ == "__main__":
    suites = sys.argv[1:] or ["exp1_1d", "exp2_1d", "exp1_2d", "exp2_2d"]
    for s in suites:
        if s == "ablations":
            show_ablations()
        elif s.endswith("1d"):
            show(s, group_keys=(), metrics=("psnr",))
        else:
            show(s)
    if not sys.argv[1:]:
        show_ablations()
