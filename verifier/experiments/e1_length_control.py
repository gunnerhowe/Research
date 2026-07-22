"""E1 confound guard: does the signal effect beta_G survive controlling for text
length? (Verbosity is a known LLM-judge bias; signaled framings may be longer.)

Fits, per judge, y ~ s + g (+ length_z) with a stem random intercept, and reports
whether beta_G shrinks when length enters. Also reports whether signaled cells are
actually longer.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from novjudge.dataio import load_dataset, CELLS  # noqa: E402


def main():
    ds = load_dataset(str(ROOT / "data" / "dataset_v1.json"))
    scores = pd.read_csv(ROOT / "results" / "e0_results.scores.csv")

    # word length per (stem, cell)
    lens = []
    for st in ds["stems"]:
        for c in CELLS:
            lens.append({"stem": st["stem_id"], "cell": c, "words": len(st[c].split())})
    lens = pd.DataFrame(lens)
    df = scores.merge(lens, on=["stem", "cell"], how="left")
    df["lenz"] = (df["words"] - df["words"].mean()) / df["words"].std()

    print("mean words per cell:")
    print(df.groupby("cell")["words"].mean().round(1).to_dict())
    # is signal longer? compare signaled (g=1) vs plain (g=0)
    print(f"mean words  plain(g=0)={df[df.g==0].words.mean():.1f}  "
          f"signaled(g=1)={df[df.g==1].words.mean():.1f}")

    def within_fit(d, cols):
        """Within-stem (fixed-effects) OLS: demean each var by its stem mean, then
        regress with no intercept. Numerically stable for 4-obs-per-stem designs."""
        dd = d.copy()
        for c in ["y"] + cols:
            dd[c] = dd[c] - dd.groupby("stem")[c].transform("mean")
        X = dd[cols].to_numpy(float)
        y = dd["y"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return dict(zip(cols, beta))

    def cluster_ci(d, cols, key, B=2000, seed=0):
        rng = np.random.default_rng(seed)
        stems = d["stem"].unique()
        by = {s: g for s, g in d.groupby("stem")}
        vals = []
        for _ in range(B):
            pick = rng.choice(stems, len(stems), replace=True)
            boot = pd.concat([by[s] for s in pick], ignore_index=True)
            try:
                vals.append(within_fit(boot, cols)[key])
            except Exception:
                pass
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    out = {}
    for jid in sorted(df["judge"].unique()):
        d = df[df["judge"] == jid]
        g0 = within_fit(d, ["s", "g"])["g"]
        f1 = within_fit(d, ["s", "g", "lenz"])
        g1, s1, lenb = f1["g"], f1["s"], f1["lenz"]
        ci_g1 = cluster_ci(d, ["s", "g", "lenz"], "g")
        out[jid] = {"beta_G_nolen": g0, "beta_G_withlen": g1, "ci_beta_G_withlen": ci_g1,
                    "beta_S_withlen": s1, "beta_len": lenb,
                    "shrink_frac": (g0 - g1) / g0 if g0 else float("nan")}
        short = jid.split("/")[-1]
        print(f"\n{short}:")
        print(f"  beta_G  no-length = {g0:+.3f}")
        print(f"  beta_G with-length= {g1:+.3f}  CI{tuple(round(x,3) for x in ci_g1)}  (shrink {100*(g0-g1)/g0:.0f}%)")
        print(f"  beta_S with-length= {s1:+.3f}   beta_len(per SD words)={lenb:+.3f}")
        print(f"  -> signal effect {'SURVIVES' if ci_g1[0] > 0 else 'NOT robust to'} length control")

    (ROOT / "results" / "e1_length_control.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/e1_length_control.json")


if __name__ == "__main__":
    main()
