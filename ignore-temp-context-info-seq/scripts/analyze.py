"""Aggregate result JSONs into tidy DataFrames, compute summary statistics and
significance tests. Importable by the figure script; runnable to dump CSVs.

    python -m scripts.analyze
"""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

try:
    from scipy import stats as sps
except Exception:
    sps = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

VARIANT_ORDER = ["nope", "sinusoidal", "learned", "rope", "alibi", "t5", "cable", "semrf"]
PRETTY = {"nope": "NoPE", "sinusoidal": "Sinusoidal", "learned": "Learned-Abs",
          "rope": "RoPE", "alibi": "ALiBi", "t5": "T5-bias", "cable": "CABLE",
          "semrf": "SemRF (ours)"}


def _load_all(pattern):
    out = []
    for p in glob.glob(pattern):
        if p.endswith(".ckpt.pt"):
            continue
        with open(p) as f:
            out.append(json.load(f))
    return out


# --------------------------- synthetic ------------------------------------- #
def load_synthetic():
    recs = _load_all(os.path.join(RES, "synthetic", "*", "*.json"))
    final, extrap, dist = [], [], []
    for r in recs:
        base = dict(task=r["task"], variant=r["variant"], seed=r["seed"], n_params=r["n_params"])
        final.append({**base, "token_acc": r["final"]["token_acc"], "seq_acc": r["final"]["seq_acc"],
                      "train_time_s": r.get("train_time_s")})
        for e in r.get("extrapolation", []):
            extrap.append({**base, "label": e["label"], "seq_len": e["seq_len"],
                           "token_acc": e["token_acc"], "seq_acc": e["seq_acc"]})
        for d in r.get("distance_buckets", []):
            dist.append({**base, "distance": d["distance"], "acc": d["acc"], "n": d["n"]})
    return (pd.DataFrame(final), pd.DataFrame(extrap), pd.DataFrame(dist))


def load_charlm():
    recs = _load_all(os.path.join(RES, "charlm", "*", "*.json"))
    main, extrap = [], []
    for r in recs:
        base = dict(corpus=r["corpus"], variant=r["variant"], seed=r["seed"], n_params=r["n_params"])
        main.append({**base, "test_bpc": r["test_bpc"], "train_time_s": r.get("train_time_s")})
        for L, bpc in r.get("extrapolation_bpc", {}).items():
            extrap.append({**base, "eval_len": int(L), "bpc": bpc})
    return pd.DataFrame(main), pd.DataFrame(extrap)


def agg(df, group, value):
    g = df.groupby(group)[value].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


def significance_synthetic(extrap):
    """SemRF vs each baseline at the longest eval length, per task (across seeds)."""
    rows = []
    if extrap.empty:
        return pd.DataFrame(rows)
    for task in sorted(extrap["task"].unique()):
        sub = extrap[extrap["task"] == task]
        Lmax = sub["seq_len"].max()
        atL = sub[sub["seq_len"] == Lmax]
        semrf = atL[atL["variant"] == "semrf"].sort_values("seed")["token_acc"].values
        for v in VARIANT_ORDER:
            if v == "semrf":
                continue
            base = atL[atL["variant"] == v].sort_values("seed")["token_acc"].values
            if len(semrf) == 0 or len(base) == 0:
                continue
            row = {"task": task, "seq_len": int(Lmax), "baseline": v,
                   "semrf_mean": float(np.mean(semrf)), "base_mean": float(np.mean(base)),
                   "delta": float(np.mean(semrf) - np.mean(base))}
            if sps is not None and len(semrf) == len(base) and len(semrf) >= 2:
                try:
                    t, p = sps.ttest_rel(semrf, base)
                    row["p_paired_t"] = float(p)
                except Exception:
                    row["p_paired_t"] = None
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    fdf, edf, ddf = load_synthetic()
    cdf, cedf = load_charlm()

    print("\n=== SYNTHETIC: final token accuracy (train length) ===")
    if not fdf.empty:
        piv = agg(fdf, ["task", "variant"], "token_acc")
        for task in sorted(piv["task"].unique()):
            print(f"\n{task}")
            t = piv[piv["task"] == task].set_index("variant").reindex(VARIANT_ORDER).dropna(how="all")
            for v, row in t.iterrows():
                print(f"  {PRETTY.get(v, v):16s} {row['mean']:.3f} ± {row['std']:.3f}  (n={int(row['count'])})")

    print("\n=== SYNTHETIC: extrapolation @ longest length ===")
    if not edf.empty:
        for task in sorted(edf["task"].unique()):
            sub = edf[edf["task"] == task]
            Lmax = sub["seq_len"].max()
            piv = agg(sub[sub["seq_len"] == Lmax], ["variant"], "token_acc").set_index("variant").reindex(VARIANT_ORDER).dropna(how="all")
            print(f"\n{task}  (len={Lmax})")
            for v, row in piv.iterrows():
                print(f"  {PRETTY.get(v, v):16s} {row['mean']:.3f} ± {row['std']:.3f}")

    sig = significance_synthetic(edf)
    if not sig.empty:
        print("\n=== SIGNIFICANCE: SemRF - baseline @ longest length ===")
        print(sig.to_string(index=False))

    print("\n=== CHAR-LM: test bpc ===")
    if not cdf.empty:
        piv = agg(cdf, ["corpus", "variant"], "test_bpc").set_index("variant").reindex(VARIANT_ORDER).dropna(how="all")
        for v, row in piv.iterrows():
            print(f"  {PRETTY.get(v, v):16s} {row['mean']:.4f} ± {row['std']:.4f}")

    os.makedirs(RES, exist_ok=True)
    if not fdf.empty:
        fdf.to_csv(os.path.join(RES, "summary_synthetic_final.csv"), index=False)
        edf.to_csv(os.path.join(RES, "summary_synthetic_extrap.csv"), index=False)
        ddf.to_csv(os.path.join(RES, "summary_synthetic_distance.csv"), index=False)
    if not cdf.empty:
        cdf.to_csv(os.path.join(RES, "summary_charlm.csv"), index=False)
        cedf.to_csv(os.path.join(RES, "summary_charlm_extrap.csv"), index=False)
    if not sig.empty:
        sig.to_csv(os.path.join(RES, "significance_synthetic.csv"), index=False)
    print("\nwrote summary CSVs to results/")


if __name__ == "__main__":
    main()
