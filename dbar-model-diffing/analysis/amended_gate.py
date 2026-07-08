"""Amended E0 gate (PLAN.md AMENDMENT 1) evaluated on the committed
results/exp0_existence.json — no data re-collected.

Amendments relative to the pre-registered gate:
- plateau eligibility: n* = argmax Δ_n over rows with d̄_n ≥ 2·floor_n, 2 ≤ n ≤ k_wall
  (a pair/seed with no eligible row has no separation);
- DSA-similar: mean DSA ≤ max(ν + 2·std_null, max_null) — indistinguishable from DSA's
  empirical null distribution.

Everything else (CKA ≥ 0.90, marginal d̄_1 ≤ 0.02, wall guard, mean − 2·std(Δ) > 0,
d̄_{n*} ≥ 2× floor_{n*}) is unchanged. Sign consistency across seeds is reported alongside
the strict 2·std inequality, since the latter penalizes effect-size heterogeneity.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.stdout.reconfigure(encoding="utf-8")

E0 = json.loads((RES / "exp0_existence.json").read_text())
R = E0["results"]


def amended_plateau(r):
    ok = [row for row in r["dbar_curve"]
          if 2 <= row["n"] <= r["k_wall"] and row["dbar"] >= 2 * row["floor"]]
    return max(ok, key=lambda x: x["delta"]) if ok else None


null_dsa = np.array([r["dsa"] for r in R["null"]])
nu, std_null = float(null_dsa.mean()), float(null_dsa.std())
dsa_sim_thresh = max(nu + 2 * std_null, float(null_dsa.max()))

report = {"nu_dsa": nu, "std_null_dsa": std_null,
          "dsa_similar_threshold": dsa_sim_thresh, "pairs": {}}

for pair in ("noise", "distill", "prune", "seed", "difftask"):
    rows = R[pair]
    wins = [amended_plateau(r) for r in rows]
    n_sep = sum(w is not None for w in wins)
    cka = float(np.mean([r["cka"] for r in rows]))
    dsa = float(np.mean([r["dsa"] for r in rows]))
    tv1 = float(np.mean([r["dbar_1"] for r in rows]))
    entry = {
        "cka_mean": cka, "cka_similar": bool(cka >= 0.90),
        "dsa_mean": dsa, "dsa_similar": bool(dsa <= dsa_sim_thresh),
        "dbar1_mean": tv1, "marginal_ok": bool(tv1 <= 0.02),
        "n_seeds_separating": int(n_sep), "n_seeds": len(rows),
        "plateaus": [
            None if w is None else
            {"seed": r["seed"], "n": w["n"], "dbar": w["dbar"],
             "floor": w["floor"], "delta": w["delta"],
             "ratio": w["dbar"] / max(w["floor"], 1e-12)}
            for r, w in zip(rows, wins)],
    }
    if n_sep == len(rows):
        deltas = np.array([w["delta"] for w in wins])
        ratios = np.array([w["dbar"] / max(w["floor"], 1e-12) for w in wins])
        entry.update({
            "delta_mean": float(deltas.mean()), "delta_std": float(deltas.std()),
            "sep_2sigma": bool(deltas.mean() - 2 * deltas.std() > 0),
            "sign_consistent": bool((deltas > 0).all()),
            "min_ratio": float(ratios.min()), "mean_ratio": float(ratios.mean()),
            "dbar_separates_all_seeds": True,
        })
        entry["PASS_strict"] = bool(entry["cka_similar"] and entry["dsa_similar"]
                                    and entry["marginal_ok"] and entry["sep_2sigma"])
        entry["PASS_sign_consistent"] = bool(
            entry["cka_similar"] and entry["dsa_similar"]
            and entry["marginal_ok"] and entry["sign_consistent"]
            and entry["min_ratio"] >= 2.0)
    else:
        entry.update({"dbar_separates_all_seeds": False,
                      "PASS_strict": False, "PASS_sign_consistent": False})
    report["pairs"][pair] = entry

gate_pairs = ("noise", "distill", "prune")
report["AMENDED_PASS_strict"] = bool(
    any(report["pairs"][p]["PASS_strict"] for p in gate_pairs))
report["AMENDED_PASS_sign_consistent"] = bool(
    any(report["pairs"][p]["PASS_sign_consistent"] for p in gate_pairs))

(RES / "amended_gate.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, indent=1))
for p in ("noise", "distill", "prune", "seed", "difftask"):
    e = report["pairs"][p]
    print(f"{p:9s} sep {e['n_seeds_separating']}/{e['n_seeds']} "
          f"CKAsim={e['cka_similar']} DSAsim={e['dsa_similar']} "
          f"strict={e.get('PASS_strict')} sign={e.get('PASS_sign_consistent')} "
          f"minratio={e.get('min_ratio', float('nan')):.1f}")
