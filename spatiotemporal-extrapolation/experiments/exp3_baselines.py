"""E3 — the credibility section: nulls and oracles.

 (i)  strict tiling (the null that translation invariance makes strong) and
      interpolated single-size nulls (interp-22, interp-88): small-L statistics
      smoothly interpolated in k, NO size flow — these isolate the flow's value;
 (ii) neural zero-shot: the L=22-trained operator evaluated on the target grid
      (locality route; the 2606.14597 analog at our scale, cite-compared);
 (iii) direct large-L training oracle at 176 (3 seeds) and reduced 1408 (1 seed),
      with compute multiples;
 (iv) EDMD fitted at the large size from LIMITED large-L data (our route uses
      zero large-L data).
K2 verdict: does any no-flow null tie the flow at L=1408?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (DATA, RESULTS, RUNS, SEEDS, SIZES_TRAIN, L_HOLDOUT,         # noqa: E402
                    L_TARGET, analyze_measurement, measure_size, n_of,
                    save_json)
from floweval import (predict_interp_null, strict_tiling_scores,                # noqa: E402
                      truth_from_measurement, score, score_new_band)
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from specext.edmd import ModeGrid                                               # noqa: E402
from specext.koopman import ConvKoopman, model_spectrum, train_model, L_BASE    # noqa: E402
from specext.scaling import interp_null                                         # noqa: E402

MODELS = RUNS / "models"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HEADLINE = ("gamma_med_rel", "s_med_log10", "c_rel_l2", "tau_med_rel",
            "slow_overlap_top16")
HIGHER_BETTER = {"slow_overlap_top16"}


def load_model(path):
    ckpt = torch.load(path, map_location=DEV, weights_only=False)
    model = ConvKoopman(flow=ckpt["flow"]).to(DEV)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt


def zero_shot_prediction(seed, L_t, curves22):
    model, _ = load_model(MODELS / f"persize_L22_s{seed}.pt")
    N_t = n_of(L_t)
    grid = ModeGrid(L_t, N_t)
    spec = model_spectrum(model, ell=1.0, m_idx=grid.m_idx, N=N_t, device=DEV)
    lam = spec["lam"]
    return {"gamma": -lam.real, "omega": np.abs(lam.imag),
            "s_density": interp_null(curves22["k"], curves22["s_density"][seed],
                                     grid.k, log_y=True)}


def oracle_prediction(L_t, seed, data_path, steps=20000, force=False):
    """Direct training at the target size (upper bound; costs large-L data)."""
    path = MODELS / f"oracle_L{L_t:g}_s{seed}.pt"
    if not path.exists() or force:
        data = np.load(data_path, mmap_mode="r")
        model, wall = train_model([{"L": L_t, "data": data}], seed=seed,
                                  flow=False, steps=steps, device=DEV,
                                  log_fn=lambda s: print(f"[oracle L{L_t:g} s{seed}]{s}"))
        torch.save({"state": model.state_dict(), "flow": False, "seed": seed,
                    "wall_s": wall, "steps": steps, "L": L_t}, path)
        print(f"trained oracle L={L_t:g} seed={seed} in {wall:.0f}s")
    model, ckpt = load_model(path)
    N_t = n_of(L_t)
    grid = ModeGrid(L_t, N_t)
    spec = model_spectrum(model, ell=1.0, m_idx=grid.m_idx, N=N_t, device=DEV)
    lam = spec["lam"]
    # oracle statics: the spectrum of its own training data (it owns large-L data)
    dat = np.load(data_path, mmap_mode="r")
    sl = dat[::50]
    p = 2.0 * (np.abs(np.fft.rfft(sl, axis=-1) / N_t) ** 2).mean(axis=0)
    s_density = p[grid.m_idx] / (2 * np.pi / L_t)
    return ({"gamma": -lam.real, "omega": np.abs(lam.imag), "s_density": s_density},
            ckpt["wall_s"])


def edmd_limited(L_t, T, tag):
    p = RUNS / f"measure_{tag}.npz"
    if not p.exists():
        p = measure_size(L_t, T=T, tag=tag)
    a = analyze_measurement(p)
    preds = [{"gamma": a["gamma"][s], "omega": a["omega"][s],
              "s_density": a["s_density"][s]} for s in range(len(SEEDS))]
    return preds, float(a["wall_s"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    args = ap.parse_args()

    curves22 = analyze_measurement(RUNS / "measure_L22.npz")
    curves44 = analyze_measurement(RUNS / "measure_L44.npz")
    curves88 = analyze_measurement(RUNS / "measure_L88.npz")
    # aggressive-base ladder {22, 33, 44} (PLAN deviation log, pre-registered)
    p33 = RUNS / "measure_L33.npz"
    if not p33.exists():
        p33 = measure_size(33.0, tag="L33")
    curves_small = {22.0: curves22, 33.0: analyze_measurement(p33),
                    44.0: curves44}
    meas = {L_HOLDOUT: analyze_measurement(RUNS / "measure_L176.npz"),
            L_TARGET: analyze_measurement(RUNS / "measure_L1408.npz")}

    results = {"targets": {}, "compute": {}}
    for L_t in (L_HOLDOUT, L_TARGET):
        truth = truth_from_measurement(meas[L_t])
        k_t = truth["k"]
        entry = {"methods": {}}

        def add(name, preds, extra=None):
            scores = [score(p, truth) for p in preds]
            nb = [score_new_band(p, truth) for p in preds]
            entry["methods"][name] = {
                "per_seed": scores,
                "median": {m: float(np.nanmedian([s[m] for s in scores]))
                           for m in scores[0]},
                "new_band": (None if nb[0] is None else
                             {m: float(np.nanmedian([s[m] for s in nb]))
                              for m in nb[0]}),
            }
            if extra:
                entry["methods"][name].update(extra)
            print(f"L={L_t:g} {name}: " + ", ".join(
                f"{m}={entry['methods'][name]['median'][m]:.4g}" for m in HEADLINE))

        add("interp22", [predict_interp_null(curves22, s, k_t) for s in range(3)])
        add("interp44", [predict_interp_null(curves44, s, k_t) for s in range(3)])
        add("interp88", [predict_interp_null(curves88, s, k_t) for s in range(3)])
        from floweval import fit_edmd_flows, predict_edmd_flow
        add("fitted_flow_smallbase",
            [predict_edmd_flow(fit_edmd_flows(curves_small, s), k_t, L_t)
             for s in range(3)])
        add("zero_shot", [zero_shot_prediction(s, L_t, curves22) for s in SEEDS])
        tile = [strict_tiling_scores(curves22, s, truth) for s in range(3)]
        entry["methods"]["strict_tiling"] = {
            "per_seed": tile,
            "median": {m: float(np.nanmedian([t[m] for t in tile]))
                       for m in ("c_rel_l2", "band_power_med_rel")}}
        print(f"L={L_t:g} strict_tiling: c_rel_l2="
              f"{entry['methods']['strict_tiling']['median']['c_rel_l2']:.4g}")

        if L_t == L_HOLDOUT:
            oracle_preds, oracle_walls = [], []
            for s in SEEDS:
                pr, w = oracle_prediction(L_t, s, DATA / f"L{L_t:g}_s{s}.npy",
                                          steps=args.steps)
                oracle_preds.append(pr)
                oracle_walls.append(w)
            add("oracle_direct", oracle_preds,
                {"train_wall_s": oracle_walls})
        else:
            # reduced single-seed oracle at 1408: dedicated stored short run
            if not (DATA / "L1408oracle_s0.npy").exists():
                p = measure_size(L_TARGET, T=25000.0, seeds=[0],
                                 store_fields=True, tag="L1408oracle")
                results["compute"]["oracle1408_datagen_wall_s"] = float(
                    np.load(p)["wall_s"])
            pr, w = oracle_prediction(L_t, 0, DATA / "L1408oracle_s0.npy",
                                      steps=args.steps)
            add("oracle_direct_reduced", [pr], {"train_wall_s": [w]})

        lim = {}
        for T in ((2000.0,) if L_t == L_HOLDOUT else (2000.0, 10000.0)):
            preds, wall = edmd_limited(L_t, T, f"L{L_t:g}_short{T:g}")
            add(f"edmd_limited_T{T:g}", preds, {"sim_wall_s": wall})
        results["targets"][f"{L_t:g}"] = entry

    # ---- K2 verdict at L_TARGET against exp2 flows
    with open(RESULTS / "exp2_flow.json") as f:
        exp2 = json.load(f)
    flows = exp2["targets"][f"{L_TARGET:g}"]["methods"]
    nulls = results["targets"][f"{L_TARGET:g}"]["methods"]
    null_names = ["interp22", "interp88", "zero_shot"]
    verdict = {}
    for fname in ("fitted_flow", "learned_flow"):
        wins = {}
        for m in HEADLINE:
            fvals = [s[m] for s in flows[fname]["per_seed"]]
            best_null, best_med = None, None
            for nn in null_names:
                nv = float(np.nanmedian([s[m] for s in nulls[nn]["per_seed"]]))
                if best_med is None or (nv > best_med if m in HIGHER_BETTER
                                        else nv < best_med):
                    best_med, best_null = nv, nn
            fmed = float(np.nanmedian(fvals))
            spread = float(np.nanstd(fvals, ddof=1)) if len(fvals) > 1 else 0.0
            better = (fmed > best_med + spread if m in HIGHER_BETTER
                      else fmed < best_med - spread)
            wins[m] = {"flow_median": fmed, "flow_spread": spread,
                       "best_null": best_null, "best_null_median": best_med,
                       "flow_wins": bool(better)}
        n_win = sum(w["flow_wins"] for w in wins.values())
        verdict[fname] = {"metrics": wins, "n_wins": n_win,
                          "majority": n_win > len(HEADLINE) / 2}
    verdict["k2_fires"] = not (verdict["fitted_flow"]["majority"] or
                               verdict["learned_flow"]["majority"])
    # aggressive-base comparison: smallbase flow vs interp-44 (same win rule)
    sb, i44 = nulls["fitted_flow_smallbase"], nulls["interp44"]
    sb_wins = {}
    for m in HEADLINE:
        fv = [s[m] for s in sb["per_seed"]]
        nv = float(np.nanmedian([s[m] for s in i44["per_seed"]]))
        fmed = float(np.nanmedian(fv))
        spread = float(np.nanstd(fv, ddof=1))
        better = (fmed > nv + spread if m in HIGHER_BETTER
                  else fmed < nv - spread)
        sb_wins[m] = {"flow_median": fmed, "interp44_median": nv,
                      "flow_wins": bool(better)}
    verdict["smallbase_vs_interp44"] = {
        "metrics": sb_wins, "n_wins": sum(w["flow_wins"] for w in sb_wins.values()),
        "majority": sum(w["flow_wins"] for w in sb_wins.values()) > len(HEADLINE) / 2}
    results["k2"] = verdict

    # ---- compute accounting (house rule: report the oracle's compute multiple)
    small_sim = sum(float(np.load(RUNS / f"measure_L{L:g}.npz")["wall_s"])
                    for L in SIZES_TRAIN)
    flow_train = float(np.mean([torch.load(MODELS / f"flow_s{s}.pt",
                                           map_location="cpu",
                                           weights_only=False)["wall_s"]
                                for s in SEEDS]))
    ours_total = small_sim + flow_train
    o176_train = float(np.mean(results["targets"][f"{L_HOLDOUT:g}"]["methods"]
                               ["oracle_direct"]["train_wall_s"]))
    o176_data = float(np.load(RUNS / "measure_L176.npz")["wall_s"])
    o1408_train = float(results["targets"][f"{L_TARGET:g}"]["methods"]
                        ["oracle_direct_reduced"]["train_wall_s"][0])
    o1408_data = results["compute"].get("oracle1408_datagen_wall_s", np.nan)
    results["compute"].update({
        "ours_smallL_sim_s": small_sim, "ours_flow_train_s": flow_train,
        "ours_total_s": ours_total,
        "oracle176_data_s": o176_data, "oracle176_train_s": o176_train,
        "oracle176_multiple": (o176_data + o176_train) / ours_total,
        "oracle1408_train_s": o1408_train,
        "oracle1408_data_s": o1408_data,
        "oracle1408_multiple": ((o1408_data + o1408_train) / ours_total
                                if np.isfinite(o1408_data) else None),
    })
    print(f"\nK2 verdict: fitted_flow wins {verdict['fitted_flow']['n_wins']}/5, "
          f"learned_flow wins {verdict['learned_flow']['n_wins']}/5 -> "
          f"{'K2 FIRES (flow adds nothing)' if verdict['k2_fires'] else 'flow adds value'}")
    save_json("exp3_baselines.json", results)


if __name__ == "__main__":
    main()
