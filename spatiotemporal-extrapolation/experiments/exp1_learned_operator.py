"""E1 — learned translation-equivariant Koopman operator at small L.

Trains per-size models (L in {22,44,66,88} x 3 seeds; the L=22 models double as
the zero-shot null for E3) and checks the K3 gate at L=22 (PLAN.md):
  - per-sector leading eigenvalues vs EDMD ground truth (median rel err <= 15%),
  - model-GENERATED spectral density vs measured (median |log10 ratio| <= 0.10,
    k in [0.1, 2.2]),
  - C(r) relative L2 <= 10%.
Also validates the data-free io-weight eigenvalue selection against the
data-correlation selection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (DATA, RUNS, SEEDS, SIZES_TRAIN, analyze_measurement,        # noqa: E402
                    n_of, save_json, DX)
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from specext.edmd import ModeGrid                                               # noqa: E402
from specext.koopman import (ConvKoopman, train_model, model_spectrum,          # noqa: E402
                             sector_stationary, generate_stationary,
                             eigcoord_corr_selection, L_BASE)
from specext.stats import (density_from_power, corr_from_power,                 # noqa: E402
                           median_abs_log10_ratio, rel_l2, band_mask)

MODELS = RUNS / "models"
MODELS.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def train_persize(L, seed, steps=20000, force=False):
    path = MODELS / f"persize_L{L:g}_s{seed}.pt"
    if path.exists() and not force:
        return path
    data = np.load(DATA / f"L{L:g}_s{seed}.npy", mmap_mode="r")
    model, wall = train_model([{"L": L, "data": data}], seed=seed, flow=False,
                              steps=steps, device=DEV,
                              log_fn=lambda s: print(f"[L{L:g} s{seed}]{s}"))
    torch.save({"state": model.state_dict(), "L": L, "seed": seed,
                "flow": False, "wall_s": wall, "steps": steps}, path)
    print(f"trained persize L={L:g} seed={seed} in {wall:.0f}s")
    return path


def load_model(path):
    ckpt = torch.load(path, map_location=DEV, weights_only=False)
    model = ConvKoopman(flow=ckpt["flow"]).to(DEV)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt


def gate_e1(edmd_curves):
    """K3 gate at L=22 for each seed's model; medians over seeds reported."""
    L = 22.0
    N = n_of(L)
    grid = ModeGrid(L, N)
    lam_edmd = (-edmd_curves["gamma"] + 1j * edmd_curves["omega"])  # (S, M)
    k = edmd_curves["k"]
    kb = band_mask(k, 0.1, 2.2)
    p_true = edmd_curves["p_mean"].mean(axis=0)     # (N//2+1,)
    s_true = density_from_power(p_true[grid.m_idx], L)
    r_grid = DX * np.arange(N)
    c_true = corr_from_power(p_true, L, r_grid)
    rows = []
    for seed in SEEDS:
        model, ckpt = load_model(MODELS / f"persize_L{L:g}_s{seed}.pt")
        spec = model_spectrum(model, ell=1.0, m_idx=grid.m_idx, N=N, device=DEV)
        lam_l = spec["lam"]
        lam_e = lam_edmd[seed]
        rel = np.abs(lam_l - lam_e) / np.abs(lam_e)
        med_lam = float(np.nanmedian(rel[kb]))
        # data-free vs data-correlation selection agreement
        dat = np.load(DATA / f"L{L:g}_s{seed}.npy", mmap_mode="r")
        lam_corr = eigcoord_corr_selection(model, dat, 1.0, grid.m_idx, N, device=DEV)
        sel_agree = float(np.nanmedian(np.abs(lam_l - lam_corr) /
                                       np.maximum(np.abs(lam_l), 1e-12)))
        # generative check
        stat = sector_stationary(model, dat, 1.0, N, device=DEV)
        gen = generate_stationary(model, stat["sig_z"], stat["m_all"], N,
                                  n_samples=4096, seed=seed, device=DEV)
        gen = gen - gen.mean(axis=1, keepdims=True)
        p_gen = 2.0 * (np.abs(np.fft.rfft(gen, axis=-1) / N) ** 2).mean(axis=0)
        p_gen[0] /= 2.0
        p_gen[-1] /= 2.0
        s_gen = density_from_power(p_gen[grid.m_idx], L)
        med_s = median_abs_log10_ratio(s_gen[kb], s_true[kb])
        c_gen = corr_from_power(p_gen, L, r_grid)
        c_err = rel_l2(c_gen, c_true)
        rows.append({"seed": seed, "median_rel_lam": med_lam,
                     "median_abs_log10_S": med_s, "c_rel_l2": c_err,
                     "selection_agreement_med": sel_agree,
                     "lam_learned": [[float(x.real), float(x.imag)] for x in lam_l],
                     "s_generated": s_gen.tolist(),
                     "train_wall_s": ckpt["wall_s"]})
        print(f"seed {seed}: lam med rel {med_lam:.3f}, S med|log10| {med_s:.3f}, "
              f"C(r) relL2 {c_err:.3f}, sel-agree {sel_agree:.4f}")
    med = lambda key: float(np.median([r[key] for r in rows]))
    gate = {"median_rel_lam": med("median_rel_lam"),
            "median_abs_log10_S": med("median_abs_log10_S"),
            "c_rel_l2": med("c_rel_l2"),
            "pass_lam": med("median_rel_lam") <= 0.15,
            "pass_S": med("median_abs_log10_S") <= 0.10,
            "pass_C": med("c_rel_l2") <= 0.10}
    gate["gate_e1_pass"] = gate["pass_lam"] and gate["pass_S"] and gate["pass_C"]
    return {"per_seed": rows, "gate": gate, "k": k.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for L in SIZES_TRAIN:
        for seed in SEEDS:
            train_persize(L, seed, steps=args.steps, force=args.force)
    edmd_curves = analyze_measurement(RUNS / "measure_L22.npz")
    out = gate_e1(edmd_curves)
    out["config"] = {"steps": args.steps, "sizes": SIZES_TRAIN, "seeds": SEEDS,
                     "device": DEV}
    save_json("exp1_learned.json", out)
    g = out["gate"]
    print(f"\nGATE E1 (K3): {'PASS' if g['gate_e1_pass'] else 'FAIL'} "
          f"(lam {g['median_rel_lam']:.3f}<=0.15 {g['pass_lam']}, "
          f"S {g['median_abs_log10_S']:.3f}<=0.10 {g['pass_S']}, "
          f"C {g['c_rel_l2']:.3f}<=0.10 {g['pass_C']})")


if __name__ == "__main__":
    main()
