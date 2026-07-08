"""Shared experiment machinery: model/run caching, the three-metric pair evaluation,
gate logic (PLAN.md), and JSON I/O. All experiments import from here so every pair in
every experiment is measured by the identical procedure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dbar_diff import models as M                              # noqa: E402
from dbar_diff.baselines import cka_states, dsa_distance       # noqa: E402
from dbar_diff.dbar import dbar_pair_curve, plateau            # noqa: E402
from dbar_diff.entropy import (entropy_rate, estimation_wall,  # noqa: E402
                               fano_lower_bound, lz78_entropy)
from dbar_diff.symbolize import (belief_readout,               # noqa: E402
                                 emitted_readout, unigram_tv)
from dbar_diff.tasks import TASKS                              # noqa: E402

RESULTS = ROOT / "results"
CKPT = ROOT / "checkpoints"
RESULTS.mkdir(exist_ok=True)
CKPT.mkdir(exist_ok=True)

# pre-registered budgets (PLAN.md)
GEN_KW = dict(B=64, T=32768, burn=512)          # ~2.1e6 symbols/run (GRU)
GEN_KW_TF = dict(B=64, T=8192, burn=256)        # ~5.2e5 symbols/run (transformer)
NS_CLAIM = (1, 2, 3, 4, 6, 8, 12, 16, 24)
NS_FULL = NS_CLAIM + (32,)
EVAL_B, EVAL_L, SKIP = 32, 256, 16
SIGMA_GRID = (0.01, 0.02, 0.05, 0.1, 0.2)
SEEDS = (0, 1, 2, 3, 4)


def save_json(name, obj):
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=1, default=_np_default))
    print(f"[saved] {path}")


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ------------------------------------------------------------------ model cache

def get_model(kind, task_name, seed, **kw):
    """Train-or-load a cached model. kind: base | feven-base | distill | prune |
    tf-base | tf-distill."""
    task = TASKS[task_name]()
    tag = f"{kind}_{task_name}_s{seed}" + "".join(
        f"_{k}{v}" for k, v in sorted(kw.items()))
    path = CKPT / f"{tag}.pt"
    is_tf = kind.startswith("tf-")
    if kind in ("base", "feven-base"):
        model = M.GRULM(task.m, 64).to(M.DEVICE)
    elif kind in ("distill", "prune"):
        model = M.GRULM(task.m, kw.get("width", 64)).to(M.DEVICE)
    elif is_tf:
        model = M.TransformerLM(task.m).to(M.DEVICE)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=M.DEVICE,
                                         weights_only=True))
        model.eval()
        return model, task
    print(f"[train] {tag}")
    if kind in ("base", "feven-base"):
        model = M.train_base(task, seed=seed)
    elif kind == "distill":
        teacher, _ = get_model("base", task_name, seed)
        model = M.distill(teacher, task, width=kw.get("width", 64), seed=seed)
    elif kind == "prune":
        base, _ = get_model("base", task_name, seed)
        model = M.prune_finetune(base, task, frac=kw.get("frac", 0.5), seed=seed)
    elif kind == "tf-base":
        model = M.train_base_transformer(task, seed=seed)
    elif kind == "tf-distill":
        teacher, _ = get_model("tf-base", task_name, seed)
        model = M.distill_transformer(teacher, task, seed=seed)
    torch.save(model.state_dict(), path)
    print(f"  val CE = {M.val_ce_bits(model, task):.4f} bits "
          f"(task h = {task.entropy_rate():.4f})")
    return model, task


# ------------------------------------------------------------- runs and states

def gen_runs(model, task, sigma, seed_base, arch="gru", gen_kw=None):
    """Two independent free-running generation runs -> dict with symbol chains and
    belief chains per run."""
    gen_kw = gen_kw or (GEN_KW if arch == "gru" else GEN_KW_TF)
    runs = []
    for r in (0, 1):
        if arch == "gru":
            syms, beliefs = M.generate(model, sigma=sigma,
                                       seed=seed_base + 71 * r, **gen_kw)
        else:
            syms, beliefs = M.generate_transformer(model, task, sigma=sigma,
                                                   seed=seed_base + 71 * r,
                                                   **gen_kw)
        runs.append({"sym": emitted_readout(syms),
                     "belief": belief_readout(beliefs) if model.m == 2 else None})
    return runs


def eval_inputs(task, seed=987654):
    return torch.from_numpy(
        task.sample(EVAL_B, EVAL_L, seed=seed).astype(np.int64)).to(M.DEVICE)


def get_states(model, task, sigma, arch="gru", seed=555):
    inputs = eval_inputs(task)
    if arch == "gru":
        _, states, _ = M.collect_states(model, inputs, sigma=sigma, seed=seed)
    else:
        _, states, _ = M.collect_states_transformer(model, inputs, sigma=sigma,
                                                    seed=seed)
    return states.cpu().numpy()


# ------------------------------------------------------------- pair evaluation

def stream_diag(sym, m):
    h, n_used, _ = entropy_rate(sym, m)
    wall, _ = estimation_wall(sym, m)
    return {"h_cond": float(h), "h_n_used": n_used,
            "h_lz78": float(lz78_entropy(sym, m)), "k_wall": float(wall)}


def eval_pair(runs_a, runs_b, states_a, states_b, m, ns=NS_FULL, repeats=4,
              seed=0, with_belief=True, with_dsa=True):
    """The three metrics side-by-side for one pair, plus diagnostics."""
    out = {}
    sym_a, sym_b = runs_a[0]["sym"], runs_b[0]["sym"]
    sym_a2, sym_b2 = runs_a[1]["sym"], runs_b[1]["sym"]

    out["tv_unigram"] = unigram_tv(sym_a, sym_b, m)
    da, db = stream_diag(sym_a, m), stream_diag(sym_b, m)
    out["diag_a"], out["diag_b"] = da, db
    out["k_wall"] = float(min(da["k_wall"], db["k_wall"]))
    out["fano_lb"] = fano_lower_bound(da["h_cond"] - db["h_cond"], m)

    rows = dbar_pair_curve(sym_a, sym_b, sym_a2, sym_b2, m, ns=ns,
                           repeats=repeats, seed=seed)
    out["dbar_curve"] = rows
    win = plateau(rows, k_wall=out["k_wall"])
    out["plateau"] = win
    out["dbar_1"] = next((r["dbar"] for r in rows if r["n"] == 1), None)

    if with_belief and runs_a[0]["belief"] is not None:
        brows = dbar_pair_curve(runs_a[0]["belief"], runs_b[0]["belief"],
                                runs_a[1]["belief"], runs_b[1]["belief"], 4,
                                ns=tuple(n for n in ns if n <= 16),
                                repeats=repeats, seed=seed + 3)
        out["belief_curve"] = brows
        bwall = min(estimation_wall(runs_a[0]["belief"], 4)[0],
                    estimation_wall(runs_b[0]["belief"], 4)[0])
        out["belief_plateau"] = plateau(brows, k_wall=bwall)
        out["belief_k_wall"] = float(bwall)

    out["cka"] = cka_states(states_a, states_b, skip=SKIP)
    if with_dsa:
        out["dsa"] = dsa_distance(states_a, states_b, device=M.DEVICE, skip=SKIP)
    return out


# ------------------------------------------------------------------ gate logic

def gate_report(pair_results, nu_dsa, kappa_dsa):
    """PLAN.md E0 gate for one pair type across seeds. pair_results: list of
    eval_pair dicts (one per seed)."""
    cka = np.array([r["cka"] for r in pair_results])
    dsa = np.array([r["dsa"] for r in pair_results])
    delta = np.array([r["plateau"]["delta"] for r in pair_results])
    dbar_star = np.array([r["plateau"]["dbar"] for r in pair_results])
    floor_star = np.array([r["plateau"]["floor"] for r in pair_results])
    tv1 = np.array([r["dbar_1"] for r in pair_results])
    n_star = [r["plateau"]["n"] for r in pair_results]
    wall = np.array([r["k_wall"] for r in pair_results])

    cka_similar = bool(cka.mean() >= 0.90)
    dsa_similar = bool(dsa.mean() <= nu_dsa + 0.25 * (kappa_dsa - nu_dsa))
    sep = bool(delta.mean() - 2 * delta.std() > 0)
    twice_floor = bool(dbar_star.mean() >= 2 * floor_star.mean())
    within_wall = bool(all(n <= w for n, w in zip(n_star, wall)))
    marginal_ok = bool(tv1.mean() <= 0.02)
    dbar_separates = sep and twice_floor and within_wall and marginal_ok
    return {
        "cka_mean": float(cka.mean()), "cka_similar": cka_similar,
        "dsa_mean": float(dsa.mean()), "dsa_similar": dsa_similar,
        "delta_mean": float(delta.mean()), "delta_std": float(delta.std()),
        "dbar_star_mean": float(dbar_star.mean()),
        "floor_star_mean": float(floor_star.mean()),
        "dbar1_mean": float(tv1.mean()), "n_star": n_star,
        "k_wall_mean": float(wall.mean()),
        "dbar_separates": dbar_separates,
        "PASS": bool(cka_similar and dsa_similar and dbar_separates),
        "checks": {"sep_2sigma": sep, "twice_floor": twice_floor,
                   "within_wall": within_wall, "marginal_ok": marginal_ok},
    }


# -------------------------------------------------------------- sigma calibration

def calibrate_sigma(model, task, seed, grid=SIGMA_GRID, tv_tol=0.01,
                    ce_rel_tol=0.02, arch="gru"):
    """PLAN.md: largest sigma with free-running unigram TV <= 0.01 and
    teacher-forced CE increase <= 2% relative. Short calibration runs."""
    kw = dict(B=64, T=4096, burn=256)
    if arch == "gru":
        base_syms, _ = M.generate(model, sigma=0.0, seed=seed + 5000,
                                  record_belief=False, **kw)
    else:
        base_syms, _ = M.generate_transformer(model, task, sigma=0.0,
                                              seed=seed + 5000,
                                              record_belief=False, **kw)
    ce0 = M.val_ce_bits(model, task)
    rows = []
    best = None
    for sig in grid:
        if arch == "gru":
            syms, _ = M.generate(model, sigma=sig, seed=seed + 6000,
                                 record_belief=False, **kw)
            ce = val_ce_noisy(model, task, sig)
        else:
            syms, _ = M.generate_transformer(model, task, sigma=sig,
                                             seed=seed + 6000,
                                             record_belief=False, **kw)
            ce = val_ce_noisy_tf(model, task, sig)
        tv = unigram_tv(emitted_readout(base_syms), emitted_readout(syms), task.m)
        ok = tv <= tv_tol and ce <= ce0 * (1 + ce_rel_tol)
        rows.append({"sigma": sig, "tv": tv, "ce": ce, "ok": bool(ok)})
        if ok:
            best = sig
    return best, {"ce0": ce0, "grid": rows}


def val_ce_noisy(model, task, sigma, seed=123456):
    return M.val_ce_bits(model, task, sigma=sigma, seed=seed)


def val_ce_noisy_tf(model, task, sigma, n_seq=64, seed=123456, skip=SKIP):
    import torch.nn.functional as F
    x = torch.from_numpy(task.sample(n_seq, 256, seed=seed).astype(np.int64)).to(M.DEVICE)
    with torch.no_grad():
        logits, _, _ = M.collect_states_transformer(model, x[:, :-1], sigma=sigma,
                                                    seed=seed)
        ce = F.cross_entropy(logits[:, skip:].reshape(-1, model.m),
                             x[:, skip + 1:].reshape(-1))
    return float(ce) / np.log(2.0)
