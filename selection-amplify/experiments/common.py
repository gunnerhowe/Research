"""Shared configuration and stack builder for the experiment ladder. The beta
grid, N, seeds, and the operating-point RULES are pre-registered in PLAN.md; the
one fixed operating point (gamma and the gate quantiles) is calibrated once by
calibrate.py BEFORE the gated seeds and recorded in PLAN.md's Deviations."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

from selamp import data                                    # noqa: E402
from selamp.bridge import RewardConfig                     # noqa: E402
from selamp.diffusion import Diffusion                     # noqa: E402
from selamp.selection import SelectionEstimator            # noqa: E402
from selamp.validate import GateKDE, IndependentValidator  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- pre-registered grid / sizes / seeds (PLAN.md) ----------------------------
BETAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
HEADLINE_SEEDS = [0, 1, 2, 3, 4]           # 5 seeds for the go/no-go curve
N_POP = 8000
N_REF = 8000
N_TEST = 8000
N_SYNTH_PER_CLASS = 1000                   # augmentation budget (matched across gens)
PRIMARY = "two_moons"

# ---- fixed operating point (calibrated once by calibrate.py; see PLAN.md) -----
# gamma is the guidance scale; the gates are pre-registered as QUANTILE RULES
# that adapt per fit, so no per-run tuning happens inside the gated sweep.
OP = dict(
    gamma=6.0,
    tau_q=0.10,          # soft density gate = 10th pct of log p_ref
    veto_q=0.02,         # hard veto      = 2nd pct of log p_ref
    u_q=0.90,            # uncertainty gate = 90th pct of epistemic unc over D_obs
    dmax_factor=6.0,     # proximity gate  = factor x median obs self-NN distance
    sigma_guide_max=0.6,
    cap_ratio=1.0,
    kde_bandwidth=0.30,
)

SELECTOR_KW = dict(n_members=5, hidden=64, depth=2, epochs=300)
DIFFUSION_KW = dict(T=200, hidden=128, depth=3)
DIFFUSION_FIT_KW = dict(epochs=2500, batch=256, lr=2e-3)


def build_reward_config(sel: SelectionEstimator, gate: GateKDE, X_ref, X_obs,
                        op=OP) -> RewardConfig:
    """Turn the pre-registered quantile RULES into a concrete RewardConfig for
    one fit (the operating point adapts by rule, not by tuning)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(X_obs)
    d, _ = tree.query(X_obs, k=2)
    med_nn = float(np.median(d[:, 1]))
    return RewardConfig(
        gamma=op["gamma"],
        tau_log=gate.quantile_threshold(X_ref, op["tau_q"]),
        veto_log=gate.quantile_threshold(X_ref, op["veto_q"]),
        u_max=sel.uncertainty_quantile(X_obs, op["u_q"]),
        d_max=op["dmax_factor"] * med_nn,
        sigma_guide_max=op["sigma_guide_max"],
        cap_ratio=op["cap_ratio"],
    )


class Stack:
    """The full per-(testbed,beta,seed) stack: corpora + selector + base
    diffusion + gate + reward config + independent validator."""

    def __init__(self, testbed, beta, seed, op=OP, verbose=False):
        self.testbed, self.beta, self.seed = testbed, beta, seed
        self.c = data.make_corpora(testbed, beta, seed, N_POP, N_REF, N_TEST)
        self.sel = SelectionEstimator(**SELECTOR_KW).fit(
            self.c.X_obs, self.c.X_ref, self.c.obs_frac, seed=seed)
        self.dm = Diffusion(**DIFFUSION_KW).fit(
            self.c.X_obs, self.c.y_obs, seed=seed, **DIFFUSION_FIT_KW)
        self.gate = GateKDE(self.c.X_ref, bandwidth=op["kde_bandwidth"], seed=seed)
        self.cfg = build_reward_config(self.sel, self.gate, self.c.X_ref,
                                       self.c.X_obs, op)
        # independent validator fit on a FRESH full-population pool (K3)
        Xval, _ = data.TESTBEDS[testbed](N_REF, seed=90000 + seed)
        self.val = IndependentValidator(Xval)
        if verbose:
            print(f"  [{testbed} b={beta} s={seed}] n_obs={len(self.c.X_obs)} "
                  f"tau={self.cfg.tau_log:.2f} veto={self.cfg.veto_log:.2f} "
                  f"u_max={self.cfg.u_max:.3f} d_max={self.cfg.d_max:.3f}")


def save(name, obj):
    path = RESULTS / name
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    print(f"[saved] {path}  ({path.stat().st_size / 1024:.1f} KB)")


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "__dict__"):
        return o.__dict__
    raise TypeError(type(o))


def stamp():
    return {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "device": DEVICE, "numpy": np.__version__,
            "op": OP, "betas": BETAS, "seeds": HEADLINE_SEEDS,
            "n_pop": N_POP, "n_synth_per_class": N_SYNTH_PER_CLASS}


def reward_cfg_dict(cfg):
    return asdict(cfg)
