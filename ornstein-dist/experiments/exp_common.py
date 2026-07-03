"""Shared dataset construction so all experiments use identical systems/surrogates.

Surrogate zoo (docs/RESEARCH_NOTES.md §7):
- truth2   : independent Lorenz run          -> negative control (all metrics ~ 0)
- iaaft    : IAAFT of truth x(t)             -> exact marginal + approx spectrum, dynamics scrambled
- speed2   : dx/dt = 2 f(x)                  -> EXACT same invariant measure, different clock
- reversed : time-reversed truth             -> same measure, same spectrum, same entropy; only
                                                direct d̄ can possibly see it (hard mode)
- rho32    : Lorenz rho=32                   -> positive control (different attractor)
"""
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ornstein.surrogates import iaaft, time_reverse  # noqa: E402
from ornstein.systems import lorenz_trajectory  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def make_datasets(n_samples=1_000_000, tau=0.1, dt=0.005):
    """Returns dict name -> {'X': (N,3) array or None, 'x': scalar series}."""
    truth = lorenz_trajectory(n_samples, tau=tau, dt=dt, seed=10)
    truth2 = lorenz_trajectory(n_samples, tau=tau, dt=dt, seed=20)
    speed2 = lorenz_trajectory(n_samples, tau=tau, dt=dt, seed=40, speed=2.0)
    rho32 = lorenz_trajectory(n_samples, tau=tau, dt=dt, seed=50, rho=32.0)
    rev = time_reverse(truth2)  # reverse an independent run, not truth itself
    ia = iaaft(truth2[:, 0], n_iter=200, seed=30)
    return {
        "truth": {"X": truth, "x": truth[:, 0]},
        "truth2": {"X": truth2, "x": truth2[:, 0]},
        "iaaft": {"X": None, "x": ia},
        "speed2": {"X": speed2, "x": speed2[:, 0]},
        "reversed": {"X": rev, "x": rev[:, 0]},
        "rho32": {"X": rho32, "x": rho32[:, 0]},
    }


SURROGATE_ORDER = ["truth2", "iaaft", "speed2", "reversed", "rho32"]
LABELS = {
    "truth2": "independent truth (neg ctrl)",
    "iaaft": "IAAFT (marginal+spectrum matched)",
    "speed2": "speed x2 (same invariant measure)",
    "reversed": "time-reversed (same measure+spectrum+entropy)",
    "rho32": "rho=32 (pos ctrl, different attractor)",
}
