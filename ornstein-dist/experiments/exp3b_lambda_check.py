"""Protocol check for the Rosenstein lambda1 baseline: compare fit windows.

The fixed window (1,15) saturates for fast systems (speed x2 diverges to attractor
scale within the window), compressing slopes. Check (1,8) and (2,10) on fresh
realizations; report truth value vs literature (0.906 nats/t) and the speed2/truth
ratio (should be 2 by construction).
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ornstein.baselines import rosenstein_lambda1
from ornstein.systems import lorenz_trajectory

t0 = time.time()
for fit in ((1, 15), (1, 8), (2, 10), (1, 5)):
    vals = {"truth": [], "speed2": [], "iaaft_r2": []}
    for s in range(4):
        tr = lorenz_trajectory(200_000, tau=0.1, seed=1000 + s)
        sp = lorenz_trajectory(200_000, tau=0.1, seed=3000 + s, speed=2.0)
        lt, r2t = rosenstein_lambda1(tr[:, 0], dt=0.1, fit_range=fit, seed=s)
        ls, r2s = rosenstein_lambda1(sp[:, 0], dt=0.1, fit_range=fit, seed=s)
        vals["truth"].append(lt)
        vals["speed2"].append(ls)
    mt, ms = np.mean(vals["truth"]), np.mean(vals["speed2"])
    print(f"fit={fit}: truth λ1={mt:.3f}±{np.std(vals['truth']):.3f} "
          f"(lit 0.906)  speed2={ms:.3f}  ratio={ms/mt:.2f} (target 2.0)")
print(f"({time.time()-t0:.0f}s)")
