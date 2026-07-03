"""Validation of the d̄ estimator and entropy machinery on analytically solvable cases.

Targets (derivations in docs/RESEARCH_NOTES.md §5 plus):
1. iid Bern(p) vs iid Bern(q): d̄_n = |p-q| exactly, for every n.
2. Same process, independent runs: estimate ≈ noise floor.
3. iid(1/2) vs period-2 alternating: d̄_1 = 0 (identical marginals!) and
   d̄_n = E[min(k, n-k)]/n with k ~ Bin(n, 1/2) exactly (nearest-target transport is
   optimal by the complement-pairing argument), rising to d̄ = 1/2. The Fano bound
   independently forces d̄ ≥ H_b⁻¹(1 bit) = 1/2, so the two pipelines must agree.
4. Entropy: iid(1/2) → 1 bit; alternating → 0; Markov(flip=0.1) → H_b(0.1) ≈ 0.4690.
5. Cross-consistency: estimated d̄ plateau ≥ Fano LB from estimated entropy gap.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ornstein.dbar import dbar_curve, dbar_n, encode_blocks, hamming_matrix
from ornstein.entropy import (binary_entropy, entropy_rate, fano_lower_bound,
                              lz78_entropy)

RNG = np.random.default_rng(42)
N = 1_000_000
results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# --- unit checks on encoding / Hamming ---------------------------------------
# np.convolve reverses the kernel -> big-endian codes: block s[i..i+n) -> sum s[i+j]*m^(n-1-j).
# Any fixed bijection works: unique/counts don't care, and Hamming (XOR-popcount or
# digit-wise after the same decode on both sides) is invariant to digit order.
codes = encode_blocks(np.array([0, 1, 1, 0, 1], dtype=np.int8), 3, 2)
expect = np.array([0b011, 0b110, 0b101], dtype=np.uint64)
check("encode_blocks binary (big-endian)", np.array_equal(codes, expect), f"{codes} vs {expect}")

M = hamming_matrix(np.array([0b101], dtype=np.uint64), np.array([0b011], dtype=np.uint64), 3, 2)
check("hamming binary", np.isclose(M[0, 0], 2 / 3), f"{M[0,0]:.4f} vs 0.6667")

c4 = encode_blocks(np.array([3, 0, 2, 1], dtype=np.int8), 2, 4)
M4 = hamming_matrix(c4[:1], c4[1:2], 2, 4)  # blocks (3,0) vs (0,2): mismatch both
check("hamming base-4", np.isclose(M4[0, 0], 1.0), f"{M4[0,0]:.4f} vs 1.0")

# --- case 1: iid Bernoulli(0.3) vs (0.5), d̄_n = 0.2 at all n ------------------
a = (RNG.random(N) < 0.3).astype(np.int8)
b = (RNG.random(N) < 0.5).astype(np.int8)
rows = dbar_curve(a, b, 2, ns=(1, 2, 4, 8, 16, 32), n_blocks=4000, repeats=3, seed=1)
print("   iid p/q curve:", [(r["n"], round(r["dbar"], 4), round(r["floor"], 4), r["regime"])
                            for r in rows])
# full regime must nail the exact value; sampled regime is judged against its own
# noise floor (finite-sample OT bias — the thing the floor exists to quantify).
err_full = max((abs(r["dbar"] - 0.2) for r in rows if r["regime"] == "full"), default=0)
err_samp = max((abs(r["dbar"] - 0.2) - r["floor"] for r in rows if r["regime"] == "sampled"),
               default=0)
check("iid Bern(.3) vs Bern(.5) = 0.2 (full regime exact)", err_full < 0.005,
      f"max |err| full = {err_full:.4f}")
check("iid Bern(.3) vs Bern(.5) = 0.2 (sampled within floor)", err_samp < 0.005,
      f"max (|err| - floor) sampled = {err_samp:.4f}")

# same pair but force the sampled regime at small n to validate the subsampler
v_s, _ = dbar_n(a, b, 4, 2, n_blocks=2000, seed=3, force_regime="sampled")
check("sampled regime unbiased (n=4)", abs(v_s - 0.2) < 0.03, f"{v_s:.4f} vs 0.2")

# --- case 2: same process independent runs → floor ----------------------------
c = (RNG.random(N) < 0.3).astype(np.int8)
rows2 = dbar_curve(a, c, 2, ns=(1, 4, 16, 32), n_blocks=4000, repeats=3, seed=2)
sep = [r["dbar"] - r["floor"] for r in rows2]
print("   same-process curve:", [(r["n"], round(r["dbar"], 4), round(r["floor"], 4)) for r in rows2])
check("same process ≈ floor", max(sep) < 0.01, f"max (est - floor) = {max(sep):.4f}")

# --- case 3: iid(1/2) vs alternating; exact analytic d̄_n ----------------------
u = (RNG.random(N) < 0.5).astype(np.int8)
alt = (np.arange(N) % 2).astype(np.int8)
rows3 = dbar_curve(u, alt, 2, ns=(1, 2, 4, 8, 16, 32), n_blocks=4000, repeats=3, seed=3)


def alt_exact(n):
    k = np.arange(n + 1)
    return float(np.sum(binom.pmf(k, n, 0.5) * np.minimum(k, n - k)) / n)


print("   alternating curve:", [(r["n"], round(r["dbar"], 4), round(alt_exact(r["n"]), 4),
                                  r["regime"]) for r in rows3])
errs3 = [abs(r["dbar"] - alt_exact(r["n"])) - (r["floor"] if r["regime"] == "sampled" else 0)
         for r in rows3]
check("iid(1/2) vs alternating matches exact d̄_n", max(errs3) < 0.01,
      f"max (|err| - floor) = {max(errs3):.4f}")
check("alternating: d̄_1 = 0 (static-blind) but d̄_32 large (dynamics-visible)",
      rows3[0]["dbar"] < 0.01 and rows3[-1]["dbar"] > 0.35,
      f"d̄_1 = {rows3[0]['dbar']:.4f}, d̄_32 = {rows3[-1]['dbar']:.4f}")

# --- case 4: entropy estimators ------------------------------------------------
h_u, n_used, _ = entropy_rate(u, 2)
check("entropy iid(1/2) ≈ 1 bit", abs(h_u - 1.0) < 0.02, f"{h_u:.4f} (n={n_used})")
h_alt, _, _ = entropy_rate(alt, 2)
check("entropy alternating ≈ 0", h_alt < 0.01, f"{h_alt:.5f}")

flips = (RNG.random(N) < 0.1).astype(np.int8)
mk = np.bitwise_xor.accumulate(flips)  # Markov chain, flip prob 0.1
h_mk, n_mk, _ = entropy_rate(mk, 2)
h_true = float(binary_entropy(0.1))
check("entropy Markov(0.1) ≈ H_b(0.1)", abs(h_mk - h_true) < 0.02,
      f"{h_mk:.4f} vs {h_true:.4f} (n={n_mk})")
lz = lz78_entropy(mk[:200_000], 2)
check("LZ78 in the right ballpark (Markov 0.1)", 0.3 < lz < 0.75, f"{lz:.4f}")

# --- case 5: Fano bound + cross-consistency -----------------------------------
check("Fano LB inversion: |Δh|=1 bit, m=2 → 0.5",
      abs(fano_lower_bound(1.0, 2) - 0.5) < 1e-6, f"{fano_lower_bound(1.0, 2):.6f}")

fano_lb = fano_lower_bound(h_u - h_mk, 2)
rows5 = dbar_curve(u, mk, 2, ns=(1, 4, 16, 32), n_blocks=4000, repeats=3, seed=5)
plateau = max(r["dbar"] for r in rows5)
print(f"   iid(1/2) vs Markov(0.1): Fano LB = {fano_lb:.4f}, d̄ plateau = {plateau:.4f}")
check("d̄ estimate ≥ Fano lower bound (consistency of the two pipelines)",
      plateau >= fano_lb - 0.02, f"{plateau:.4f} ≥ {fano_lb:.4f}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
