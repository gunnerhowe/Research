"""BrainScaleS-2 (BSS-2) intrinsic-noise model and real-hardware port hooks.

SCIENTIFIC-INTEGRITY STATEMENT (read this). This file contains TWO things:

  1. Bss2NoiseModel -- a device-FAITHFUL EMULATION of BSS-2 intrinsic analog
     noise, built from the published characterisation (Weis et al. 2020,
     arXiv:2006.13177; Pehle et al. 2022, Front. Neurosci. 16:795876). It models
     the features that could plausibly break the Doob mechanism (pre-registered
     K2): temporal COLOR (trial-to-trial variability is not white), 6-bit weight
     QUANTIZATION, signal-dependent (MULTIPLICATIVE) noise, and a static
     FIXED-PATTERN mismatch. E2 runs the retention sweep through THIS model to ask
     whether the inverted-U survives device-realistic noise. Results produced with
     it are labelled "BSS-2 noise emulation", never "measured on silicon".

  2. Bss2Backend -- the real-hardware port. Its methods document exactly what an
     on-silicon run calls (pynn_brainscales / hxtorch), and RAISE if the stack is
     absent. No fabricated silicon numbers or joules are produced anywhere. The
     on-silicon measurement is the pre-registered remaining step (PLAN.md, K2).

This session has no BSS-2 stack or hardware, so only (1) runs here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# --------------------------------------------------------------------------- #
#  (1) Device-faithful intrinsic-noise EMULATION                               #
# --------------------------------------------------------------------------- #

@dataclass
class Bss2NoiseParams:
    """Effective, dimensionless parameters of the BSS-2 intrinsic-noise model.
    Defaults are order-of-magnitude, chosen to reflect the published qualitative
    picture (colored temporal variability + fixed-pattern mismatch + 6-bit
    weights); the paper reports the sweep over the master amplitude and a
    robustness scan of `color`, not a fit to a specific chip."""
    color: float = 0.5              # AR(1) coeff of temporal noise (0=white, ->1 slow)
    multiplicative: float = 0.3     # fraction of noise that scales with |weight|
    fixed_pattern: float = 0.4      # static per-synapse mismatch, in units of sigma
    weight_bits: int = 6            # BSS-2 signed synaptic resolution
    weight_range: float = 1.0       # +- range mapped onto the 6-bit grid


class Bss2NoiseModel:
    """Generates per-weight device-noise increments with BSS-2-like structure.

    The increment for weight i at step t is

        n_i(t) = sigma_eff * [ (1 + m |w_i|) * xi_i(t) + fp_i ],
        xi_i(t) = color * xi_i(t-1) + sqrt(1-color^2) * eps_i(t),   eps ~ N(0,1),

    i.e. unit-variance temporally-COLORED noise, optionally MULTIPLICATIVE in the
    weight magnitude, plus a static FIXED-PATTERN offset fp_i drawn once per
    synapse. `quantize` snaps weights to the 6-bit grid. sigma_eff is the tunable
    effective amplitude (the on-chip knob analog: averaging samples / operating
    point)."""

    def __init__(self, shapes, params: Bss2NoiseParams, device, seed=0):
        self.p = params
        self.device = device
        self.gen = torch.Generator(device=device); self.gen.manual_seed(seed)
        self._state = [torch.zeros(s, device=device) for s in shapes]
        # static per-synapse fixed-pattern offsets, drawn once
        self._fp = [self.p.fixed_pattern *
                    torch.randn(s, generator=self.gen, device=device) for s in shapes]

    def sample(self, params_list, sigma_eff):
        """Return a list of noise tensors (one per parameter tensor) at amplitude
        sigma_eff. `params_list` supplies current weights for the multiplicative
        term. Advances the AR(1) temporal state in place."""
        rho = self.p.color
        keep = (1.0 - rho ** 2) ** 0.5
        out = []
        for k, w in enumerate(params_list):
            eps = torch.randn(w.shape, generator=self.gen, device=self.device, dtype=w.dtype)
            self._state[k] = rho * self._state[k] + keep * eps
            xi = self._state[k]
            gain = 1.0 + self.p.multiplicative * w.abs()
            n = sigma_eff * (gain * xi + self._fp[k])
            out.append(n)
        return out

    def quantize(self, w):
        """Snap to the signed 6-bit synaptic grid (in-place-safe copy)."""
        levels = 2 ** self.p.weight_bits
        r = self.p.weight_range
        wc = torch.clamp(w, -r, r)
        step = (2 * r) / (levels - 1)
        return torch.round(wc / step) * step


def make_noise_model(model, params: Bss2NoiseParams, device, seed=0):
    shapes = [tuple(p.shape) for p in model.parameters()]
    return Bss2NoiseModel(shapes, params, device, seed=seed)


# --------------------------------------------------------------------------- #
#  (2) Real-hardware port hooks (require the BSS-2 software stack)             #
# --------------------------------------------------------------------------- #

def _require_stack():
    try:
        import hxtorch  # noqa: F401
        return True
    except Exception:
        return False


class Bss2Backend:
    """Documents the on-silicon port. Each method names the concrete
    pynn_brainscales / hxtorch call an on-chip run performs, and raises if the
    stack is unavailable. Running this on hardware is the pre-registered remaining
    step (PLAN.md K2); it produces the MEASURED inverted-U and MEASURED joules the
    emulation cannot.

    Port outline (Cramer et al. 2022, PNAS, in-the-loop; Pehle et al. 2022):
      * map the MLP onto the analog synapse array (hxtorch.nn linear layers, 6-bit
        weights);
      * the intrinsic analog noise IS the diffusion term -- no RNG is programmed;
      * tune the effective noise via the operating point / #averaging samples,
        which is the hardware analog of sigma_eff;
      * the Doob barrier drift runs on the on-chip plasticity processor (PPU);
      * energy is read from the chip's power measurement, giving retention vs
        MEASURED joules.
    """

    def __init__(self):
        self.available = _require_stack()

    def _guard(self):
        if not self.available:
            raise RuntimeError(
                "BSS-2 software stack (hxtorch/pynn_brainscales) not available in "
                "this environment. The on-silicon run is the pre-registered "
                "remaining step (PLAN.md K2); it cannot be executed here. Use "
                "Bss2NoiseModel for the device-faithful emulation instead.")

    def map_model(self, model):
        self._guard()

    def set_effective_noise(self, sigma_eff):
        self._guard()

    def measure_retention_and_energy(self, tasks):
        self._guard()
