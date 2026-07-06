"""Per-synapse consolidation rules as Euler-Maruyama updates on the weights.

Framing (see paper Sec. 2). Each weight w_i follows, during a consolidation
task, the stochastic differential equation

    dw_i = [ -grad_i L_task            (task learning; shared by every method)
             - s_i (w_i - mu_i)        (anchored consolidation drift)
             + sigma_i^2 * score_i(w) ] dt   (Doob barrier steering; OURS ONLY)
           + sigma_i dW_i              (intrinsic noise; identical across methods)

The anchored drift  -s_i (w_i - mu_i)  is the SURRENDERED term: it is the small-
noise / OU limit shared by OUA (Garcia Fernandez et al. 2024), MESU (Brusca et
al. 2025) and EWC (Kirkpatrick et al. 2017). We claim no novelty for it.

The novel piece is the Doob-h-transform steering  sigma_i^2 * d/dw log h_i(w) ,
the extra drift a diffusion acquires when CONDITIONED on the event of never
crossing a memory-critical barrier. For a weight conditioned to remain in the
iso-loss interval (mu_i - b_i, mu_i + b_i), the ground-state (quasi-stationary)
h-transform has

    h_i(w) = cos( pi (w - mu_i) / (2 b_i) )   on the interval, 0 outside,
    d/dw log h_i = -(pi / 2 b_i) tan( pi (w - mu_i) / (2 b_i) ),

a restoring force that DIVERGES at the barrier and is amplified by sigma^2. The
barrier half-width is the iso-loss-increase radius b_i = sqrt(2 c / (s_i + eps)):
important (high-Fisher) synapses get a tight barrier, unimportant ones a barrier
at infinity (no constraint). The SAME global intrinsic noise sigma is therefore
STEERED per synapse by the memory geometry -- the mechanism of the paper.

Key design choice for a fair falsifier (GATE F): every method receives the
IDENTICAL injected noise sigma dW; the methods differ ONLY in their drift. So a
non-monotone retention-vs-sigma curve for OURS that the matched anchored-drift
baselines do NOT produce isolates the barrier conditioning, not "noise helps".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Fisher importance and anchor snapshot                                       #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _zeros_like_params(model):
    return [torch.zeros_like(p) for p in model.parameters()]


def diagonal_fisher(model, loss_fn, loader, device, n_batches=None):
    """Empirical diagonal Fisher F_i = E[(d log p / d w_i)^2] estimated from the
    task's own data. We use the model-sampled label form (true Fisher): sample a
    label from the model's predictive distribution and backprop its log-prob, so
    F is PSD and does not depend on the (possibly wrong) task labels. Returns a
    list of tensors shaped like model.parameters()."""
    fisher = _zeros_like_params(model)
    model.eval()
    n = 0
    for bi, (x, _y) in enumerate(loader):
        if n_batches is not None and bi >= n_batches:
            break
        x = x.to(device)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        if not torch.isfinite(logits).all():
            continue                              # skip an unstable batch
        logp = torch.log_softmax(logits, dim=1)
        # sample labels from the model's predictive distribution (true Fisher)
        with torch.no_grad():
            probs = torch.nan_to_num(logp.exp(), nan=0.0)
            probs = probs.clamp_min(0.0)
            if (probs.sum(1) <= 0).any():
                continue
            sampled = torch.multinomial(probs, 1).squeeze(1)
        idx = torch.arange(x.shape[0], device=x.device)
        picked = logp[idx, sampled]
        picked.sum().backward()
        for f, p in zip(fisher, model.parameters()):
            if p.grad is not None:
                f += p.grad.detach() ** 2
        n += x.shape[0]
    model.zero_grad(set_to_none=True)
    scale = max(n, 1)
    return [f / scale for f in fisher]


@dataclass
class Memory:
    """Consolidation state carried between tasks: anchor mu, accumulated
    importance s (online-EWC running sum of diagonal Fisher), and per-weight
    barrier half-width b derived from s."""
    mu: list = field(default_factory=list)          # anchor (list of tensors)
    s: list = field(default_factory=list)           # importance (Fisher sum)
    b: list = field(default_factory=list)           # barrier half-width
    has_anchor: bool = False


@torch.no_grad()
def update_memory(mem: Memory, model, fisher, *, decay=1.0, barrier_scale=0.3,
                  bmin_frac=0.05, eps=1e-12):
    """Fold a finished task into the consolidation state. Anchor <- current
    weights; importance <- decay * importance + fisher (online EWC).

    Barrier half-width (softened iso-loss form, robust to Fisher units):

        b_i = barrier_scale / sqrt( 1 + s_i / s_ref ),   s_ref = median(s),

    so an UNIMPORTANT synapse (s_i << s_ref) gets b_i ~ barrier_scale (a loose
    barrier ~ the weight scale -> free to adapt), while an IMPORTANT synapse
    (s_i >> s_ref) gets b_i ~ barrier_scale * sqrt(s_ref / s_i) (tight, ~1/sqrt(s)
    like the true iso-loss radius). `barrier_scale` is the single length knob
    (calibrated once; see PLAN.md), decoupled from the arbitrary scale of the
    Fisher. b is clamped to [bmin_frac * barrier_scale, barrier_scale] so no
    synapse gets a singular barrier."""
    params = list(model.parameters())
    if not mem.has_anchor:
        mem.mu = [p.detach().clone() for p in params]
        mem.s = [f.detach().clone() for f in fisher]
    else:
        mem.mu = [p.detach().clone() for p in params]
        mem.s = [decay * si + f.detach() for si, f in zip(mem.s, fisher)]
    alls = torch.cat([si.reshape(-1) for si in mem.s])
    # reference importance scale: median (robust for heavy-tailed Fisher); fall
    # back to the mean, then to 1, if the median is degenerate (~all-zero Fisher).
    s_ref = torch.median(alls)
    if s_ref <= 0:
        s_ref = alls.mean()
    if s_ref <= 0:
        s_ref = torch.tensor(1.0, device=alls.device)
    s_ref = s_ref + eps
    lo, hi = bmin_frac * barrier_scale, barrier_scale
    mem.b = [torch.clamp(barrier_scale / torch.sqrt(1.0 + si / s_ref), lo, hi)
             for si in mem.s]
    mem.has_anchor = True
    return mem


# --------------------------------------------------------------------------- #
#  Consolidation operators (applied AFTER the shared task-gradient step)       #
# --------------------------------------------------------------------------- #

@dataclass
class ConsolConfig:
    method: str = "doob"        # doob | ou | ewc | mesu | none
    sigma: float = 0.0          # intrinsic-noise amplitude (the swept variable)
    lr_c: float = 0.1           # consolidation step size (dt of the EM scheme)
    anchor_strength: float = 1.0  # multiplies s (drift). EWC uses a larger value.
    kappa: float = 1.0          # Doob-steering strength (1=full h-transform; 0=ablated)
    noise_on_all: bool = True   # inject noise on all weights (global device noise)
    fisher_noise_floor: float = 0.0  # MESU: minimum precision
    max_step_frac: float = 0.25  # cap Doob per-step move to this fraction of b
                                 # (finite restoring force: real analog steering
                                 #  has finite bandwidth; also stabilises Euler)


class Consolidator:
    """Wraps the consolidation state + config and applies the per-step SDE update
    in place on the model parameters. The task-gradient step is applied
    separately by a shared torch optimizer, so the ONLY thing that differs
    between methods is what this object adds."""

    def __init__(self, mem: Memory, cfg: ConsolConfig, device, seed=0,
                 noise_model=None, quantize=False):
        self.mem = mem
        self.cfg = cfg
        self.device = device
        # dedicated RNG so the injected-noise stream is reproducible and identical
        # in distribution across methods at matched sigma.
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        # optional BSS-2 device-noise model (E2). None -> white Gaussian.
        self.noise_model = noise_model
        self.quantize = quantize
        # MESU keeps a per-weight variance; initialised lazily.
        self._mesu_var = None

    @torch.no_grad()
    def _add_noise(self, params):
        s = self.cfg.sigma
        if s <= 0:
            return
        rt = math.sqrt(self.cfg.lr_c)
        if self.noise_model is not None:
            # device-faithful BSS-2 noise (colored/multiplicative/fixed-pattern)
            incs = self.noise_model.sample(params, s)
            for p, inc in zip(params, incs):
                p.add_(rt * inc)
        else:
            for p in params:
                xi = torch.randn(p.shape, generator=self.gen, device=p.device, dtype=p.dtype)
                p.add_(s * rt * xi)
        if self.quantize and self.noise_model is not None:
            for p in params:
                p.copy_(self.noise_model.quantize(p))

    @torch.no_grad()
    def step(self):
        """Apply the method's consolidation drift (+ intrinsic noise) once."""
        if not self.mem.has_anchor or self.cfg.method == "none":
            # first task or no consolidation: optionally still inject device noise
            if self.cfg.method == "none":
                self._add_noise(list(self.model.parameters()))
            return

        m = self.cfg
        params = list(self.model.parameters())
        mu, s, b = self.mem.mu, self.mem.s, self.mem.b

        if m.method in ("ou", "ewc", "doob"):
            a = m.anchor_strength
            for p, mui, si, bi in zip(params, mu, s, b):
                # anchored consolidation drift (surrendered OUA/MESU/EWC limit)
                p.add_(m.lr_c * (-a * si * (p - mui)))
                if m.method == "doob" and m.sigma > 0 and m.kappa > 0:
                    # Doob barrier steering: sigma^2 * d/dw log h, h the ground
                    # state of the interval (mu-b, mu+b). Clamp the tan argument
                    # just inside +-pi/2 (the barrier) for the EM discretisation.
                    z = (p - mui) / bi
                    arg = torch.clamp(0.5 * math.pi * z, -0.5 * math.pi + 1e-4,
                                      0.5 * math.pi - 1e-4)
                    score = -(math.pi / (2.0 * bi)) * torch.tan(arg)
                    step = m.lr_c * m.kappa * (m.sigma ** 2) * score
                    # finite restoring force: cap the per-step move at a fraction
                    # of the barrier half-width (physical + stabilises Euler).
                    cap = m.max_step_frac * bi
                    p.add_(torch.clamp(step, -cap, cap))
            # intrinsic noise (identical injection across ou/ewc/doob)
            if m.noise_on_all:
                self._add_noise(params)
            return

        if m.method == "mesu":
            self._mesu_step(params, mu, s)
            return

        raise ValueError(f"unknown method {m.method}")

    # ----- MESU: variance-scaled Bayesian anchor (Brusca et al. 2025, Eq. 11) -- #
    @torch.no_grad()
    def _mesu_step(self, params, mu, s):
        """Metaplasticity-from-Synaptic-Uncertainty style update. Maintains a
        per-weight posterior precision lambda = 1/var; the anchor pull toward mu
        is scaled by the posterior variance and the number of past tasks. We add
        the SAME intrinsic noise sigma dW as the other methods so MESU sits on the
        identical noise axis. This is a faithful-in-spirit reimplementation (the
        original targets local Bayesian updates); it is a matched anchored-drift
        control, not our contribution."""
        m = self.cfg
        if self._mesu_var is None:
            # precision proportional to accumulated Fisher (uncertainty from data)
            self._mesu_var = [1.0 / (si + m.fisher_noise_floor + 1e-3) for si in s]
        for p, mui, si, var in zip(params, mu, s, self._mesu_var):
            # variance-scaled pull toward the consolidated mean
            drift = -(var * si) * (p - mui)
            p.add_(m.lr_c * drift)
        if m.noise_on_all:
            self._add_noise(params)

    def bind(self, model):
        self.model = model
        return self


# --------------------------------------------------------------------------- #
#  Benna-Fusi complex synapse (E3 baseline, boundary-free consolidation)       #
# --------------------------------------------------------------------------- #

class BennaFusiState:
    """Linear cascade of hidden variables (Benna & Fusi 2016). The visible
    weight is u[0]; hidden beakers u[1..L-1] have geometrically growing
    capacities C_k and couple diffusively with geometrically shrinking
    conductances g_k. Plasticity writes to u[0]; the chain integrates it over a
    spectrum of timescales, giving power-law forgetting WITHOUT an explicit
    anchor or barrier. Used only as an incumbent baseline."""

    def __init__(self, model, n_levels=4, base=4.0, g1=0.3, device="cpu"):
        self.params = list(model.parameters())
        self.L = n_levels
        # capacities C_k = base^k ; couplings g_{k,k+1} = g1 * base^{-k}
        self.C = [base ** k for k in range(n_levels)]
        self.g = [g1 * base ** (-k) for k in range(n_levels - 1)]
        # hidden beakers (u[0] mirrors the live weight, kept in sync by the trainer)
        self.u = [[torch.zeros_like(p) for _ in range(n_levels)] for p in self.params]
        for us, p in zip(self.u, self.params):
            us[0] = p.detach().clone()

    @torch.no_grad()
    def relax(self, steps=1, dt=1.0):
        """Diffuse the visible value into the hidden cascade (called each step)."""
        for us, p in zip(self.u, self.params):
            us[0] = p.detach().clone()          # visible beaker tracks the weight
            for _ in range(steps):
                flux = [torch.zeros_like(p) for _ in range(self.L)]
                for k in range(self.L - 1):
                    j = self.g[k] * (us[k] - us[k + 1])
                    flux[k] -= j
                    flux[k + 1] += j
                for k in range(1, self.L):       # k=0 is clamped to the weight
                    us[k] = us[k] + dt * flux[k] / self.C[k]
            # pull the visible weight toward the first hidden beaker (consolidation)
            p.add_(dt * self.g[0] * (us[1] - us[0]) / self.C[0])
