"""Generative next-step surrogate for ring-topology chaotic states.

Conditional flow matching (rectified-flow / diffusion-family) over the
normalized state RESIDUAL g = (xn_{t+1} - xn_t) / sigma_d, conditioned on the
current normalized state xn_t. Few-step deterministic Euler sampler; the only
stochasticity is the reparameterized initial noise z, so rollouts are fully
differentiable. Backbone: 1D circular CNN with FiLM time conditioning.

Also provides the deterministic AR baseline (same backbone, direct residual
regression) and a rollout helper with per-step gradient checkpointing and a
detach interval (truncated BPTT through chaotic dynamics).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def timestep_embedding(tau: torch.Tensor, dim: int) -> torch.Tensor:
    """tau: (B,) in [0,1] -> (B, dim) sinusoidal features."""
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=tau.device) / half)
    ang = tau[:, None] * freqs[None, :] * 1000.0
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class CircConv(nn.Conv1d):
    def forward(self, x):
        p = self.kernel_size[0] // 2
        x = F.pad(x, (p, p), mode="circular")
        return F.conv1d(x, self.weight, self.bias, self.stride, 0, self.dilation)


class ResBlock(nn.Module):
    def __init__(self, ch: int, temb_dim: int, kernel: int = 5):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = CircConv(ch, ch, kernel)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = CircConv(ch, ch, kernel)
        self.film = nn.Linear(temb_dim, 2 * ch)

    def forward(self, x, temb):
        scale, shift = self.film(temb)[:, :, None].chunk(2, dim=1)
        h = self.conv1(F.silu(self.norm1(x)))
        h = h * (1 + scale) + shift
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class Backbone(nn.Module):
    def __init__(self, in_ch: int, width: int = 96, depth: int = 4,
                 temb_dim: int = 128, kernel: int = 5):
        super().__init__()
        self.temb_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim))
        self.temb_dim = temb_dim
        self.stem = CircConv(in_ch, width, kernel)
        self.blocks = nn.ModuleList(
            [ResBlock(width, temb_dim, kernel) for _ in range(depth)])
        self.head = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(),
                                  CircConv(width, 1, kernel))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, chans: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """chans: (B, in_ch, K); tau: (B,) in [0,1]. Returns (B, K)."""
        temb = self.temb_mlp(timestep_embedding(tau, self.temb_dim))
        h = self.stem(chans)
        for blk in self.blocks:
            h = blk(h, temb)
        return self.head(h).squeeze(1)


class MLPBackbone(nn.Module):
    """Weak-inductive-bias alternative: no translation equivariance."""

    def __init__(self, in_ch: int, K: int = 40, hidden: int = 512,
                 depth: int = 3, temb_dim: int = 128):
        super().__init__()
        self.temb_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim))
        self.temb_dim = temb_dim
        self.inp = nn.Linear(in_ch * K + temb_dim, hidden)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(depth - 1)])
        self.out = nn.Linear(hidden, K)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, chans: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        temb = self.temb_mlp(timestep_embedding(tau, self.temb_dim))
        h = torch.cat([chans.flatten(1), temb], dim=1)
        h = torch.nn.functional.silu(self.inp(h))
        for lay in self.hidden_layers:
            h = torch.nn.functional.silu(lay(h)) + h
        return self.out(h)


def make_backbone(arch: str, in_ch: int, width: int, depth: int):
    if arch == "mlp":
        return MLPBackbone(in_ch=in_ch)
    return Backbone(in_ch=in_ch, width=width, depth=depth)


class FlowSurrogate(nn.Module):
    """v-field model: v(g_tau, tau | xn_t). Interpolant g_tau = (1-tau) z + tau g."""

    def __init__(self, width: int = 96, depth: int = 4, arch: str = "cnn"):
        super().__init__()
        self.net = make_backbone(arch, 2, width, depth)
        # data normalization buffers, set once from the training split
        self.register_buffer("x_mean", torch.tensor(0.0))
        self.register_buffer("x_std", torch.tensor(1.0))
        self.register_buffer("sigma_d", torch.tensor(1.0))  # std of xn residual

    def set_norm(self, x_mean, x_std, sigma_d):
        self.x_mean.fill_(float(x_mean))
        self.x_std.fill_(float(x_std))
        self.sigma_d.fill_(float(sigma_d))

    def normalize(self, x):
        return (x - self.x_mean) / self.x_std

    def denormalize(self, xn):
        return xn * self.x_std + self.x_mean

    def v(self, g_tau, tau, cond_xn):
        return self.net(torch.stack([g_tau, cond_xn], dim=1), tau)

    def fm_loss(self, xn_t, xn_tp1):
        """Conditional flow-matching loss on one batch of transitions (B, K)."""
        g = (xn_tp1 - xn_t) / self.sigma_d
        z = torch.randn_like(g)
        tau = torch.rand(g.shape[0], device=g.device)
        g_tau = (1 - tau[:, None]) * z + tau[:, None] * g
        v_star = g - z
        v_hat = self.v(g_tau, tau, xn_t)
        return F.mse_loss(v_hat, v_star)

    def sample_step(self, xn_t, z, n_sampler_steps: int = 6):
        """One emulator step: xn_t (B, K), z (B, K) -> xn_{t+1} (B, K).
        Differentiable (deterministic given z)."""
        g = z
        S = n_sampler_steps
        for i in range(S):
            tau = torch.full((g.shape[0],), i / S, device=g.device)
            g = g + (1.0 / S) * self.v(g, tau, xn_t)
        return xn_t + self.sigma_d * g


class DetSurrogate(nn.Module):
    """Deterministic AR baseline: xn_t -> residual prediction (same backbone)."""

    def __init__(self, width: int = 96, depth: int = 4, arch: str = "cnn"):
        super().__init__()
        self.net = make_backbone(arch, 1, width, depth)
        self.register_buffer("x_mean", torch.tensor(0.0))
        self.register_buffer("x_std", torch.tensor(1.0))
        self.register_buffer("sigma_d", torch.tensor(1.0))

    def set_norm(self, x_mean, x_std, sigma_d):
        self.x_mean.fill_(float(x_mean))
        self.x_std.fill_(float(x_std))
        self.sigma_d.fill_(float(sigma_d))

    def normalize(self, x):
        return (x - self.x_mean) / self.x_std

    def denormalize(self, xn):
        return xn * self.x_std + self.x_mean

    def mse_loss(self, xn_t, xn_tp1):
        g = (xn_tp1 - xn_t) / self.sigma_d
        tau = torch.zeros(xn_t.shape[0], device=xn_t.device)
        g_hat = self.net(xn_t.unsqueeze(1), tau)
        return F.mse_loss(g_hat, g)

    def sample_step(self, xn_t, z=None, n_sampler_steps: int = 0):
        tau = torch.zeros(xn_t.shape[0], device=xn_t.device)
        return xn_t + self.sigma_d * self.net(xn_t.unsqueeze(1), tau)


def rollout(model, xn0: torch.Tensor, n_steps: int, n_sampler_steps: int = 6,
            detach_every: int = 0, use_checkpoint: bool = True,
            generator: torch.Generator | None = None) -> torch.Tensor:
    """Roll the surrogate forward. xn0: (B, K) normalized initial states.
    Returns xn trajectory (B, n_steps+1, K) INCLUDING the initial state.

    detach_every > 0 cuts the dynamics graph every that many steps (truncated
    BPTT through chaos); the returned trajectory tensors keep their local
    segment graphs so downstream losses get bounded gradients.
    """
    xs = [xn0]
    x = xn0
    is_det = isinstance(model, DetSurrogate)
    for t in range(n_steps):
        if is_det:
            z = None
        elif generator is not None:
            z = torch.randn(x.shape, generator=generator, device=x.device)
        else:
            z = torch.randn_like(x)
        if use_checkpoint and torch.is_grad_enabled():
            x = checkpoint(model.sample_step, x, z, n_sampler_steps,
                           use_reentrant=False)
        else:
            x = model.sample_step(x, z, n_sampler_steps)
        xs.append(x)
        if detach_every and (t + 1) % detach_every == 0:
            x = x.detach()
    return torch.stack(xs, dim=1)


def record_rollout(model, xn0: torch.Tensor, n_steps: int,
                   n_sampler_steps: int = 6):
    """No-grad rollout recording states and noises (eager path).
    Returns S_rec (B, T+1, K), Z (B, T, K)."""
    with torch.no_grad():
        states = [xn0]
        zs = []
        x = xn0
        for _ in range(n_steps):
            z = torch.randn_like(x)
            zs.append(z)
            x = model.sample_step(x, z, n_sampler_steps)
            states.append(x)
        return torch.stack(states, dim=1), torch.stack(zs, dim=1)


class GraphedRollout:
    """CUDA-graph-captured no-grad rollout (fixed B, T, K, S).

    The tiny per-step CNN is launch-overhead-bound (~5 ms/eval on Windows);
    capturing the whole T*S-eval rollout as one graph removes that. Model
    parameters are referenced by pointer, and optimizer steps update them
    in-place, so replays track training. Noise is filled into a static buffer
    OUTSIDE the capture (no RNG inside the graph).
    """

    def __init__(self, model, B: int, K: int, n_steps: int,
                 n_sampler_steps: int, device: str = "cuda"):
        self.n_steps = n_steps
        self.Z = torch.empty(B, n_steps, K, device=device)
        self.X0 = torch.zeros(B, K, device=device)
        self.Z.normal_()
        with torch.no_grad():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    _ = model.sample_step(self.X0, self.Z[:, 0], n_sampler_steps)
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                x = self.X0
                outs = [x]
                for t in range(n_steps):
                    x = model.sample_step(x, self.Z[:, t], n_sampler_steps)
                    outs.append(x)
                self.S_rec = torch.stack(outs, dim=1)

    def __call__(self, xn0: torch.Tensor):
        """Returns (S_rec, Z) clones — safe to reuse the instance."""
        self.X0.copy_(xn0)
        self.Z.normal_()
        self.graph.replay()
        return self.S_rec.clone(), self.Z.clone()


def reforward_windows(model, S_rec: torch.Tensor, Z: torch.Tensor,
                      win: int, n_sampler_steps: int = 6) -> torch.Tensor:
    """With-grad re-forward of a recorded rollout, all windows in parallel.

    Equivalent to rollout(..., detach_every=win): identical states (the
    sampler is deterministic given z) and identical gradient semantics (each
    state's gradient reaches back at most to its window's detached start) —
    but sequential graph depth is win*S evals instead of T*S, batched over
    B*W windows. Returns (B, T+1, K); initial state detached.
    """
    B, Tp1, K = S_rec.shape
    T = Tp1 - 1
    assert T % win == 0, "n_steps must be a multiple of win"
    W = T // win
    x0w = S_rec[:, :-1][:, ::win].reshape(B * W, K)          # window starts
    zw = Z.reshape(B, W, win, K).reshape(B * W, win, K)
    outs = []
    xw = x0w
    for t in range(win):
        xw = model.sample_step(xw, zw[:, t], n_sampler_steps)
        outs.append(xw)
    xr = torch.stack(outs, dim=1).reshape(B, W * win, K)     # (B, T, K)
    return torch.cat([S_rec[:, :1], xr], dim=1)


def rollout_windowed(model, xn0: torch.Tensor, n_steps: int, win: int,
                     n_sampler_steps: int = 6,
                     recorder=None) -> torch.Tensor:
    """record_rollout (or a GraphedRollout recorder) + reforward_windows."""
    if recorder is not None:
        S_rec, Z = recorder(xn0)
    else:
        S_rec, Z = record_rollout(model, xn0, n_steps, n_sampler_steps)
    return reforward_windows(model, S_rec, Z, win, n_sampler_steps)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)
