r"""Faithful delta-network cells (Neil et al. 2016 semantics).

Mechanism (Delta Networks for Optimized Recurrent Network Computation,
arXiv:1612.05571): each vector v feeding a matrix-vector product keeps an
anchor memory v_hat of its LAST TRANSMITTED value. At step t, component i
fires an event iff |v_t[i] - v_hat[i]| > theta; on an event the delta
(v_t[i] - v_hat[i]) is transmitted and the anchor is updated to v_t[i]
(anchor-on-event, which prevents drift accumulation); otherwise nothing is
transmitted and the anchor is kept. The product W v_t is maintained
incrementally, z_t = z_{t-1} + W (delta_t), so compute and weight fetches are
only spent on firing components.

theta = 0 must reproduce the dense network exactly (unit-tested).

Event accounting: one event = one transmitted component = one weight-column
fetch + fan-out MACs. energy.py turns (events, fan-out) into modeled energy.
"""

import torch
import torch.nn as nn


class DeltaEncoder:
    """Stateful send-on-delta encoder for a stream of vectors.

    Anchors initialize to zero (Neil et al.: the first step transmits the
    full input since |v_1 - 0| typically exceeds theta).
    """

    def __init__(self, theta):
        self.theta = float(theta)
        self.anchor = None

    def step(self, v):
        """v: (B, C). Returns (delta, events) where delta = v - anchor on
        firing components (0 elsewhere) and events is the firing mask."""
        if self.anchor is None:
            self.anchor = torch.zeros_like(v)
        diff = v - self.anchor
        if self.theta == 0.0:
            events = torch.ones_like(v, dtype=torch.bool)
            delta = diff
        else:
            events = diff.abs() > self.theta
            delta = diff * events
        self.anchor = self.anchor + delta
        return delta, events


def delta_encode_trace(traces, theta):
    """Run the send-on-delta state machine over full traces (no feedback).

    traces: (B, T, C) -> (anchors (B, T, C), events (B, T, C) bool).
    anchors[t] is the value the decoder holds AFTER step t's transmission,
    i.e. the delta approximation of traces[t].
    """
    enc = DeltaEncoder(theta)
    anchors, events = [], []
    for t in range(traces.shape[1]):
        _, ev = enc.step(traces[:, t])
        anchors.append(enc.anchor)
        events.append(ev)
    return torch.stack(anchors, dim=1), torch.stack(events, dim=1)


def sod_event_rate(traces, thetas):
    """Measured send-on-delta event rate per unit component per step.

    traces: (B, T, C); thetas: iterable -> tensor (len(thetas),) of rates.
    """
    out = []
    for th in thetas:
        _, ev = delta_encode_trace(traces, float(th))
        out.append(ev.float().mean().item())
    return torch.tensor(out)


class DeltaGRU(nn.Module):
    """Multi-layer GRU with delta-encoded inputs and hidden states.

    Weight-compatible with torch.nn.GRU (same parameter names and gate math,
    the cuDNN variant: n = tanh(W_in x + b_in + r * (W_hn h + b_hn))), so a
    trained nn.GRU state_dict loads directly. thetas: per-layer (theta_x,
    theta_h) pairs, or a scalar used everywhere.

    forward returns (output (B, T, H), stats) where stats has per-layer input
    and hidden event counts and dense component counts for energy accounting.
    """

    def __init__(self, input_size, hidden_size, num_layers=2, thetas=0.0):
        super().__init__()
        self.input_size, self.hidden_size, self.num_layers = input_size, hidden_size, num_layers
        for l in range(num_layers):
            in_sz = input_size if l == 0 else hidden_size
            setattr(self, f"weight_ih_l{l}", nn.Parameter(torch.empty(3 * hidden_size, in_sz)))
            setattr(self, f"weight_hh_l{l}", nn.Parameter(torch.empty(3 * hidden_size, hidden_size)))
            setattr(self, f"bias_ih_l{l}", nn.Parameter(torch.empty(3 * hidden_size)))
            setattr(self, f"bias_hh_l{l}", nn.Parameter(torch.empty(3 * hidden_size)))
        self.set_thetas(thetas)
        self.reset_parameters()

    def set_thetas(self, thetas):
        if isinstance(thetas, (int, float)):
            self.thetas = [(float(thetas), float(thetas))] * self.num_layers
        else:
            self.thetas = [(float(tx), float(th)) for tx, th in thetas]

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.uniform_(p, -1.0 / self.hidden_size ** 0.5, 1.0 / self.hidden_size ** 0.5)

    @torch.no_grad()
    def forward(self, x):
        B, T, _ = x.shape
        H = self.hidden_size
        stats = []
        layer_in = x
        for l in range(self.num_layers):
            th_x, th_h = self.thetas[l]
            W_ih = getattr(self, f"weight_ih_l{l}")
            W_hh = getattr(self, f"weight_hh_l{l}")
            b_ih = getattr(self, f"bias_ih_l{l}")
            b_hh = getattr(self, f"bias_hh_l{l}")
            enc_x, enc_h = DeltaEncoder(th_x), DeltaEncoder(th_h)
            h = torch.zeros(B, H, device=x.device, dtype=x.dtype)
            acc_ih = torch.zeros(B, 3 * H, device=x.device, dtype=x.dtype)
            acc_hh = torch.zeros(B, 3 * H, device=x.device, dtype=x.dtype)
            n_ev_x = torch.zeros((), device=x.device)
            n_ev_h = torch.zeros((), device=x.device)
            outs = []
            for t in range(T):
                dx, ev_x = enc_x.step(layer_in[:, t])
                acc_ih = acc_ih + dx @ W_ih.t()
                dh, ev_h = enc_h.step(h)
                acc_hh = acc_hh + dh @ W_hh.t()
                gi = acc_ih + b_ih
                gh = acc_hh + b_hh
                i_r, i_z, i_n = gi.chunk(3, dim=1)
                h_r, h_z, h_n = gh.chunk(3, dim=1)
                r = torch.sigmoid(i_r + h_r)
                z = torch.sigmoid(i_z + h_z)
                n = torch.tanh(i_n + r * h_n)
                h = (1.0 - z) * n + z * h
                n_ev_x = n_ev_x + ev_x.sum()
                n_ev_h = n_ev_h + ev_h.sum()
                outs.append(h)
            layer_out = torch.stack(outs, dim=1)
            n_ev_x = float(n_ev_x.item())
            n_ev_h = float(n_ev_h.item())
            in_sz = layer_in.shape[-1]
            stats.append(dict(
                layer=l, theta_x=th_x, theta_h=th_h,
                events_x=n_ev_x, events_h=n_ev_h,
                dense_x=float(B * T * in_sz), dense_h=float(B * T * H),
                fanout_x=3 * H, fanout_h=3 * H,
            ))
            layer_in = layer_out
        return layer_in, stats


class GRUClassifier(nn.Module):
    """Base (dense) model: stack of single-layer nn.GRUs + pool + linear head.

    Layers are separate nn.GRU modules so per-layer hidden traces come out of
    the normal (differentiable, cuDNN) forward — the crossing budget needs
    gradients through the traces. State dict maps 1:1 onto DeltaGRU.
    """

    def __init__(self, input_size, hidden_size, num_layers, n_classes, pool="mean"):
        super().__init__()
        self.input_size, self.hidden_size, self.num_layers = input_size, hidden_size, num_layers
        self.layers = nn.ModuleList([
            nn.GRU(input_size if l == 0 else hidden_size, hidden_size, 1,
                   batch_first=True)
            for l in range(num_layers)
        ])
        self.head = nn.Linear(hidden_size, n_classes)
        self.pool = pool

    def forward(self, x, return_traces=False):
        traces = []
        layer_in = x
        for cell in self.layers:
            layer_in, _ = cell(layer_in)
            traces.append(layer_in)
        feat = traces[-1].mean(dim=1) if self.pool == "mean" else traces[-1][:, -1]
        logits = self.head(feat)
        return (logits, traces) if return_traces else logits

    def as_delta(self, thetas):
        """Weight-shared DeltaGRU for event-counted evaluation."""
        d = DeltaGRU(self.input_size, self.hidden_size, self.num_layers, thetas)
        sd = {}
        for l, cell in enumerate(self.layers):
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                sd[f"{name}_l{l}"] = getattr(cell, f"{name}_l0").data
        d.load_state_dict(sd, strict=True)
        return d.to(next(self.parameters()).device)


class TransformerLM(nn.Module):
    """Small causal char-LM whose projection matvecs (QKV, attention-out,
    FFN fc1/fc2) can be delta-encoded along the sequence axis (streaming
    semantics). Attention score/value matmuls stay dense, as in the
    event-driven transformer literature: the delta mechanism targets the
    weight-stationary projections.

    set_thetas(theta) wraps/updates DeltaLinear thetas; theta=0 is exactly
    the dense model. return_traces yields the input streams of every wrapped
    linear (for crossing statistics and the budget).
    """

    def __init__(self, vocab, dim=128, n_layers=2, n_heads=4, ffn=512,
                 seq_len=256):
        super().__init__()
        self.dim, self.n_heads, self.seq_len = dim, n_heads, seq_len
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            b = nn.ModuleDict(dict(
                ln1=nn.LayerNorm(dim),
                qkv=DeltaLinear(nn.Linear(dim, 3 * dim), 0.0),
                out=DeltaLinear(nn.Linear(dim, dim), 0.0),
                ln2=nn.LayerNorm(dim),
                fc1=DeltaLinear(nn.Linear(dim, ffn), 0.0),
                fc2=DeltaLinear(nn.Linear(ffn, dim), 0.0),
            ))
            self.blocks.append(b)
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def set_thetas(self, theta):
        """theta: scalar or per-layer list of dicts {qkv, out, fc1, fc2}."""
        for l, b in enumerate(self.blocks):
            th = theta if isinstance(theta, (int, float)) else theta[l]
            for k in ("qkv", "out", "fc1", "fc2"):
                b[k].theta = float(th) if isinstance(th, (int, float)) else float(th[k])

    def reset_event_counts(self):
        for b in self.blocks:
            for k in ("qkv", "out", "fc1", "fc2"):
                b[k].events = 0.0
                b[k].dense = 0.0

    def event_stats(self):
        stats = []
        for l, b in enumerate(self.blocks):
            for k in ("qkv", "out", "fc1", "fc2"):
                m = b[k]
                stats.append(dict(layer=l, name=k, events_x=m.events,
                                  events_h=0.0, dense_x=m.dense, dense_h=0.0,
                                  fanout_x=m.fanout, fanout_h=0))
        return stats

    def forward(self, tokens, return_traces=False):
        B, T = tokens.shape
        x = self.embed(tokens) + self.pos[:, :T]
        traces = {}
        mask = torch.triu(torch.ones(T, T, device=tokens.device, dtype=torch.bool), 1)
        for l, b in enumerate(self.blocks):
            h = b["ln1"](x)
            if return_traces:
                traces[f"l{l}.qkv_in"] = h
            qkv = b["qkv"](h)
            q, k, v = qkv.chunk(3, dim=-1)
            hd = self.dim // self.n_heads
            q = q.view(B, T, self.n_heads, hd).transpose(1, 2)
            k = k.view(B, T, self.n_heads, hd).transpose(1, 2)
            v = v.view(B, T, self.n_heads, hd).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / hd ** 0.5
            att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
            a = (att @ v).transpose(1, 2).reshape(B, T, self.dim)
            if return_traces:
                traces[f"l{l}.out_in"] = a
            x = x + b["out"](a)
            h2 = b["ln2"](x)
            if return_traces:
                traces[f"l{l}.fc1_in"] = h2
            f1 = torch.nn.functional.gelu(b["fc1"](h2))
            if return_traces:
                traces[f"l{l}.fc2_in"] = f1
            x = x + b["fc2"](f1)
        logits = self.head(self.ln_f(x))
        return (logits, traces) if return_traces else logits


class DeltaLinear(nn.Module):
    """Linear layer whose input stream (over the sequence axis) is
    send-on-delta encoded: y_t = W x_hat_t + b with x_hat the anchor sequence.
    Vectorized: the anchor state machine runs over time (cheap, elementwise),
    then one dense matmul on the anchors. Event counts accumulate in .events /
    .dense for energy accounting."""

    def __init__(self, linear, theta):
        super().__init__()
        self.linear = linear
        self.theta = float(theta)
        self.events = 0.0
        self.dense = 0.0

    def forward(self, x):
        # x: (B, T, C) sequence-major stream
        if self.theta > 0.0:
            anchors, ev = delta_encode_trace(x, self.theta)
            self.events += ev.float().sum().item()
        else:
            anchors = x
            self.events += float(x.numel())
        self.dense += float(x.numel())
        return self.linear(anchors)

    @property
    def fanout(self):
        return self.linear.out_features
