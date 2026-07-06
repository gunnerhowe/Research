"""LOCATE: estimate the selection function s(x)=P(observed|x) by density-ratio-
via-classification (Sugiyama et al. 2012 reduction) and quantify where that
estimate is trustworthy.

A calibrated deep ensemble separates the curated corpus D_obs (label 1) from
the uncensored reference pool D_ref ~ p(x) (label 0), trained BALANCED so the
calibrated classifier c(x) satisfies c/(1-c) = p_obs(x)/p_ref(x) = s(x)/Z with
Z=P(observed) the known marginal selection rate; hence

    s_hat(x) = clip( Z * c(x)/(1-c(x)),  0, 1 ).

IDENTIFIABILITY (load-bearing). In the deep censored region D_obs has zero
points, so every member drives c(x)->0 by DEFAULT (reference-only), not by
measurement: s_hat there is off-support extrapolation. Ensemble members
disagree about HOW they extrapolate, so the epistemic std u(x)=std_m s_hat_m(x)
spikes there. The trustworthy region -- the recoverable COLLAR -- is gated by
{p_hat>=tau} AND {proximity to observed support} AND {low u(x)}. The bridge is
steered only inside that collar and NEVER by high-variance s_hat.

s_hat is differentiable in x (torch), so the reconstruction-guidance bridge can
backprop grad_x log r(x_hat_0) through it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _MLP(nn.Module):
    def __init__(self, hidden=64, depth=2, in_dim=2):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SelectionEstimator:
    """Calibrated deep-ensemble density-ratio selection estimator."""

    def __init__(self, n_members=5, hidden=64, depth=2, epochs=300, lr=3e-3,
                 weight_decay=1e-4, device=DEVICE):
        self.n_members = n_members
        self.hidden, self.depth = hidden, depth
        self.epochs, self.lr, self.wd = epochs, lr, weight_decay
        self.device = device
        self.members: list[_MLP] = []
        self.temps: list[float] = []
        self.obs_frac = 0.5
        self._mu = None
        self._sd = None
        self._obs_tree = None
        self._val_probs = None      # held-out calibrated probs (for ECE)
        self._val_labels = None

    # -------------------------------------------------------------- fit
    def fit(self, X_obs, X_ref, obs_frac, seed=0):
        self.obs_frac = float(np.clip(obs_frac, 1e-3, 1 - 1e-3))
        self._obs_tree = cKDTree(X_obs)
        self.in_dim = X_ref.shape[1]
        # standardize inputs on the reference pool (full support)
        self._mu = X_ref.mean(0)
        self._sd = X_ref.std(0) + 1e-8

        Xo = self._z(X_obs)
        Xr = self._z(X_ref)
        rng = np.random.default_rng(seed)
        self.members, self.temps = [], []
        val_p, val_y = [], []
        for m in range(self.n_members):
            mseed = int(rng.integers(1 << 30))
            model, temp, vp, vy = self._fit_member(Xo, Xr, mseed)
            self.members.append(model)
            self.temps.append(temp)
            val_p.append(vp)
            val_y.append(vy)
        self._val_probs = np.concatenate(val_p)
        self._val_labels = np.concatenate(val_y)
        return self

    def _fit_member(self, Xo, Xr, seed):
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        # balance classes by subsampling the larger pool; bootstrap each
        n = min(len(Xo), len(Xr))
        io = torch.randint(len(Xo), (n,), generator=g)
        ir = torch.randint(len(Xr), (n,), generator=g)
        Xo_b, Xr_b = Xo[io.numpy()], Xr[ir.numpy()]
        # 85/15 train/val split for temperature calibration
        ntr = int(0.85 * n)
        Xtr = np.concatenate([Xo_b[:ntr], Xr_b[:ntr]])
        ytr = np.concatenate([np.ones(ntr), np.zeros(ntr)])
        Xva = np.concatenate([Xo_b[ntr:], Xr_b[ntr:]])
        yva = np.concatenate([np.ones(n - ntr), np.zeros(n - ntr)])

        Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=self.device)
        ytr_t = torch.tensor(ytr, dtype=torch.float32, device=self.device)
        model = _MLP(self.hidden, self.depth, in_dim=self.in_dim).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr,
                               weight_decay=self.wd)
        lossf = nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = lossf(model(Xtr_t), ytr_t)
            loss.backward()
            opt.step()
        model.eval()

        # temperature scaling on the validation split (NLL-optimal scalar T)
        Xva_t = torch.tensor(Xva, dtype=torch.float32, device=self.device)
        yva_t = torch.tensor(yva, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logit_va = model(Xva_t)
        logT = torch.zeros(1, device=self.device, requires_grad=True)
        opt_t = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)

        def closure():
            opt_t.zero_grad()
            l = lossf(logit_va / torch.exp(logT), yva_t)
            l.backward()
            return l
        opt_t.step(closure)
        temp = float(torch.exp(logT).item())
        with torch.no_grad():
            pva = torch.sigmoid(logit_va / temp).cpu().numpy()
        return model, temp, pva, yva

    # ------------------------------------------------------- s_hat (torch)
    def _z_torch(self, X):
        mu = torch.tensor(self._mu, dtype=torch.float32, device=X.device)
        sd = torch.tensor(self._sd, dtype=torch.float32, device=X.device)
        return (X - mu) / sd

    def s_hat_members_torch(self, X):
        """(n_members, n) calibrated s_hat per member; differentiable in X."""
        Xz = self._z_torch(X)
        out = []
        for model, temp in zip(self.members, self.temps):
            c = torch.sigmoid(model(Xz) / temp).clamp(1e-6, 1 - 1e-6)
            dr = c / (1 - c)
            out.append((self.obs_frac * dr).clamp(0.0, 1.0))
        return torch.stack(out, 0)

    def s_hat_torch(self, X):
        return self.s_hat_members_torch(X).mean(0)

    def logit_members_torch(self, X):
        """(n_members, n) calibrated log-odds per member."""
        Xz = self._z_torch(X)
        return torch.stack([model(Xz) / temp
                            for model, temp in zip(self.members, self.temps)], 0)

    # ------------------------------------------------------- s_hat (numpy)
    def _z(self, X):
        return (X - self._mu) / self._sd

    @torch.no_grad()
    def s_hat(self, X):
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        return self.s_hat_members_torch(Xt).mean(0).cpu().numpy()

    @torch.no_grad()
    def uncertainty(self, X):
        """Epistemic uncertainty of s_hat = ensemble std of the calibrated
        LOG-ODDS. This is the load-bearing identifiability signal: unlike the
        std of s_hat itself (which saturates to ~0 as s_hat->0), the log-odds
        std GROWS in the deep censored complement, where members extrapolate to
        different large-negative logits with no observed data to pin them."""
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        return self.logit_members_torch(Xt).std(0).cpu().numpy()

    def uncertainty_quantile(self, X, q):
        """A q-quantile of the epistemic uncertainty over a pool X; the collar's
        u_max is pre-registered as such a quantile (adapts per fit by rule)."""
        return float(np.quantile(self.uncertainty(X), q))

    def proximity(self, X):
        """Distance to the nearest observed point (short-extrapolation gate)."""
        d, _ = self._obs_tree.query(np.asarray(X), k=1)
        return d

    # ----------------------------------------------------------- calibration
    def ece(self, n_bins=15):
        """Expected calibration error of the (obs vs ref) classifier on the
        held-out calibration split."""
        p, y = self._val_probs, self._val_labels
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
        ece = 0.0
        for b in range(n_bins):
            m = idx == b
            if m.any():
                ece += m.mean() * abs(p[m].mean() - y[m].mean())
        return float(ece)
