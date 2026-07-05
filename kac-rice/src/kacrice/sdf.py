"""Neural SDF fitting from point clouds, with topology-control losses (exp10).

Conventions: network f approximates the SIGNED distance, negative INSIDE.
The occupancy field used for topology is o = -f (positive inside), so
superlevel sets {o >= u} for small |u| are the solid shape eroded/dilated.
Euler characteristics of the solids: ball 1, solid torus 0, genus-2 solid -1.

Baseline: STITCH-style persistent-homology loss on a cubical grid (gudhi),
differentiable through the critical-cell values, reimplemented faithfully in
spirit: penalize the persistence of all spurious H0/H1/H2 features of the
occupancy grid beyond those the target class allows.

Evaluation: voxel Betti numbers b0 (components), b2 (cavities), and b1 via
b1 = b0 + b2 - chi with chi computed exactly on the cubical complex.
"""

import numpy as np
import torch

from .minkowski import EulerProfileLoss, field_derivatives  # noqa: F401


# ---------------------------------------------------------------- shapes

def sdf_sphere(p, r=0.55):
    return p.norm(dim=-1) - r


def sdf_torus(p, R=0.5, r=0.22):
    q = torch.stack([p[:, :2].norm(dim=-1) - R, p[:, 2]], dim=-1)
    return q.norm(dim=-1) - r


def sdf_genus2(p, R=0.35, r=0.16, sep=0.38):
    """Smooth union of two offset tori: genus-2 solid (chi = -1)."""
    p1 = p.clone(); p1[:, 0] += sep
    p2 = p.clone(); p2[:, 0] -= sep
    d1, d2 = sdf_torus(p1, R, r), sdf_torus(p2, R, r)
    k = 0.05  # smooth-min radius
    h = (0.5 + 0.5 * (d2 - d1) / k).clamp(0, 1)
    return d2 + (d1 - d2) * h - k * h * (1 - h)


SHAPES = {
    "sphere": (sdf_sphere, 1),   # chi of the solid
    "torus": (sdf_torus, 0),
    "genus2": (sdf_genus2, -1),
}


def sample_pointcloud(shape, n=2000, noise=0.02, outlier_frac=0.05, seed=0):
    """Noisy surface samples + uniform outliers (the 'bad scan' regime that
    induces spurious topology in unregularized fits)."""
    fn, _ = SHAPES[shape]
    g = torch.Generator().manual_seed(seed)
    pts = []
    while sum(p.shape[0] for p in pts) < n:
        cand = torch.rand(20000, 3, generator=g) * 2 - 1
        d = fn(cand)
        near = cand[d.abs() < 0.08]
        # project to surface via a few gradient steps on the analytic sdf
        near = near.clone().requires_grad_(True)
        for _ in range(3):
            dd = fn(near)
            (gr,) = torch.autograd.grad(dd.sum(), near)
            near = (near - dd.unsqueeze(-1) * gr /
                    gr.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    ).detach().requires_grad_(True)
        pts.append(near.detach())
    surf = torch.cat(pts)[:n]
    surf = surf + noise * torch.randn(n, 3, generator=g)
    n_out = int(outlier_frac * n)
    outliers = torch.rand(n_out, 3, generator=g) * 2 - 1
    return torch.cat([surf, outliers])


# ---------------------------------------------------------------- eval

def occupancy_grid(net, n=96, device="cpu", chunk=200_000):
    xs = torch.linspace(-1, 1, n, device=device)
    gz, gy, gx = torch.meshgrid(xs, xs, xs, indexing="ij")
    pts = torch.stack([gx, gy, gz], -1).reshape(-1, 3)
    out = []
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk):
            out.append(net(pts[i:i + chunk]).squeeze(-1).cpu())
    return (-torch.cat(out)).reshape(n, n, n).numpy()  # occupancy = -sdf


def voxel_betti(occ_field, level=0.0):
    """b0, b1, b2 of the solid {occ > level} on the voxel grid.
    chi via the cell-counting formula; b1 = b0 + b2 - chi."""
    from scipy import ndimage
    o = occ_field > level
    if not o.any():
        return 0, 0, 0
    b0 = ndimage.label(o)[1]
    b2 = ndimage.label(~o)[1] - 1  # cavities: complement comps minus outside
    # chi = V - E + F - C of the cubical complex of occupied voxels
    p = np.pad(o, 1).astype(np.int8)
    C = int(o.sum())
    # faces: adjacencies along each axis count shared faces once
    fx = np.logical_and(p[1:, :, :], p[:-1, :, :]).sum()
    fy = np.logical_and(p[:, 1:, :], p[:, :-1, :]).sum()
    fz = np.logical_and(p[:, :, 1:], p[:, :, :-1]).sum()
    # complex counts: each cube contributes; use inclusion of maxima of shifts
    # vertices: a grid corner exists if ANY of its 8 adjacent voxels occupied
    occ8 = np.zeros((p.shape[0] + 1, p.shape[1] + 1, p.shape[2] + 1), bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                occ8[dx:dx + p.shape[0], dy:dy + p.shape[1],
                     dz:dz + p.shape[2]] |= p.astype(bool)
    V = int(occ8.sum())
    # edges along x: exists if any of 4 adjacent voxels occupied
    ex = np.zeros((p.shape[0], p.shape[1] + 1, p.shape[2] + 1), bool)
    for dy in (0, 1):
        for dz in (0, 1):
            ex[:, dy:dy + p.shape[1], dz:dz + p.shape[2]] |= p.astype(bool)
    ey = np.zeros((p.shape[0] + 1, p.shape[1], p.shape[2] + 1), bool)
    for dx in (0, 1):
        for dz in (0, 1):
            ey[dx:dx + p.shape[0], :, dz:dz + p.shape[2]] |= p.astype(bool)
    ez = np.zeros((p.shape[0] + 1, p.shape[1] + 1, p.shape[2]), bool)
    for dx in (0, 1):
        for dy in (0, 1):
            ez[dx:dx + p.shape[0], dy:dy + p.shape[1], :] |= p.astype(bool)
    E = int(ex.sum() + ey.sum() + ez.sum())
    # 2-faces: xy-planes etc., exists if either voxel sharing it is occupied
    fxy = np.zeros((p.shape[0], p.shape[1], p.shape[2] + 1), bool)
    for dz in (0, 1):
        fxy[:, :, dz:dz + p.shape[2]] |= p.astype(bool)
    fxz = np.zeros((p.shape[0], p.shape[1] + 1, p.shape[2]), bool)
    for dy in (0, 1):
        fxz[:, dy:dy + p.shape[1], :] |= p.astype(bool)
    fyz = np.zeros((p.shape[0] + 1, p.shape[1], p.shape[2]), bool)
    for dx in (0, 1):
        fyz[dx:dx + p.shape[0], :, :] |= p.astype(bool)
    F = int(fxy.sum() + fxz.sum() + fyz.sum())
    chi = V - E + F - C
    b1 = b0 + b2 - chi
    return int(b0), int(b1), int(b2)


# ---------------------------------------------------------------- PH loss

class CubicalPHLoss(torch.nn.Module):
    """STITCH-style topological loss: persistence of spurious features of the
    occupancy field on a cubical grid, differentiable through the values at
    the critical cells (gudhi supplies the cell indices).

    allowed: dict like {0: 1, 1: 0, 2: 0} - number of PERMITTED features per
    homology dimension (e.g. torus solid: {0:1, 1:1, 2:0}). All features
    beyond the allowed count (ranked by persistence) are penalized by
    (death - birth)^2; infinite-death features use the field max.
    """

    def __init__(self, allowed, grid_n=48, device="cpu"):
        super().__init__()
        self.allowed = allowed
        self.n = grid_n
        xs = torch.linspace(-1, 1, grid_n, device=device)
        gz, gy, gx = torch.meshgrid(xs, xs, xs, indexing="ij")
        self.register_buffer("pts",
                             torch.stack([gx, gy, gz], -1).reshape(-1, 3))

    def forward(self, net):
        import gudhi

        occ = -net(self.pts).squeeze(-1)          # occupancy, WITH grad
        vals = (-occ).detach().cpu().numpy()      # gudhi filters SUBLEVEL of f
        cc = gudhi.CubicalComplex(
            dimensions=[self.n, self.n, self.n],
            top_dimensional_cells=vals.reshape(-1))
        cc.compute_persistence()
        pairs = cc.cofaces_of_persistence_pairs()
        loss = occ.sum() * 0.0
        reg, ess = pairs[0], pairs[1]
        f_flat = -occ  # sublevel filtration values, WITH grad
        for dim, dim_pairs in enumerate(reg):
            if len(dim_pairs) == 0:
                continue
            births = f_flat[torch.as_tensor(dim_pairs[:, 0],
                                            device=occ.device, dtype=torch.long)]
            deaths = f_flat[torch.as_tensor(dim_pairs[:, 1],
                                            device=occ.device, dtype=torch.long)]
            pers = deaths - births
            allowed = self.allowed.get(dim, 0)
            # essential features occupy the top slots of this dimension
            n_ess = len(ess[dim]) if dim < len(ess) else 0
            keep = max(allowed - n_ess, 0)
            if pers.numel() > keep:
                spurious = torch.topk(pers, pers.numel())[0][keep:]
                loss = loss + (spurious ** 2).sum()
        return loss
