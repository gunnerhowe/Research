"""Generate publication figures + LaTeX tables from results/paper/*.json.

Outputs to paper/figs/ and paper/tables/. Run after run_paper.py.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import ROOT

RES = ROOT / "results" / "paper"
RECONS = RES / "recons"
FIGS = ROOT / "paper" / "figs"
TABS = ROOT / "paper" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.4,
})

LABELS = {
    "pemlp_mse": "MSE only",
    "pemlp_ffl": "+ FFL",
    "pemlp_sobolev": "+ Sobolev",
    "pemlp_kacrice": "+ Kac-Rice (ours)",
    "siren_mse": "SIREN (MSE)",
    "finer_mse": "FINER (MSE)",
    "mse": "MSE only",
    "ffl_interp": "+ FFL (interp.)",
    "sobolev_est": "+ Sobolev (est. grad)",
    "kacrice_est": "+ Kac-Rice (est. grad, ours)",
    "sobolev_oracle": "+ Sobolev (oracle grad)",
    "kacrice_oracle": "+ Kac-Rice (oracle grad)",
    "siren_kacrice_est": "SIREN + Kac-Rice (ours)",
    "siren_ffl_interp": "SIREN + FFL (interp.)",
    "siren_sobolev_est": "SIREN + Sobolev (est. grad)",
}
COLORS = {
    "pemlp_mse": "#555555", "mse": "#555555",
    "pemlp_ffl": "#1f77b4", "ffl_interp": "#1f77b4",
    "pemlp_sobolev": "#2ca02c", "sobolev_est": "#2ca02c",
    "pemlp_kacrice": "#d62728", "kacrice_est": "#d62728",
    "sobolev_oracle": "#98df8a", "kacrice_oracle": "#ff9896",
    "siren_mse": "#9467bd", "siren_kacrice_est": "#8c564b",
    "finer_mse": "#e377c2",
}


def load(suite):
    """Merge results from all parallel workers (suite.json, suite_a.json, ...),
    dropping duplicate (image, mode, config, seed, tag) records — workers launched
    before resume-support could redo runs another worker already recorded."""
    recs, seen = [], set()
    for p in sorted(RES.glob(f"{suite}*.json")):
        for r in json.loads(p.read_text()):
            key = tuple(r.get(k) for k in ("image", "mode", "config", "seed",
                                           "tag", "n_samples"))
            if key in seen:
                continue
            seen.add(key)
            recs.append(r)
    return recs


def curves(records, key="psnr"):
    """-> {config: (iters, mean, std)} across seeds."""
    by_cfg = defaultdict(list)
    for r in records:
        it = [h["iter"] for h in r["history"] if key in h]
        v = [h[key] for h in r["history"] if key in h]
        by_cfg[r["config"]].append((it, v))
    out = {}
    for cfg, runs in by_cfg.items():
        iters = runs[0][0]
        vals = np.array([v for _, v in runs])
        out[cfg] = (iters, vals.mean(0), vals.std(0))
    return out


def finals(records, keys=("psnr", "hf_psnr", "ssim")):
    """-> {config: {key: (mean, std)}} using last history entry."""
    by_cfg = defaultdict(lambda: defaultdict(list))
    for r in records:
        last = r["history"][-1]
        for k in keys:
            if k in last:
                by_cfg[r["config"]][k].append(last[k])
    return {
        cfg: {k: (float(np.mean(v)), float(np.std(v))) for k, v in d.items()}
        for cfg, d in by_cfg.items()
    }


def plot_curve_panel(ax, cs, order, key_label, title):
    for cfg in order:
        if cfg not in cs:
            continue
        it, m, s = cs[cfg]
        c = COLORS.get(cfg)
        ax.plot(it, m, label=LABELS.get(cfg, cfg), color=c)
        ax.fill_between(it, m - s, m + s, alpha=0.15, color=c, lw=0)
    ax.set_xlabel("iteration")
    ax.set_ylabel(key_label)
    ax.set_title(title)


# ------------------------------------------------------------- fig: method

def fig_method():
    """Illustration: signal + levels + crossings; crossing-density profile."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from kacrice.crossing import crossing_density
    from kacrice.data import multisine

    f, df = multisine(freqs=(2, 5, 11, 23, 47), seed=0)
    xg = torch.linspace(-1, 1, 4000)
    yg = f(xg)
    u = 0.6
    sgn = (yg[:-1] - u) * (yg[1:] - u) < 0
    xc = xg[:-1][sgn]

    x_mc = torch.rand(200_000) * 2 - 1
    levels = torch.linspace(yg.min().item(), yg.max().item(), 60)
    c_true = crossing_density(f(x_mc), df(x_mc).abs(), levels, eps=0.05)

    # a band-limited (low-pass) approximation = what a spectrally-biased INR fits
    flo, dflo = multisine(freqs=(2, 5, 11), amps=(1.0, 1 / math.sqrt(2), 1 / math.sqrt(3)), seed=0)
    c_lo = crossing_density(flo(x_mc), dflo(x_mc).abs(), levels, eps=0.05)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax = axes[0]
    ax.plot(xg, yg, lw=0.9, color="#333")
    ax.axhline(u, color="#d62728", lw=1.0, ls="--")
    ax.plot(xc, torch.full_like(xc, u), ".", color="#d62728", ms=4)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title(f"level $u={u}$: {len(xc)} crossings on $[-1,1]$")

    ax = axes[1]
    ax.plot(levels, c_true, color="#333", label="target signal")
    ax.plot(levels, c_lo, color="#1f77b4", ls="--",
            label="low-pass fit (spectral bias)")
    ax.set_xlabel("level $u$")
    ax.set_ylabel("crossing density $c(u)$")
    ax.set_title("crossing-density profiles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_method.pdf")
    plt.close(fig)


# --------------------------------------------------------- fig: validation

def fig_validation():
    """MC estimator vs direct counting vs Rice formula across levels."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from kacrice.crossing import crossing_density

    torch.manual_seed(0)
    m = 60
    freqs = torch.rand(m) * 100 + 20
    amps = torch.randn(m).abs() * 0.3
    phases = torch.rand(m) * 2 * math.pi

    def f(x):
        return sum(a * torch.sin(w * x + p) for a, w, p in zip(amps, freqs, phases))

    def df(x):
        return sum(a * w * torch.cos(w * x + p)
                   for a, w, p in zip(amps, freqs, phases))

    x = torch.rand(400_000) * 2 - 1
    fv, dv = f(x), df(x)
    l0, l2 = fv.var().item(), dv.var().item()
    sd = math.sqrt(l0)
    levels = torch.linspace(-2.5 * sd, 2.5 * sd, 41)

    c_mc = crossing_density(fv, dv.abs(), levels, eps=0.05)

    xg = torch.linspace(-1, 1, 2_000_000)
    fg = f(xg)
    c_count = [(((fg[:-1] - u) * (fg[1:] - u)) < 0).sum().item() / 2.0
               for u in levels]
    rice = [(1 / math.pi) * math.sqrt(l2 / l0) * math.exp(-(u**2) / (2 * l0))
            for u in levels.tolist()]

    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.plot(levels, c_count, "o", ms=3, color="#333", label="direct count (dense grid)")
    ax.plot(levels, c_mc, color="#d62728", label=r"MC estimator $\hat c_\varepsilon(u)$")
    ax.plot(levels, rice, "--", color="#1f77b4", label="Rice formula (empir. moments)")
    ax.set_xlabel("level $u$")
    ax.set_ylabel("crossings per unit length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_validation.pdf")
    plt.close(fig)


# ------------------------------------------------------------- fig: exp1

def fig_exp1():
    r1 = load("exp1_1d")
    r2 = [r for r in load("exp1_2d") if r["image"] == "camera"]
    order = ["pemlp_mse", "pemlp_ffl", "pemlp_sobolev", "pemlp_kacrice",
             "siren_mse", "finer_mse"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    plot_curve_panel(axes[0], curves(r1), order, "PSNR (dB)",
                     "1D multisine, regular grid")
    plot_curve_panel(axes[1], curves(r2), order, "PSNR (dB)",
                     "2D camera $128^2$, regular grid")
    axes[0].legend(ncol=1)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_exp1_curves.pdf")
    plt.close(fig)


# ------------------------------------------------------------- fig: exp2

EXP2_ORDER = ["mse", "ffl_interp", "sobolev_est", "kacrice_est"]


def fig_exp2_curves():
    recs = [r for r in load("exp2_2d")
            if r["image"] == "camera" and r["mode"] == "blobs"
            and r["config"] in EXP2_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    plot_curve_panel(axes[0], curves(recs, "psnr"), EXP2_ORDER, "PSNR (dB)",
                     "scattered (blobs): full-image PSNR")
    plot_curve_panel(axes[1], curves(recs, "hf_psnr"), EXP2_ORDER,
                     "HF-PSNR (dB)", "scattered (blobs): high-frequency PSNR")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_exp2_curves.pdf")
    plt.close(fig)


def fig_bands():
    """Per-band spectral error: grid (left) vs scattered (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    r_grid = [r for r in load("exp1_2d") if r["image"] == "camera"]
    r_scat = [r for r in load("exp2_2d")
              if r["image"] == "camera" and r["mode"] == "blobs"]
    for ax, recs, order, title in (
        (axes[0], r_grid, ["pemlp_mse", "pemlp_ffl", "pemlp_sobolev",
                           "pemlp_kacrice"], "regular grid"),
        (axes[1], r_scat, EXP2_ORDER, "scattered (blobs)"),
    ):
        by_cfg = defaultdict(list)
        for r in recs:
            if r["config"] in order:
                by_cfg[r["config"]].append(r["history"][-1]["bands"])
        n = len(next(iter(by_cfg.values()))[0])
        x = np.arange(n)
        w = 0.8 / len(order)
        for i, cfg in enumerate(order):
            if cfg not in by_cfg:
                continue
            arr = np.array(by_cfg[cfg])
            ax.bar(x + i * w, arr.mean(0), w, yerr=arr.std(0),
                   label=LABELS.get(cfg, cfg), color=COLORS.get(cfg),
                   error_kw=dict(lw=0.7))
        ax.set_yscale("log")
        ax.set_xticks(x + 0.4)
        ax.set_xticklabels([str(b) for b in x])
        ax.set_xlabel("radial frequency band (0 = low)")
        ax.set_ylabel("rel. spectral error")
        ax.set_title(title)
    axes[1].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_bands.pdf")
    plt.close(fig)


def fig_recons():
    """Qualitative: GT, samples, interp, and reconstructions with HF rows."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from kacrice.metrics import highpass

    gt = torch.from_numpy(np.load(RECONS / "exp2_2d_gt.npy"))
    interp = torch.from_numpy(np.load(RECONS / "exp2_2d_interp.npy"))
    pts = np.load(RECONS / "exp2_2d_pts.npy")

    cols = [("GT", gt), ("griddata interp.", interp)]
    for cfg in EXP2_ORDER:
        p = RECONS / f"exp2_2d_{cfg}.npy"
        if p.exists():
            cols.append((LABELS.get(cfg, cfg), torch.from_numpy(np.load(p))))

    n = len(cols) + 1
    fig, axes = plt.subplots(2, n, figsize=(1.55 * n, 3.3))
    # sample-location panel
    axes[0, 0].scatter(pts[:, 0], -pts[:, 1], s=0.05, color="#333")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_title("sample locations", fontsize=7)
    axes[1, 0].axis("off")
    for j, (name, img) in enumerate(cols, start=1):
        axes[0, j].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(name, fontsize=7)
        axes[1, j].imshow(highpass(img), cmap="gray", vmin=-0.12, vmax=0.12)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[1, 1].set_ylabel("high-pass", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_recons.pdf")
    plt.close(fig)


def fig_exp2_1d():
    """1D irregular: reconstruction detail + summary."""
    gt = np.load(RECONS / "exp1_1d_gt.npy")
    xs = np.linspace(-1, 1, len(gt))
    samples = np.load(RECONS / "exp2_1d_samples.npy")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    ax = axes[0]
    ax.plot(xs, gt, color="#333", lw=0.8, label="GT")
    for cfg in ("mse", "ffl_interp", "kacrice_est"):
        p = RECONS / f"exp2_1d_{cfg}.npy"
        if p.exists():
            ax.plot(xs, np.load(p), lw=0.8, color=COLORS.get(cfg),
                    label=LABELS.get(cfg, cfg))
    ax.plot(samples[:, 0], samples[:, 1], ".", ms=2, color="#999",
            label="samples")
    ax.set_xlim(-1.0, -0.55)  # sparse region: where methods differ
    ax.set_xlabel("$x$ (sparse region)")
    ax.set_ylabel("$f(x)$")
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("1D scattered fit, sparse region")

    recs = load("exp2_1d")
    fin = finals(recs, keys=("psnr", "hf_err"))
    order = [c for c in EXP2_ORDER if c in fin]
    ax = axes[1]
    m = [fin[c]["psnr"][0] for c in order]
    s = [fin[c]["psnr"][1] for c in order]
    ax.bar(range(len(order)), m, yerr=s,
           color=[COLORS.get(c) for c in order])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABELS.get(c, c).replace(" (ours)", "") for c in order],
                       rotation=15, fontsize=6.5)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("1D scattered: final PSNR")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_exp2_1d.pdf")
    plt.close(fig)


# ---------------------------------------------------------- fig: ablations

def fig_ablations():
    recs = load("ablations")
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.3))

    def collect(prefix):
        pts = defaultdict(list)
        for r in recs:
            if r["tag"].startswith(prefix):
                val = r["tag"].split("_", 1)[1]
                pts[float(val)].append(r["history"][-1]["psnr"])
        return pts

    # the eps_0.15 runs are the all-defaults reference (L=16, beta=0.05)
    default_runs = collect("eps_").get(0.15, [])

    def sweep(prefix, xlabel, ax, default_x=None):
        pts = collect(prefix)
        if default_x is not None and default_runs:
            pts[default_x] = pts.get(default_x, []) + list(default_runs)
        xs = sorted(pts)
        m = [np.mean(pts[x]) for x in xs]
        s = [np.std(pts[x]) for x in xs]
        ax.errorbar(xs, m, yerr=s, marker="o", ms=3, color="#d62728", capsize=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PSNR (dB)")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=7)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    sweep("eps_", r"bandwidth scale $\varepsilon/\sigma_y$", axes[0])
    sweep("levels_", "number of levels $L$", axes[1], default_x=16)
    sweep("beta_", r"weight $\beta$", axes[2], default_x=0.05)

    # sample budget: 3 configs, all points from within the ablation suite
    ax = axes[3]
    for cfg in ("mse", "ffl_interp", "kacrice_est"):
        pts = defaultdict(list)
        for r in recs:
            if r["tag"].startswith("n") and r["config"] == cfg:
                pts[r["n_samples"]].append(r["history"][-1]["psnr"])
        xs = sorted(pts)
        ax.errorbar(xs, [np.mean(pts[x]) for x in xs],
                    yerr=[np.std(pts[x]) for x in xs], marker="o", ms=3,
                    label=LABELS.get(cfg, cfg), color=COLORS.get(cfg), capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel("sample budget $N$")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ablations.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ tables

def fmt(mean_std):
    m, s = mean_std
    return f"{m:.2f} $\\pm$ {s:.2f}"


def tab_exp1():
    rows_1d = finals(load("exp1_1d"), keys=("psnr",))
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & \\multicolumn{1}{c}{1D multisine} & \\multicolumn{3}{c}{2D camera $128^2$} \\\\",
        "\\cmidrule(lr){2-2}\\cmidrule(lr){3-5}",
        "Config & PSNR & PSNR & HF-PSNR & SSIM \\\\",
        "\\midrule",
    ]
    r2 = defaultdict(dict)
    for img in ("camera",):
        r2 = finals([r for r in load("exp1_2d") if r["image"] == img])
    order = ["pemlp_mse", "pemlp_ffl", "pemlp_sobolev", "pemlp_kacrice",
             "siren_mse", "finer_mse"]
    for cfg in order:
        if cfg not in rows_1d and cfg not in r2:
            continue
        c1 = fmt(rows_1d[cfg]["psnr"]) if cfg in rows_1d else "--"
        if cfg in r2:
            c2, c3 = fmt(r2[cfg]["psnr"]), fmt(r2[cfg]["hf_psnr"])
            c4 = fmt(r2[cfg]["ssim"])
        else:
            c2 = c3 = c4 = "--"
        name = LABELS.get(cfg, cfg).replace("+ ", "$+$ ")
        lines.append(f"{name} & {c1} & {c2} & {c3} & {c4} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABS / "tab_exp1.tex").write_text("\n".join(lines))


def tab_exp2():
    recs = load("exp2_2d")
    modes = ["blobs", "ramp", "uniform"]
    order = EXP2_ORDER + ["sobolev_oracle", "kacrice_oracle", "siren_mse",
                          "siren_ffl_interp", "siren_sobolev_est",
                          "siren_kacrice_est"]
    lines = [
        "\\begin{tabular}{l" + "cc" * len(modes) + "}",
        "\\toprule",
        " " + "".join(
            f"& \\multicolumn{{2}}{{c}}{{{m}}} " for m in modes) + "\\\\",
        "".join(f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(modes))),
        "Config " + "& PSNR & HF-PSNR " * len(modes) + "\\\\",
        "\\midrule",
    ]
    per_mode = {
        m: finals([r for r in recs if r["image"] == "camera" and r["mode"] == m])
        for m in modes
    }
    # griddata interpolation reference row
    interp_row = "griddata interp.\\ (input) "
    for m in modes:
        rs = [r for r in recs if r["image"] == "camera" and r["mode"] == m]
        ps = list({r["seed"]: r["interp"]["psnr"] for r in rs}.values())
        hs = list({r["seed"]: r["interp"]["hf_psnr"] for r in rs}.values())
        interp_row += (f"& {np.mean(ps):.2f} $\\pm$ {np.std(ps):.2f} "
                       f"& {np.mean(hs):.2f} $\\pm$ {np.std(hs):.2f} ")
    lines.append(interp_row + "\\\\")
    lines.append("\\midrule")
    for cfg in order:
        row = LABELS.get(cfg, cfg).replace("+ ", "$+$ ") + " "
        seen = False
        for m in modes:
            f_ = per_mode[m]
            if cfg in f_:
                row += f"& {fmt(f_[cfg]['psnr'])} & {fmt(f_[cfg]['hf_psnr'])} "
                seen = True
            else:
                row += "& -- & -- "
        if seen:
            lines.append(row + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABS / "tab_exp2.tex").write_text("\n".join(lines))


def tab_synthetic():
    recs = [r for r in load("exp2_2d") if r["image"] == "synthetic"]
    f_ = finals(recs)
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "Config & PSNR & HF-PSNR & SSIM \\\\", "\\midrule"]
    for cfg in EXP2_ORDER:
        if cfg in f_:
            lines.append(
                f"{LABELS.get(cfg, cfg).replace('+ ', '$+$ ')} & "
                f"{fmt(f_[cfg]['psnr'])} & {fmt(f_[cfg]['hf_psnr'])} & "
                f"{fmt(f_[cfg]['ssim'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABS / "tab_synthetic.tex").write_text("\n".join(lines))


def main():
    done, skipped = [], []
    for name, fn in [
        ("fig_method", fig_method), ("fig_validation", fig_validation),
        ("fig_exp1", fig_exp1), ("fig_exp2_curves", fig_exp2_curves),
        ("fig_bands", fig_bands), ("fig_recons", fig_recons),
        ("fig_exp2_1d", fig_exp2_1d), ("fig_ablations", fig_ablations),
        ("tab_exp1", tab_exp1), ("tab_exp2", tab_exp2),
        ("tab_synthetic", tab_synthetic),
    ]:
        try:
            fn()
            done.append(name)
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{name}: {type(e).__name__}: {e}")
    print("done:", ", ".join(done))
    for s in skipped:
        print("SKIP", s)


if __name__ == "__main__":
    main()
