"""Interpretability analysis of learned SemRF anchors on the enwik8 checkpoint.

Produces:
  * fig_anchors        : PCA of token embeddings colored by anchor assignment.
  * fig_frame_time     : per-head, per-frame learned temporal-decay slopes, with
                         frames ordered by decay, illustrating that different
                         semantic frames receive different time sensitivities.
  * anchor_clusters.json : the characters assigned to each anchor (for the text).

    python -m scripts.anchor_analysis
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from semrf.config import ModelConfig
from semrf.model import TransformerLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "results", "charlm", "enwik8", "semrf__seed0.ckpt.pt")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)


def char_label(b):
    c = chr(b)
    if c == " ":
        return "␣"
    if c == "\n":
        return "\\n"
    return c if c.isprintable() else f"x{b:02x}"


def char_class(b):
    c = chr(b)
    if c.isalpha() and c.islower():
        return "lower"
    if c.isalpha() and c.isupper():
        return "upper"
    if c.isdigit():
        return "digit"
    if c.isspace():
        return "space"
    if 32 <= b < 127:
        return "punct"
    return "other"


def main():
    if not os.path.exists(CKPT):
        print(f"no checkpoint at {CKPT}; run the char-LM SemRF seed-0 job first.")
        return
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_cfg"])
    model = TransformerLM(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    pos = model.pos
    assert pos.__class__.__name__ == "SemRF"

    V = cfg.vocab_size
    ids = torch.arange(V)
    with torch.no_grad():
        emb = model.tok_emb(ids)                      # (V, d)
        u = pos.to_u(emb)                             # (V, da)
        sim = u @ pos.anchors.t()                     # (V, K)
        assign = sim.argmax(-1).numpy()               # (V,)

    # bytes that actually occur in enwik8 (printable + common control)
    occurring = [b for b in range(V) if 32 <= b < 127 or b in (9, 10, 13)]

    # ---- cluster listing ----
    clusters = {}
    for b in occurring:
        clusters.setdefault(int(assign[b]), []).append(char_label(b))
    clusters = {k: v for k, v in sorted(clusters.items())}
    with open(os.path.join(ROOT, "results", "anchor_clusters.json"), "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    print(f"{len(clusters)} anchors used by {len(occurring)} occurring bytes")
    for k, v in list(clusters.items())[:12]:
        print(f"  anchor {k:2d}: {' '.join(v[:28])}")

    # ---- PCA of embeddings colored by anchor ----
    from sklearn.decomposition import PCA
    sub = np.array(occurring)
    X = emb[sub].detach().numpy()
    xy = PCA(n_components=2).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=assign[sub], cmap="tab20", s=140, alpha=0.85,
                    edgecolors="white", linewidths=0.5)
    for i, b in enumerate(sub):
        ax.annotate(char_label(b), (xy[i, 0], xy[i, 1]), fontsize=7, ha="center", va="center")
    ax.set_title("Token embeddings colored by SemRF anchor assignment (enwik8)")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_anchors.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_anchors")

    # ---- frame-conditioned temporal slopes ----
    if pos.use_time:
        slope = torch.nn.functional.softplus(pos.time_slope).detach().numpy()  # (H, K)
        # restrict to anchors that are actually used, order by mean slope
        used = sorted(clusters.keys())
        S = slope[:, used]
        order = np.argsort(S.mean(0))
        S = S[:, order]
        used_ord = [used[i] for i in order]
        fig, axes = plt.subplots(1, 2, figsize=(max(11, 0.8 * len(used)), 3.8))
        im = axes[0].imshow(S, aspect="auto", cmap="viridis")
        axes[0].set_xlabel("anchor (semantic frame), ordered by mean decay")
        axes[0].set_ylabel("attention head")
        axes[0].set_xticks(range(len(used_ord)))
        axes[0].set_xticklabels(used_ord, fontsize=7)
        axes[0].set_title("Decay slopes $\\mathrm{softplus}(s_{hk})$")
        fig.colorbar(im, ax=axes[0], label="slope")
        # normalized: each head row / its own mean -> per-frame relative deviation
        Sn = S / S.mean(axis=1, keepdims=True)
        im2 = axes[1].imshow(Sn, aspect="auto", cmap="RdBu_r", vmin=0.7, vmax=1.3)
        axes[1].set_xlabel("anchor (semantic frame), ordered by mean decay")
        axes[1].set_ylabel("attention head")
        axes[1].set_xticks(range(len(used_ord)))
        axes[1].set_xticklabels(used_ord, fontsize=7)
        axes[1].set_title("Relative to head mean (frame differentiation)")
        fig.colorbar(im2, ax=axes[1], label="slope / head mean")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIG, f"fig_frame_time.{ext}"), bbox_inches="tight")
        plt.close(fig)
        rel_spread = (Sn.max(1) - Sn.min(1))
        print(f"wrote fig_frame_time; within-head relative spread per head: "
              + " ".join(f"{x:.0%}" for x in rel_spread))

        # a compact summary: slowest vs fastest decaying frames (by chars)
        mean_slope = S.mean(0)
        summary = {
            "slowest_frames": [(used_ord[i], clusters[used_ord[i]][:12]) for i in np.argsort(mean_slope)[:3]],
            "fastest_frames": [(used_ord[i], clusters[used_ord[i]][:12]) for i in np.argsort(mean_slope)[-3:]],
        }
        print("slowest-decay frames (retain long context):")
        for k, chars in summary["slowest_frames"]:
            print(f"  anchor {k}: {' '.join(chars)}")
        print("fastest-decay frames (local):")
        for k, chars in summary["fastest_frames"]:
            print(f"  anchor {k}: {' '.join(chars)}")
        with open(os.path.join(ROOT, "results", "frame_time_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
