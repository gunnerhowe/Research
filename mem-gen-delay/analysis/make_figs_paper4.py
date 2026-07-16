"""Figures for the emergence-forecasting paper (paper4/). All from committed artifacts."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "analysis/out6/figs"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=170)
    plt.close(fig)


# ---------- fig 2: the money figure — per-seed forecasts + blind gate ----------
r2 = json.load(open("analysis/out6/r2_scored.json"))
r5 = json.load(open("analysis/out6/r5_scored.json"))
fig, ax = plt.subplots(figsize=(6.4, 5.0))
tp = [p["t_pv"] for p in r2["per_seed"]]
te = [p["t_event"] for p in r2["per_seed"]]
ax.scatter(tp, te, s=34, color="#1f77b4", alpha=0.85, label="fleet (config A, n=30)")
tp5 = [r["t_pv"] for r in r5["rows"]]
te5 = [r["t_event"] for r in r5["rows"]]
for r in r5["rows"]:
    ax.plot([r["t_pv"], r["t_pv"]], [r["interval"][0], r["interval"][1]],
            color="#d62728", lw=2.4, alpha=0.55, zorder=1)
ax.scatter(tp5, te5, s=46, color="#d62728", marker="D", zorder=3,
           label="BLIND gate (config B, n=10)")
xs = np.array([min(tp + tp5) - 150, max(tp + tp5) + 150])
ax.plot(xs, xs + 975, "k--", lw=1.2,
        label="frozen rule: event = alarm + 975 (±150)")
ax.fill_between(xs, xs + 825, xs + 1125, color="k", alpha=0.07)
ax.set(xlabel="precursor alarm time  $t_{\\mathrm{pv}}$  (steps)",
       ylabel="emergence time  $t_{\\mathrm{event}}$  (steps)",
       title="Per-seed emergence forecasting: fleet fit + blind transfer")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
save(fig, "fig2_forecast")

# ---------- fig 3: example run — alarm, interval, event ----------
recs = [json.loads(l) for l in open("runs/grid6r5/rep_s101/metrics.jsonl") if l.strip()]
st = [r["step"] for r in recs]
fig, ax = plt.subplots(figsize=(6.6, 4.0))
ax.plot(st, [r["copy_adv"] for r in recs], color="#d62728", lw=2,
        label="copy advantage (capability)")
ax2 = ax.twinx()
ax2.plot(st, [r["prevtok_by_layer"][0] for r in recs], color="#2ca02c", lw=1.7, ls=":",
         label="prev-token head (precursor)")
ax2.plot(st, [max(r["prefix_by_layer"]) for r in recs], color="#1f77b4", lw=1.7, ls="--",
         label="induction score (circuit)")
t_pv, ev = 5375, 6325
ax.axvline(t_pv, color="#2ca02c", lw=1.4)
ax.text(t_pv - 120, 6.6, "ALARM", color="#2ca02c", fontsize=8, rotation=90, va="top")
ax.axvspan(t_pv + 825, t_pv + 1125, color="k", alpha=0.12)
ax.text(t_pv + 840, 8.4, "forecast\ninterval", fontsize=7)
ax.axvline(ev, color="#d62728", lw=1.4)
ax.text(ev + 60, 6.6, "EVENT", color="#d62728", fontsize=8, rotation=90, va="top")
ax.set(xlabel="training step", ylabel="copy advantage (nats)", xlim=(3000, 9000),
       title="A blind run (config B, seed 101): alarm → interval → emergence")
ax2.set_ylabel("attention score")
ax2.set_ylim(0, 1.05)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left")
fig.tight_layout()
save(fig, "fig3_examplerun")

# ---------- fig 4: the P5 tradeoff (MNIST) ----------
fz = json.load(open("analysis/out5/frozen_eval.json"))
r1r2 = json.load(open("analysis/out5/r1r2_stats.json"))
fig, ax = plt.subplots(figsize=(5.6, 3.8))
labels = ["weight norm\n(scalar clock)", "multivariate\n(FA-controlled)",
          "content probe\n(cos gap slope)"]
inl = [1000, 10400, 4000]
shl = [0, 0, 16400]
x = np.arange(3)
ax.bar(x - 0.2, inl, 0.38, color="#1f77b4", label="in-distribution")
ax.bar(x + 0.2, shl, 0.38, color="#d62728", label="under intervention shift")
ax.set_xticks(x, labels, fontsize=8)
ax.set_ylabel("median warning time (steps)")
ax.set_title("P5 (MNIST grokking): the robustness–false-alarm tradeoff")
ax.legend(fontsize=8)
for xi, v in zip(x - 0.2, inl):
    ax.text(xi, v + 250, f"{v:,}", ha="center", fontsize=7)
for xi, v in zip(x + 0.2, shl):
    ax.text(xi, v + 250, f"{v:,}", ha="center", fontsize=7)
fig.tight_layout()
save(fig, "fig4_tradeoff")
print("figs written to", OUT)
