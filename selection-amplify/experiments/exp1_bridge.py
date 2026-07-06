"""E1 -- BRIDGE ON 2D (early test of K3 + K3b).

Confirm the guided bridge puts synthesized points in the collar-complement
(on-manifold high-density, low true selection, near observed support) and NOT
off-manifold, and that selection guidance beats the misdirected decoy and the
selection-agnostic B2 at matched budget on the complement-hit metric.

GATE (pre-registered, PLAN.md): method complement-hit-rate > decoy AND > B2 at
matched budget (paired over seeds), AND method off-manifold reject-rate < 0.15,
AND guidance is confined to the low-uncertainty collar (unc-gating active, not
steering deep). Emits results/exp1_bridge.json.
"""
from __future__ import annotations

import numpy as np

import common as C
from selamp import data
from selamp.bridge import generate_labeled

BETAS = [1.0, 2.0, 4.0, 8.0]
SEEDS = [0, 1, 2, 3, 4]


def synth_metrics(st, X):
    beta, tb = st.beta, st.testbed
    s_true = data.selection_prob(X, beta, tb)
    v = st.val.score(X)
    on = ~v["reject_mask"]
    prox = st.sel.proximity(X)
    return {
        "complement_hit": float((on & (s_true < 0.5)).mean()),
        "reject_rate": v["reject_rate"],
        "mean_s_true": float(s_true.mean()),
        "in_slice": float((data.phi_std(X, tb) > 0.5).mean()),
        "mean_prox": float(prox.mean()),
        "frac_near_support": float((prox < st.cfg.d_max).mean()),
    }


def one(beta, seed):
    st = C.Stack(C.PRIMARY, beta, seed)
    npc = C.N_SYNTH_PER_CLASS
    Xm, _, dg = generate_labeled(st.dm, st.sel, st.gate, npc, st.cfg,
                                 decoy=None, seed=seed)
    Xd, _, _ = generate_labeled(st.dm, st.sel, st.gate, npc, st.cfg,
                                decoy="rotate", seed=seed)
    Xb2 = np.concatenate([st.dm.sample(npc, k, seed=seed + k) for k in range(2)])
    row = dict(beta=beta, seed=seed,
               method=synth_metrics(st, Xm),
               decoy=synth_metrics(st, Xd),
               b2=synth_metrics(st, Xb2),
               guided_frac=dg[0].guided_frac,
               guide_norm=dg[0].mean_guide_norm,
               base_norm=dg[0].mean_base_norm,
               frac_veto=dg[0].frac_veto,
               frac_unc=dg[0].frac_unc_gate,
               frac_prox=dg[0].frac_prox_gate,
               frac_capped=dg[0].frac_capped)
    return row


def run():
    rows = []
    for beta in BETAS:
        for seed in SEEDS:
            rows.append(one(beta, seed))
        sub = [r for r in rows if r["beta"] == beta]
        mh = np.mean([r["method"]["complement_hit"] for r in sub])
        dh = np.mean([r["decoy"]["complement_hit"] for r in sub])
        bh = np.mean([r["b2"]["complement_hit"] for r in sub])
        rej = np.mean([r["method"]["reject_rate"] for r in sub])
        print(f"beta={beta:>3} | complement-hit method={mh:.3f} decoy={dh:.3f} "
              f"b2={bh:.3f} | method_reject={rej:.3f} "
              f"unc_gate={np.mean([r['frac_unc'] for r in sub]):.2f}")
    C.save("exp1_bridge.json", {"rows": rows, "stamp": C.stamp()})


if __name__ == "__main__":
    run()
