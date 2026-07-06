"""E2 -- THE GO/NO-GO CURVE.

Full beta sweep x >=3 seeds. Downstream classifier on D_obs vs D_obs U
synthesized; evaluate on frozen full test AND the censored slice. The
discriminating quantity is the METHOD-MINUS-MISDIRECTED-SELECTOR gap on the
censored slice vs I(O;X) -- never the method's own curve (its rise is trivial:
more MNAR = more headroom). Baselines B0-B4 plus the sharp permuted-selector
control and the diversity-matched B2.

Emits results/exp2_curve.json; gates K1/K2 are adjudicated in
gen_paper_numbers.py / make_figures.py from these rows.
"""
from __future__ import annotations

import numpy as np

import common as C
from selamp import data
from selamp.bridge import generate_labeled
from selamp.downstream import match_temperature, smote_low_density, train_eval
from selamp.entropy import mutual_info_OX


def _oracle_set(testbed, beta, seed, n):
    X, y = data.TESTBEDS[testbed](n, seed=50000 + seed)   # fresh full population
    return X, y


def one(testbed, beta, seed):
    st = C.Stack(testbed, beta, seed, verbose=True)
    c, sel, dm, gate, cfg = st.c, st.sel, st.dm, st.gate, st.cfg
    npc = C.N_SYNTH_PER_CLASS
    slice_mask = c.slice_mask

    # ---- generators ----
    Xm, ym, dgm = generate_labeled(dm, sel, gate, npc, cfg, decoy=None, seed=seed)
    Xd, yd, dgd = generate_labeled(dm, sel, gate, npc, cfg, decoy="rotate", seed=seed)
    # B2: selection-agnostic (gamma=0) unconditional generation
    Xb2 = np.concatenate([dm.sample(npc, k, seed=seed + k) for k in range(2)])
    yb2 = np.concatenate([np.full(npc, k) for k in range(2)]).astype(int)
    # B2-div: temperature-matched to the guided sampler's spread
    Xb2d, yb2d, T_used, _ = match_temperature(dm, sel, gate, Xm, npc, seed)
    # B1: SMOTE of low-selection observed points
    s_hat_obs = sel.s_hat(c.X_obs)
    Xb1, yb1 = smote_low_density(c.X_obs, c.y_obs, s_hat_obs, 2 * npc, seed=seed)
    # B4 oracle: fresh full-population labels, matched budget
    Xor, yor = _oracle_set(testbed, beta, seed, len(c.X_obs) + 2 * npc)

    def aug(Xs, ys):
        return (np.concatenate([c.X_obs, Xs]), np.concatenate([c.y_obs, ys]))

    train_sets = {
        "B0_obs": (c.X_obs, c.y_obs, None),
        "B1_smote": (*aug(Xb1, yb1), None),
        "B2_uncond": (*aug(Xb2, yb2), None),
        "B2div_matched": (*aug(Xb2d, yb2d), None),
        "B3_reweight": (c.X_obs, c.y_obs, 1.0 / np.clip(s_hat_obs, 0.05, 1.0)),
        "B4_oracle": (Xor, yor, None),
        "method": (*aug(Xm, ym), None),
        "decoy_rotate": (*aug(Xd, yd), None),
    }

    rows = []
    for name, (Xtr, ytr, w) in train_sets.items():
        ev = train_eval(Xtr, ytr, c.X_test, c.y_test, slice_mask,
                        sample_weight=w, seed=seed)
        rows.append(dict(testbed=testbed, beta=beta, seed=seed, method=name,
                         **ev))

    # I(O;X) axis for this beta (from the TRUE selector over a large pop sample)
    Xbig, _ = data.TESTBEDS[testbed](40000, seed=7)
    iox = mutual_info_OX(data.selection_prob(Xbig, beta, testbed))

    # synth diagnostics (E1 / K3 / K3b)
    vm, vd = st.val.score(Xm), st.val.score(Xd)
    s_m = data.selection_prob(Xm, beta, testbed).mean()
    s_d = data.selection_prob(Xd, beta, testbed).mean()
    diag = dict(testbed=testbed, beta=beta, seed=seed, method="_diag", iox=iox,
                T_b2div=T_used,
                method_reject=vm["reject_rate"], decoy_reject=vd["reject_rate"],
                method_mean_s=float(s_m), decoy_mean_s=float(s_d),
                method_guided_frac=dgm[0].guided_frac,
                method_guide_norm=dgm[0].mean_guide_norm,
                method_base_norm=dgm[0].mean_base_norm,
                method_frac_veto=dgm[0].frac_veto,
                method_frac_unc=dgm[0].frac_unc_gate,
                method_frac_prox=dgm[0].frac_prox_gate,
                method_in_slice=float((data.phi_std(Xm, testbed) > 0.5).mean()),
                decoy_in_slice=float((data.phi_std(Xd, testbed) > 0.5).mean()),
                tau_log=cfg.tau_log, veto_log=cfg.veto_log,
                u_max=cfg.u_max, d_max=cfg.d_max)
    rows.append(diag)
    return rows


def run(testbed=C.PRIMARY, seeds=C.HEADLINE_SEEDS):
    all_rows = []
    for beta in C.BETAS:
        for seed in seeds:
            all_rows += one(testbed, beta, seed)
        # progress: method vs decoy vs B0 on the slice
        d = [r for r in all_rows if r["beta"] == beta and r["method"] == "_diag"]
        def m(name, key):
            v = [r[key] for r in all_rows if r["beta"] == beta
                 and r["method"] == name]
            return np.nanmean(v)
        print(f"beta={beta:>3} IOX={np.mean([x['iox'] for x in d]):.3f} | "
              f"slice B0={m('B0_obs','acc_slice'):.3f} "
              f"method={m('method','acc_slice'):.3f} "
              f"decoy={m('decoy_rotate','acc_slice'):.3f} "
              f"B3={m('B3_reweight','acc_slice'):.3f} "
              f"B4={m('B4_oracle','acc_slice'):.3f} | "
              f"gap={m('method','acc_slice')-m('decoy_rotate','acc_slice'):+.4f}")
    C.save(f"exp2_curve.json", {"rows": all_rows, "stamp": C.stamp(),
                                "testbed": testbed})


if __name__ == "__main__":
    run()
