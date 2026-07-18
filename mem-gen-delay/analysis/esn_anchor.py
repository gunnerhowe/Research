"""R-ESN: reservoir-computing generic anchors over training trajectories (CPU only).

Prereg: plan_p6.md commit c012d5c. Two-stage discipline:
  python analysis/esn_anchor.py --calibrate   # fit readouts (grid6r2 even), tau (odd),
                                              # write analysis/out6/esn_frozen.json
  python analysis/esn_anchor.py --test        # ONE-SHOT scoring of all frozen test cells
                                              # (gates B/C, gap cells, law cell, THE TRAP)

Reservoir (frozen): leaky ESN N=500, spectral radius 0.9, leak 0.3, 90% sparse, seed 7.
Cells: NAIVE {train_loss, copy_adv, indist_adv} | FULL {+prevtok x2, prefix x2} |
LOSSONLY {train_loss}. Ridge readout on within-W labels (W=3000), alpha by grouped
5-fold on the fit set; alarm = score >= tau on 2 consecutive probes, warmup step >= 125.
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np

OUT = "analysis/out6"
W_HORIZON = 3000
WARMUP_STEP = 125
SUSTAIN = 2
N_RES = 500
RHO = 0.9
LEAK = 0.3
SPARSITY = 0.9
SEED = 7
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]
CELLS = {
    "NAIVE": ["train_loss", "copy_adv", "indist_adv"],
    "FULL": ["train_loss", "copy_adv", "indist_adv",
             "prevtok0", "prevtok1", "prefix0", "prefix1"],
    "LOSSONLY": ["train_loss"],
}


def load_run(path):
    summ = json.load(open(f"{path}/summary.json"))
    recs = [json.loads(l) for l in open(f"{path}/metrics.jsonl") if l.strip()]
    ch = {
        "step": np.array([r["step"] for r in recs], float),
        "train_loss": np.array([r["train_loss"] for r in recs], float),
        "copy_adv": np.array([r["copy_adv"] for r in recs], float),
        "indist_adv": np.array([r["indist_adv"] for r in recs], float),
        "prevtok0": np.array([r["prevtok_by_layer"][0] for r in recs], float),
        "prevtok1": np.array([r["prevtok_by_layer"][-1] for r in recs], float),
        "prefix0": np.array([r["prefix_by_layer"][0] for r in recs], float),
        "prefix1": np.array([r["prefix_by_layer"][-1] for r in recs], float),
    }
    return dict(name=os.path.basename(path), t_event=summ["t_event"], ch=ch)


def corpus(grid, pat="*"):
    return [load_run(d) for d in sorted(glob.glob(f"runs/{grid}/{pat}"))
            if os.path.exists(f"{d}/summary.json")]


def reservoir_weights(n_in):
    g = np.random.default_rng(SEED)
    W = g.normal(0, 1, (N_RES, N_RES))
    W[g.random((N_RES, N_RES)) < SPARSITY] = 0.0
    eig = np.max(np.abs(np.linalg.eigvals(W)))
    W *= RHO / max(eig, 1e-9)
    W_in = g.uniform(-1, 1, (N_RES, n_in))
    return W.astype(np.float32), W_in.astype(np.float32)


def states_for(run, chans, mu, sd, W, W_in):
    U = np.stack([(run["ch"][c] - mu[c]) / sd[c] for c in chans], 1).astype(np.float32)
    x = np.zeros(N_RES, np.float32)
    out = np.empty((len(U), N_RES), np.float32)
    for t in range(len(U)):
        x = (1 - LEAK) * x + LEAK * np.tanh(W @ x + W_in @ U[t])
        out[t] = x
    return out


def zstats(runs, chans):
    mu, sd = {}, {}
    for c in chans:
        v = np.concatenate([r["ch"][c] for r in runs])
        mu[c], sd[c] = float(v.mean()), float(v.std() + 1e-9)
    return mu, sd


def labels_for(run):
    steps = run["ch"]["step"]
    te = run["t_event"]
    if te is None:
        return np.zeros(len(steps)), np.ones(len(steps), bool)
    y = ((te - steps > 0) & (te - steps <= W_HORIZON)).astype(float)
    keep = steps <= te
    return y, keep


def fit_ridge(X, y, alpha):
    Xb = np.hstack([X, np.ones((len(X), 1), np.float32)])
    A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, Xb.T @ y.astype(np.float32))
    return w


def predict(w, X):
    return np.hstack([X, np.ones((len(X), 1), np.float32)]) @ w


def alarm_time(steps, scores, tau):
    streak = 0
    for i in range(len(steps)):
        if steps[i] < WARMUP_STEP:
            continue
        if scores[i] >= tau:
            streak += 1
            if streak >= SUSTAIN:
                return float(steps[i])
        else:
            streak = 0
    return None


def score_cellset(runs, chans, mu, sd, W, W_in, w, tau):
    leads, cs, miss, fa, npos, nneg = [], [], 0, 0, 0, 0
    per = []
    for r in runs:
        s = predict(w, states_for(r, chans, mu, sd, W, W_in))
        ta = alarm_time(r["ch"]["step"], s, tau)
        if r["t_event"] is None:
            nneg += 1
            fa += int(ta is not None)
            per.append(dict(name=r["name"], t_event=None, alarm=ta))
        else:
            npos += 1
            lead = 0.0 if (ta is None or ta >= r["t_event"]) else r["t_event"] - ta
            miss += int(ta is None or ta >= r["t_event"])
            leads.append(lead)
            if ta is not None and ta < r["t_event"]:
                cs.append(ta / r["t_event"])
            per.append(dict(name=r["name"], t_event=r["t_event"], alarm=ta, lead=lead))
    return dict(median_lead=float(np.median(leads)) if leads else 0.0,
                miss=f"{miss}/{npos}", fa=f"{fa}/{nneg}" if nneg else "0/0",
                median_c=float(np.median(cs)) if cs else None,
                n_pos=npos, n_neg=nneg, per_run=per)


def calibrate():
    os.makedirs(OUT, exist_ok=True)
    cal = corpus("grid6r2")
    fit_runs = [r for r in cal if int(r["name"].split("_s")[1]) % 2 == 0]
    tau_runs = [r for r in cal if int(r["name"].split("_s")[1]) % 2 == 1]
    frozen = {"spec": dict(N=N_RES, rho=RHO, leak=LEAK, sparsity=SPARSITY, seed=SEED,
                           W=W_HORIZON, warmup=WARMUP_STEP, sustain=SUSTAIN)}
    for cell, chans in CELLS.items():
        mu, sd = zstats(fit_runs, chans)
        W, W_in = reservoir_weights(len(chans))
        whash = hashlib.sha256(W.tobytes() + W_in.tobytes()).hexdigest()[:16]
        Xs, ys = [], []
        groups = []
        for gi, r in enumerate(fit_runs):
            X = states_for(r, chans, mu, sd, W, W_in)
            y, keep = labels_for(r)
            Xs.append(X[keep])
            ys.append(y[keep])
            groups.append(np.full(int(keep.sum()), gi))
        X, y, G = np.vstack(Xs), np.concatenate(ys), np.concatenate(groups)
        best = None
        for a in ALPHAS:
            errs = []
            for f in range(5):
                tr = G % 5 != f
                w = fit_ridge(X[tr], y[tr], a)
                errs.append(float(np.mean((predict(w, X[~tr]) - y[~tr]) ** 2)))
            m = float(np.mean(errs))
            if best is None or m < best[0]:
                best = (m, a)
        alpha = best[1]
        w = fit_ridge(X, y, alpha)
        # tau on odd runs: max median lead s.t. zero alarms on odd negatives
        neg_scores, cand = [], []
        for r in tau_runs:
            s = predict(w, states_for(r, chans, mu, sd, W, W_in))
            if r["t_event"] is None:
                neg_scores.append(s[r["ch"]["step"] >= WARMUP_STEP])
            cand.append(s)
        floor = max(float(np.max(np.concatenate(neg_scores))), 0.0) if neg_scores else 0.0
        taus = np.unique(np.quantile(np.concatenate(cand), np.linspace(0.5, 0.999, 60)))
        taus = taus[taus > floor]
        pick = None
        for tau in list(taus) + [floor + 1e-6]:
            sc = score_cellset(tau_runs, chans, mu, sd, W, W_in, w, float(tau))
            if sc["fa"].split("/")[0] != "0":
                continue
            if pick is None or sc["median_lead"] > pick[1]["median_lead"]:
                pick = (float(tau), sc)
        frozen[cell] = dict(chans=chans, alpha=alpha, tau=pick[0], mu=mu, sd=sd,
                            weights_sha=whash, readout=w.tolist(),
                            calibration_card={k: v for k, v in pick[1].items()
                                              if k != "per_run"})
        print(f"{cell}: alpha={alpha} tau={pick[0]:.4f} "
              f"odd-seed card={frozen[cell]['calibration_card']}")
    json.dump(frozen, open(f"{OUT}/esn_frozen.json", "w"))
    print("FROZEN -> analysis/out6/esn_frozen.json (commit before --test)")


def run_sealed_test(cells_runs, out_name):
    frozen = json.load(open(f"{OUT}/esn_frozen.json"))
    if os.path.exists(f"{OUT}/{out_name}"):
        raise RuntimeError(f"{out_name} exists — one-shot test already ran")
    res = {}
    for cell in CELLS:
        f = frozen[cell]
        W, W_in = reservoir_weights(len(f["chans"]))
        assert hashlib.sha256(W.tobytes() + W_in.tobytes()).hexdigest()[:16] == f["weights_sha"]
        w = np.array(f["readout"], np.float32)
        res[cell] = {}
        for name, runs in cells_runs.items():
            sc = score_cellset(runs, f["chans"], f["mu"], f["sd"], W, W_in, w, f["tau"])
            res[cell][name] = sc
            print(f"{cell:8s} {name:16s} lead={sc['median_lead']:>7.0f} "
                  f"miss={sc['miss']:>5s} FA={sc['fa']:>5s} c={sc['median_c']}")
    json.dump(res, open(f"{OUT}/{out_name}", "w"), indent=1)
    print(f"wrote analysis/out6/{out_name}")


def test():
    run_sealed_test({
        "gateB_grid6r5": corpus("grid6r5"), "gateC_grid6r7": corpus("grid6r7"),
        "gap_grid6r8": corpus("grid6r8"), "law_grid6r9": corpus("grid6r9"),
        "TRAP_grid6r6": corpus("grid6r6"),
    }, "esn_scored.json")


def test_b():
    """R-ESNb (prereg 4ca549e): fourth-corner + prospective cells vs the SAME frozen
    artifacts. Refuses to run until the full pre-registered fleet exists."""
    trapone = corpus("grid6esnb", "trapone_*")
    fresh_pos = corpus("grid6esnb", "fresh[0-9]")
    fresh_neg = corpus("grid6esnb", "freshn*")
    if not (len(trapone) == 10 and len(fresh_pos) == 5 and len(fresh_neg) == 5):
        raise RuntimeError(f"fleet incomplete: trapone={len(trapone)} "
                           f"pos={len(fresh_pos)} neg={len(fresh_neg)} — not scoring")
    run_sealed_test({
        "TRAPONE": trapone,
        "FRESH": fresh_pos + fresh_neg,
    }, "esn_scored_b.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--test_b", action="store_true")
    args = ap.parse_args()
    if args.calibrate:
        calibrate()
    elif args.test:
        test()
    elif args.test_b:
        test_b()
    else:
        raise SystemExit("pass --calibrate, --test, or --test_b")
