"""P5: pooled multivariate forecaster (the pre-registered SECONDARY model, plan_p5.md).

Motivation from diag_p5_algzero.py: single-signal alarms die on two trap types --
structure-complete-but-norm-trapped negatives (supcon@c35: fourier 0.996, never groks) and
wrong-structure look-alikes (band arms). The mechanism papers say generalization requires
structure AND a viable norm regime, so the forecaster is given both: logistic regression on
S, R, and S+R feature sets (levels + slopes), per domain.

Protocol identical to the threshold alarms: train on even seeds, alarm when P(grok within W)
>= tau with tau fit on train at FA <= 5% over negative runs, report TEST (odd seeds) median
lead. R2 variant: train on control-family even seeds, test on prior arms. W = 0.2 x median
t_gen of train positives per domain (prereg). Missing features imputed with train means
(disclosed). Numpy logistic (GD), no sklearn dependency.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_emergence import load_corpus, add_slopes, WARMUP, SIG_S, SIG_R

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "analysis", "out5")
FA_CAP = 0.05


def featnames(domain, which):
    base = {"S": SIG_S[domain], "R": SIG_R[domain],
            "SR": SIG_S[domain] + SIG_R[domain]}[which]
    names = [s for b in base for s in (b, "d." + b)]
    if which.endswith("q"):
        pass  # handled by caller passing base set name
    return names


def quad_names(domain, which):
    """Mechanism-motivated nonlinear features: the viable-norm window is a band-pass in
    wnorm (Papers 1-3: too low = inversion trap, too high = delay-law censoring), which a
    linear model cannot express. Add wnorm^2 and structure-x-norm interactions."""
    names = featnames(domain, which)
    out = list(names)
    if "wnorm" in names:
        out.append("wnorm^2")
        for r in SIG_R[domain]:
            if r in names:
                out += [f"{r}*wnorm", f"{r}*wnorm^2"]
    return out


def run_matrix(run, names):
    cols = []
    n = len(run["times"])
    base = run["_sigs"]
    for k in names:
        if k == "wnorm^2":
            w = base.get("wnorm")
            cols.append(w ** 2 if w is not None else np.full(n, np.nan))
        elif "*" in k:
            a, b = k.split("*", 1)
            va = base.get(a)
            vb = base.get("wnorm")
            if va is None or vb is None:
                cols.append(np.full(n, np.nan))
            else:
                cols.append(va * (vb ** 2 if b == "wnorm^2" else vb))
        else:
            v = base.get(k)
            cols.append(v if v is not None else np.full(n, np.nan))
    return np.stack(cols, 1)  # (T, F)


def build_dataset(runs, names, W):
    X, y = [], []
    for r in runs:
        M = run_matrix(r, names)
        t = r["times"]
        for i in range(WARMUP, len(t)):
            X.append(M[i])
            if r["t_gen"] is None:
                y.append(0.0)
            else:
                y.append(1.0 if (r["t_gen"] - t[i]) <= W and t[i] <= r["t_gen"] else 0.0)
    return np.array(X), np.array(y)


def fit_logistic(X, y, l2=1e-3, iters=400, lr=0.5):
    mu = np.nanmean(X, 0)
    sd = np.nanstd(X, 0) + 1e-9
    Xz = (np.where(np.isnan(X), mu, X) - mu) / sd
    Xz = np.hstack([Xz, np.ones((len(Xz), 1))])
    w = np.zeros(Xz.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xz @ w))
        g = Xz.T @ (p - y) / len(y) + l2 * w
        w -= lr * g
    return w, mu, sd


def predict_run(run, names, w, mu, sd):
    M = run_matrix(run, names)
    Mz = (np.where(np.isnan(M), mu, M) - mu) / sd
    Mz = np.hstack([Mz, np.ones((len(Mz), 1))])
    return 1 / (1 + np.exp(-Mz @ w))


def alarm_lead(run, p, tau):
    for i in range(WARMUP, len(p)):
        if p[i] >= tau:
            ta = run["times"][i]
            if run["t_gen"] is None:
                return None, True          # false alarm on a negative
            return max(0.0, run["t_gen"] - ta), False
    return (0.0, False) if run["t_gen"] is not None else (None, False)


def evaluate(runs, names, w, mu, sd, tau):
    leads, rel, miss, fa, npos, nneg = [], [], 0, 0, 0, 0
    for r in runs:
        p = predict_run(r, names, w, mu, sd)
        lead, false_alarm = alarm_lead(r, p, tau)
        if r["t_gen"] is None:
            nneg += 1
            fa += int(false_alarm)
        else:
            npos += 1
            leads.append(lead)
            rel.append(lead / r["t_gen"])
            miss += int(lead == 0.0 and p[WARMUP:].max() < tau)
    return dict(median_lead=float(np.median(leads)) if leads else 0.0,
                median_rel=float(np.median(rel)) if rel else 0.0,
                miss_rate=miss / npos if npos else None,
                fa_rate=fa / nneg if nneg else None, n_pos=npos, n_neg=nneg)


def fit_tau(train, names, w, mu, sd):
    taus = np.linspace(0.5, 0.999, 60)
    best = None
    for tau in taus:
        sc = evaluate(train, names, w, mu, sd, tau)
        if sc["fa_rate"] is not None and sc["fa_rate"] > FA_CAP:
            continue
        key = sc["median_lead"]
        if best is None or key > best[0]:
            best = (key, tau, sc)
    return best


def block(title, train, test, domain, results, tag):
    tg = [r["t_gen"] for r in train if r["t_gen"] is not None]
    W = 0.2 * float(np.median(tg))
    print(f"\n{'=' * 96}\n{title} [{domain}] W={W:.0f}  train={len(train)} test={len(test)}\n{'=' * 96}")
    print(f"{'set':4s} {'tau':>6s} {'trainLead':>10s} {'TESTlead':>10s} {'rel':>7s} "
          f"{'miss':>6s} {'FA':>6s} {'nPos':>5s} {'nNeg':>5s}")
    out = {}
    for which in ("S", "R", "SR"):
        names = featnames(domain, which)
        X, y = build_dataset(train, names, W)
        w, mu, sd = fit_logistic(X, y)
        fit = fit_tau(train, names, w, mu, sd)
        if fit is None:
            print(f"{which:4s}  no tau meets FA cap on train")
            continue
        _, tau, tr = fit
        te = evaluate(test, names, w, mu, sd, tau)
        out[which] = dict(tau=round(float(tau), 4),
                          train_lead=round(tr["median_lead"], 1),
                          test_lead=round(te["median_lead"], 1),
                          test_rel=round(te["median_rel"], 4),
                          miss=te["miss_rate"], fa=te["fa_rate"],
                          n_pos=te["n_pos"], n_neg=te["n_neg"],
                          weights={n: round(float(v), 3) for n, v in
                                   zip(names + ["bias"], w)})
        print(f"{which:4s} {tau:>6.3f} {tr['median_lead']:>10.0f} {te['median_lead']:>10.0f} "
              f"{te['median_rel']:>7.3f} "
              f"{te['miss_rate'] if te['miss_rate'] is not None else float('nan'):>6.2f} "
              f"{te['fa_rate'] if te['fa_rate'] is not None else float('nan'):>6.2f} "
              f"{te['n_pos']:>5d} {te['n_neg']:>5d}")
    results[tag] = out


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = load_corpus()
    for r in runs:
        r["_sigs"] = add_slopes(r)
    results = {}
    for domain in ("alg", "mnist"):
        dom = [r for r in runs if r["domain"] == domain]
        block("R1 multivariate (train=even, test=odd)",
              [r for r in dom if r["seed"] % 2 == 0],
              [r for r in dom if r["seed"] % 2 == 1], domain, results, f"r1_{domain}")
        block("R2 multivariate shift (train=control even, test=PRIOR arms)",
              [r for r in dom if r["family"] == "control" and r["seed"] % 2 == 0],
              [r for r in dom if r["family"] == "prior"], domain, results, f"r2_{domain}")
    json.dump(results, open(os.path.join(OUT, "ml_stats.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(OUT, 'ml_stats.json')}")


if __name__ == "__main__":
    main()
