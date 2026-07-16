"""P6 R2/R3: one-shot fleet scoring against the frozen prereg (commit 6b05770).

P2a: event spread >= 1.3x across the 30 positives.
P2b: Spearman(t_pv, t_event) >= 0.5 with bootstrap 95% CI excluding 0
     (t_pv = first eval with layer-0 prevtok >= 0.10).
P2c: best probe-time Spearman (t_pv or t_ind, t_ind = first eval indist_adv >= 0.10)
     exceeds best-case loss-threshold Spearman (theta swept, granted).
P2d: bare precursor false-alarms on norep; conjunction (prevtok>=0.10 AND indist_adv>=0.10
     at the same eval) achieves FA <= 1/10 on norep while alarming pre-event on >= 25/30
     positives.
P2e: onelayer: zero events; prefix < 0.05 throughout; conjunction alarms <= 1/5.
K2:  all probe-time Spearman CIs include 0.
Seed 0 excluded by construction (fleet is seeds 1-30 / 1-10 / 1-5).
"""
import json
import os

import numpy as np

GRID = "runs/grid6r2"
OUT = "analysis/out6"


def load(name):
    s = json.load(open(f"{GRID}/{name}/summary.json"))
    recs = [json.loads(l) for l in open(f"{GRID}/{name}/metrics.jsonl") if l.strip()]
    return s, recs


def first_step(recs, fn):
    for r in recs:
        if fn(r):
            return r["step"]
    return None


def spearman(x, y):
    def rank(a):
        o = np.argsort(a)
        rk = np.empty(len(a))
        rk[o] = np.arange(len(a))
        return rk
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_ci(x, y, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    rhos = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 1 and len(set(y[i])) > 1:
            rhos.append(spearman(x[i], y[i]))
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    pos = []
    for s in range(1, 31):
        summ, recs = load(f"rep_s{s}")
        pos.append(dict(
            seed=s, t_event=summ["t_event"],
            t_pv=first_step(recs, lambda r: r["prevtok_by_layer"][0] >= 0.10),
            t_ind=first_step(recs, lambda r: r["indist_adv"] >= 0.10),
            conj=first_step(recs, lambda r: r["prevtok_by_layer"][0] >= 0.10
                            and r["indist_adv"] >= 0.10),
            losses=[(r["step"], r["train_loss"]) for r in recs]))
    ev = [p["t_event"] for p in pos]
    assert all(e is not None for e in ev), "censored positive!"
    # P2a
    spread = max(ev) / min(ev)
    res["P2a"] = dict(min=min(ev), max=max(ev), spread=round(spread, 4),
                      passed=bool(spread >= 1.3))
    # P2b
    tpv = [p["t_pv"] for p in pos]
    assert all(t is not None for t in tpv), "precursor never crossed on a positive!"
    rho_pv = spearman(tpv, ev)
    ci_pv = boot_ci(tpv, ev)
    res["P2b"] = dict(rho=round(rho_pv, 4), ci=[round(c, 4) for c in ci_pv],
                      passed=bool(rho_pv >= 0.5 and ci_pv[0] > 0))
    # t_ind Spearman
    tind = [p["t_ind"] for p in pos]
    rho_ind = spearman(tind, ev) if all(t is not None for t in tind) else None
    ci_ind = boot_ci(tind, ev) if rho_ind is not None else (None, None)
    # P2c: best-case loss threshold
    all_losses = sorted({l for p in pos for _, l in p["losses"]})
    thetas = np.quantile(np.array(all_losses), np.linspace(0.01, 0.99, 99))
    best_loss_rho = None
    for th in thetas:
        tl = [first_step_loss(p["losses"], th) for p in pos]
        if any(t is None for t in tl) or len(set(tl)) < 2:
            continue
        r = spearman(tl, ev)
        if best_loss_rho is None or r > best_loss_rho[0]:
            best_loss_rho = (r, float(th))
    best_probe = max([r for r in (rho_pv, rho_ind) if r is not None])
    res["P2c"] = dict(best_probe_rho=round(best_probe, 4),
                      rho_ind=None if rho_ind is None else round(rho_ind, 4),
                      ci_ind=[None if c is None else round(c, 4) for c in ci_ind],
                      best_loss_rho=None if best_loss_rho is None else round(best_loss_rho[0], 4),
                      loss_theta=None if best_loss_rho is None else round(best_loss_rho[1], 4),
                      passed=bool(best_loss_rho is None or best_probe > best_loss_rho[0]))
    # P2d: negatives
    bare_fa, conj_fa = 0, 0
    norep_pv_max = []
    for s in range(1, 11):
        _, recs = load(f"norep_s{s}")
        pv = first_step(recs, lambda r: r["prevtok_by_layer"][0] >= 0.10)
        cj = first_step(recs, lambda r: r["prevtok_by_layer"][0] >= 0.10
                        and r["indist_adv"] >= 0.10)
        bare_fa += pv is not None
        conj_fa += cj is not None
        norep_pv_max.append(max(r["prevtok_by_layer"][0] for r in recs))
    conj_pre_event = sum(1 for p in pos if p["conj"] is not None
                         and p["conj"] < p["t_event"])
    res["P2d"] = dict(bare_precursor_fa=f"{bare_fa}/10",
                      norep_prevtok_max=[round(v, 3) for v in norep_pv_max],
                      conj_fa=f"{conj_fa}/10", conj_pre_event=f"{conj_pre_event}/30",
                      passed=bool(conj_fa <= 1 and conj_pre_event >= 25))
    # P2e: onelayer
    ol_events, ol_prefix_max, ol_conj = 0, [], 0
    for s in range(1, 6):
        summ, recs = load(f"onelayer_s{s}")
        ol_events += summ["t_event"] is not None
        ol_prefix_max.append(max(max(r["prefix_by_layer"]) for r in recs))
        cj = first_step(recs, lambda r: r["prevtok_by_layer"][0] >= 0.10
                        and r["indist_adv"] >= 0.10)
        ol_conj += cj is not None
    res["P2e"] = dict(events=ol_events, prefix_max=[round(v, 3) for v in ol_prefix_max],
                      conj_alarms=f"{ol_conj}/5",
                      passed=bool(ol_events == 0 and max(ol_prefix_max) < 0.05
                                  and ol_conj <= 1))
    # K2
    res["K2_fires"] = bool(not (ci_pv[0] > 0) and
                           not (ci_ind[0] is not None and ci_ind[0] > 0))
    # lead stats for reporting
    leads = [p["t_event"] - p["t_pv"] for p in pos]
    res["precursor_leads"] = dict(median=float(np.median(leads)), min=min(leads),
                                  max=max(leads))
    res["per_seed"] = [dict(seed=p["seed"], t_event=p["t_event"], t_pv=p["t_pv"],
                            t_ind=p["t_ind"]) for p in pos]
    json.dump(res, open(f"{OUT}/r2_scored.json", "w"), indent=2)
    for k in ("P2a", "P2b", "P2c", "P2d", "P2e"):
        print(k, json.dumps({kk: vv for kk, vv in res[k].items() if kk != "per_seed"}))
    print("K2 fires:", res["K2_fires"])
    print("precursor leads:", res["precursor_leads"])


def first_step_loss(losses, th):
    for step, l in losses:
        if l <= th:
            return step
    return None


if __name__ == "__main__":
    main()
