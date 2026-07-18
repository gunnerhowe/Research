"""Report cards for forecasters — the scoring conventions from the paper, as code.

Non-negotiables encoded here:
- Forecast quality is a (ranking, lead) PAIR. A rule can rank runs perfectly by
  detecting the event as it happens (our best-case loss rule tied rho=0.977 with a
  50-step lead — a nowcast). Never report correlation without lead.
- False-alarm rates require negatives; below MIN_NEGATIVES per side the number is
  flagged as uncertifiable (a single-negative cap once passed a degenerate fit in our
  own work).
"""
from __future__ import annotations

import warnings

import numpy as np

MIN_NEGATIVES = 5


def spearman(x, y) -> float:
    def rank(a):
        o = np.argsort(a)
        r = np.empty(len(a))
        r[o] = np.arange(len(a))
        return r
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def report_card(positives: list[dict], negatives: list[dict],
                budget: float | None = None) -> dict:
    """positives: [{'t_anchor': float|None, 't_event': float}], negatives:
    [{'t_anchor': float|None}] (an anchor on a negative is a false alarm).

    Returns the full (ranking, lead, miss, FA) card. Leads score 0 for misses and for
    post-event anchors, per the paper's convention.
    """
    anchors, events, leads = [], [], []
    miss = 0
    for p in positives:
        te = p["t_event"]
        ta = p.get("t_anchor")
        events.append(te)
        anchors.append(ta if ta is not None else float("nan"))
        if ta is None or ta >= te:
            miss += 1
            leads.append(0.0)
        else:
            leads.append(te - ta)
    fa = sum(1 for n in negatives if n.get("t_anchor") is not None)
    ok = [i for i, a in enumerate(anchors) if not np.isnan(a)]
    rho = spearman([anchors[i] for i in ok], [events[i] for i in ok]) if len(ok) >= 3 else float("nan")
    card = {
        "n_pos": len(positives), "n_neg": len(negatives),
        "spearman_anchor_vs_event": round(rho, 4) if rho == rho else None,
        "median_lead": float(np.median(leads)) if leads else 0.0,
        "median_relative_lead": float(np.median([l / e for l, e in zip(leads, events)]))
        if leads else 0.0,
        "miss_rate": miss / len(positives) if positives else None,
        "false_alarm_rate": fa / len(negatives) if negatives else None,
        "fa_certified": len(negatives) >= MIN_NEGATIVES,
    }
    if not card["fa_certified"]:
        warnings.warn(
            f"false-alarm rate computed on {len(negatives)} negatives (< {MIN_NEGATIVES})"
            " — UNCERTIFIABLE; manufacture more capability-blocked runs", stacklevel=2)
    if budget is not None and leads:
        card["median_lead_frac_of_budget"] = float(np.median(leads)) / budget
    return card
