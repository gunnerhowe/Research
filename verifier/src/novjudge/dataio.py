"""Dataset assembly: the judge-independent G-separation gate (local half of K4),
and conversion of validated stems into estimator-ready scoring rows.

A stem dict has: stem_id, domain, subfield, topic, prior_work, s_low_content,
s_high_content, and the four cell texts LL/LH/HL/HH (S in {low,high} x G in
{low,high}). LL=s_low+plain, LH=s_low+signaled, HL=s_high+plain, HH=s_high+signaled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from novjudge.signal import lexicon_count, SignalEmbedder

CELLS = ("LL", "LH", "HL", "HH")
# cell -> (s, g) coded 0/1
CELL_SG = {"LL": (0, 0), "LH": (0, 1), "HL": (1, 0), "HH": (1, 1)}


@dataclass
class GSep:
    stem_id: str
    lex: dict           # cell -> lexicon count
    emb: dict           # cell -> MiniLM signal score
    passes: bool        # G-high > G-low within BOTH content levels (embedding),
                        # and not contradicted by lexicon


def g_separation(stem: dict, embedder: SignalEmbedder) -> GSep:
    """Require the signaled framing to out-signal the plain framing within each
    content level, on the MiniLM score (primary) with lexicon as a guard."""
    lex = {c: lexicon_count(stem[c]) for c in CELLS}
    emb = dict(zip(CELLS, embedder.score_many([stem[c] for c in CELLS])))
    emb_ok = (emb["LH"] > emb["LL"]) and (emb["HH"] > emb["HL"])
    # lexicon guard: signaled must not have FEWER signal words than plain.
    lex_ok = (lex["LH"] >= lex["LL"]) and (lex["HH"] >= lex["HL"])
    return GSep(stem_id=stem["stem_id"], lex=lex, emb=emb, passes=bool(emb_ok and lex_ok))


def filter_g_separation(stems: list[dict], embedder: SignalEmbedder | None = None):
    """Return (kept_stems, reports). Kept stems gain a `_gsep` field."""
    embedder = embedder or SignalEmbedder()
    kept, reports = [], []
    for s in stems:
        rep = g_separation(s, embedder)
        reports.append(rep)
        if rep.passes:
            s = dict(s)
            s["_gsep"] = {"lex": rep.lex, "emb": rep.emb}
            kept.append(s)
    return kept, reports


def scoring_rows(stems: list[dict]) -> list[dict]:
    """Flatten to one row per (stem, cell) for judge scoring."""
    rows = []
    for st in stems:
        for c in CELLS:
            s, g = CELL_SG[c]
            rows.append({
                "stem": st["stem_id"], "domain": st["domain"], "cell": c,
                "s": s, "g": g, "text": st[c], "prior_work": st["prior_work"],
            })
    return rows


def save_dataset(stems: list[dict], path: str, meta: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "stems": stems}, f, ensure_ascii=False, indent=2)


def load_dataset(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
