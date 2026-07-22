"""Contract gates for the frozen S x G schema. No model calls here."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from novjudge.schema import Item, Stem, Cross  # noqa: E402


def _item(s, g, **kw):
    base = dict(
        item_id=f"i_{s}_{g}", stem_id="s0", domain="ml",
        s_level=s, g_level=g, text="x",
    )
    base.update(kw)
    return Item(**base)


def test_cell_encoding():
    assert _item("low", "low").cell == "LL"
    assert _item("low", "high").cell == "LH"
    assert _item("high", "low").cell == "HL"
    assert _item("high", "high").cell == "HH"


def test_validated_requires_all_checks():
    # Missing validation → not validated.
    assert _item("high", "high").validated is False
    # Adjudicated S must match constructed S (K4 guard).
    bad = _item("high", "high", correctness_ok=True, s_adjudicated="low",
                g_classifier_score=0.9)
    assert bad.validated is False
    # All checks pass → validated.
    good = _item("high", "high", correctness_ok=True, s_adjudicated="high",
                 g_classifier_score=0.9)
    assert good.validated is True


def test_cross_complete_needs_four_validated_cells():
    stem = Stem(stem_id="s0", domain="ml", prior_work="pw", base_idea="bi")
    cells = {
        "LL": _item("low", "low"), "LH": _item("low", "high"),
        "HL": _item("high", "low"), "HH": _item("high", "high"),
    }
    incomplete = Cross(stem=stem, items=cells)
    assert incomplete.complete is False  # none validated yet

    val = {
        c: _item(it.s_level, it.g_level, correctness_ok=True,
                 s_adjudicated=it.s_level, g_classifier_score=0.5)
        for c, it in cells.items()
    }
    assert Cross(stem=stem, items=val).complete is True
