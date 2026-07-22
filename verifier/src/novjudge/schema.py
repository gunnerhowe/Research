"""Frozen data schema for the S x G crossed design (see PLAN.md).

This is the fixed contract that item construction, validation, and the judge
harness are all built against. No measurement logic lives here — only the shape
of the data — so that it can be committed alongside the frozen pre-registration.

Design recap:
  - A *Stem* fixes a prior-work context and a base technical idea in a domain.
  - Each Stem expands into four *Items* crossing substantive novelty S in
    {low, high} with novelty signal G in {low, high}, correctness held fixed.
  - Items pass S/G/correctness validation (K4) BEFORE any judge scores them.
  - A judge produces a *JudgeScore* per (item, judge, rubric, mode).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Level = Literal["low", "high"]
Domain = Literal["ml", "econ", "bio"]
Mode = Literal["pointwise", "pairwise"]


@dataclass(frozen=True)
class Stem:
    """A fixed prior-work context + base idea that expands into the S x G cells."""

    stem_id: str
    domain: Domain
    prior_work: str          # the cited-background context S is measured against
    base_idea: str           # the technical proposal, before S/G manipulation
    source: str = ""         # provenance (arXiv id / OpenReview id / synthetic)
    contaminated: bool = False  # real-idea anchor whose outcome judges may know


@dataclass(frozen=True)
class Item:
    """One cell of a stem's 2x2. `cell` in {LL, LH, HL, HH} = (S_level, G_level)."""

    item_id: str
    stem_id: str
    domain: Domain
    s_level: Level           # substantive novelty (technical content)
    g_level: Level           # novelty signal (framing/rhetoric)
    text: str                # the item shown to the judge

    # --- validation fields, filled by the construction/adjudication pipeline ---
    # (Items only enter scoring once validated; unvalidated items are dropped.)
    g_lexicon_count: Optional[int] = None       # frozen signal-lexicon hits
    g_classifier_score: Optional[float] = None  # judge-independent embedding G
    s_adjudicated: Optional[Level] = None       # strong-model blind S label
    correctness_ok: Optional[bool] = None        # sound + no manipulation leak
    human_spot_checked: Optional[bool] = None

    @property
    def cell(self) -> str:
        return {"low": "L", "high": "H"}[self.s_level] + {"low": "L", "high": "H"}[self.g_level]

    @property
    def validated(self) -> bool:
        return (
            self.correctness_ok is True
            and self.s_adjudicated == self.s_level
            and self.g_classifier_score is not None
        )


@dataclass(frozen=True)
class JudgeScore:
    """A judge's novelty rating of an item under a frozen rubric."""

    item_id: str
    judge: str               # model id (local 4-bit or API)
    rubric_id: str           # committed, frozen prompt id
    mode: Mode
    score: float             # novelty rating on the rubric's scale
    raw: str = ""            # raw judge completion (for audit)
    # steering context (E4); null on the untouched baseline:
    steer_direction: Optional[str] = None
    steer_coeff: float = 0.0
    steer_layer: Optional[int] = None


@dataclass
class Cross:
    """The four validated items of one stem, for the mixed-model long table."""

    stem: Stem
    items: dict[str, Item] = field(default_factory=dict)  # keyed by cell LL/LH/HL/HH

    @property
    def complete(self) -> bool:
        return set(self.items) == {"LL", "LH", "HL", "HH"} and all(
            it.validated for it in self.items.values()
        )
