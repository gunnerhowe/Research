"""novjudge — signal vs. substance in LLM research judges.

Project P1 of the `verifier` program. See PLAN.md for the frozen
pre-registration. Modules are added only after PLAN.md is frozen (committed);
this package currently exposes the frozen data schema only, so that item
construction and the judge harness are built against a fixed contract.
"""

__version__ = "0.0.1"

from novjudge.schema import Item, Stem, JudgeScore  # noqa: F401
