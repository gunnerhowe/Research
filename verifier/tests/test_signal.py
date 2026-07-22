"""Judge-independent signal (G) measurement gates."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from novjudge.signal import lexicon_count, lexicon_density, SignalEmbedder  # noqa: E402

SIGNALED = ("Contrary to the prevailing assumption, we propose a fundamentally new "
            "and unprecedented paradigm that overturns conventional wisdom.")
PLAIN = ("We extend the standard estimator with a small modification and evaluate "
         "it on the usual benchmark, following prior work.")


def test_lexicon_counts_signal_higher():
    assert lexicon_count(SIGNALED) > lexicon_count(PLAIN)
    assert lexicon_count(PLAIN) == 0
    assert lexicon_density(SIGNALED) > 0


def test_embedder_scores_signal_higher():
    # Uses cached MiniLM; if unavailable the environment isn't ready to run
    # experiments anyway, so let the failure surface rather than skipping.
    emb = SignalEmbedder()
    assert emb.score(SIGNALED) > emb.score(PLAIN)
