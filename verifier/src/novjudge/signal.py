"""Judge-independent measurement of novelty SIGNAL (G).

Two frozen, judge-independent measures so the G manipulation is *checked*, not
assumed (PLAN.md K4):
  1. `lexicon_count` — occurrences of a frozen list of novelty-asserting phrases.
  2. `SignalEmbedder` — a MiniLM difference-of-centroids score: cosine to a
     frozen set of novelty-signaling exemplars minus cosine to plain exemplars.

Neither uses any judge model, so a G-high vs G-low gap on these is independent
evidence that the framing manipulation landed.
"""

from __future__ import annotations

import re

# Frozen lexicon of novelty-asserting rhetoric (surface signal, not substance).
FROZEN_SIGNAL_LEXICON: tuple[str, ...] = (
    r"novel", r"for the first time", r"unprecedented", r"paradigm",
    r"fundamentally new", r"a new class of", r"radically", r"groundbreaking",
    r"contrary to (the )?(prevailing|conventional|standard|established|common)",
    r"challenge[sd]? (the )?(prevailing|conventional|standard|established|assumption)",
    r"surprisingl?y", r"counterintuitive", r"overturn", r"upend", r"rethink",
    r"we are the first", r"never before", r"breakthrough", r"revolutioniz",
    r"departs? from", r"beyond (existing|current|prior)", r"new frontier",
)

_LEX_RE = re.compile("|".join(f"(?:{p})" for p in FROZEN_SIGNAL_LEXICON), re.IGNORECASE)

# Frozen exemplars for the embedding score (never drawn from test stems).
SIGNAL_EXEMPLARS: tuple[str, ...] = (
    "Contrary to the prevailing assumption, we propose a fundamentally new approach.",
    "For the first time, we overturn the conventional wisdom in this field.",
    "This is a groundbreaking, unprecedented paradigm shift.",
    "Surprisingly, our radically new method challenges established practice.",
    "We introduce a novel framework that departs sharply from all prior work.",
)
PLAIN_EXEMPLARS: tuple[str, ...] = (
    "We extend the standard method with a small modification and evaluate it.",
    "Building on prior work, we apply the established technique to this setting.",
    "This follows the usual procedure and reports the resulting measurements.",
    "We combine two existing components and describe the implementation.",
    "The approach is a routine adaptation of a well-known baseline.",
)


def lexicon_count(text: str) -> int:
    """Number of novelty-signaling phrase hits (surface signal)."""
    return len(_LEX_RE.findall(text or ""))


def lexicon_density(text: str) -> float:
    """Signal hits per 100 words (length-normalized)."""
    n_words = max(1, len((text or "").split()))
    return 100.0 * lexicon_count(text) / n_words


class SignalEmbedder:
    """MiniLM difference-of-centroids signal score. Lazily loads the model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._sig_centroid = None
        self._plain_centroid = None

    def _ensure(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        import numpy as np

        self._model = SentenceTransformer(self.model_name)
        sig = self._model.encode(list(SIGNAL_EXEMPLARS), normalize_embeddings=True)
        plain = self._model.encode(list(PLAIN_EXEMPLARS), normalize_embeddings=True)
        self._sig_centroid = np.asarray(sig).mean(0)
        self._plain_centroid = np.asarray(plain).mean(0)

    def score(self, text: str) -> float:
        """cosine(text, signal-centroid) - cosine(text, plain-centroid).

        Positive => more signaling; the embeddings are L2-normalized so cosine
        reduces to a dot product.
        """
        self._ensure()
        import numpy as np

        v = self._model.encode([text or ""], normalize_embeddings=True)[0]
        v = np.asarray(v)
        return float(v @ self._sig_centroid - v @ self._plain_centroid)

    def score_many(self, texts: list[str]) -> list[float]:
        self._ensure()
        import numpy as np

        vs = np.asarray(self._model.encode(list(texts), normalize_embeddings=True))
        return list(vs @ self._sig_centroid - vs @ self._plain_centroid)
