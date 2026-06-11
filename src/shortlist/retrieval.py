"""Hybrid retrieval — cheap, high-recall shortlist before deep scoring.

Two complementary channels, mirroring a production search stack:
  sparse: curated facet-term matching (the evidence engine's vocabulary) —
          precise, explainable, zero-cost misses;
  dense:  embedding similarity from the precomputed cache — catches profiles
          that describe the work without using any canonical vocabulary.

The shortlist is a union (never an intersection): a candidate only needs to
look interesting to ONE channel to earn full scoring. Obviously-irrelevant
profiles (an accountant whose narrative matches nothing and embeds far from
the JD) are dropped here, which is what keeps the 5-minute budget safe.
"""

from __future__ import annotations

import re

from .dense import DenseChannel
from .evidence import EvidenceEngine
from .loader import Candidate

# Titles that always survive to full scoring regardless of channel hits —
# the cost of scoring a few thousand extra candidates is trivial, the cost of
# dropping one hidden gem is not.
SAFETY_TITLE_RE = re.compile(
    r"ml|machine learning|\bai\b|data scien|recommendation|search|nlp|applied scien|deep learning",
    re.IGNORECASE,
)

DENSE_TOP_K = 2000


class HybridRetriever:
    def __init__(self, engine: EvidenceEngine, dense: DenseChannel | None):
        self.engine = engine
        self.dense = dense
        # One combined alternation over every active facet term: a single
        # cheap regex pass decides whether deep evidence extraction is worth it.
        all_terms = sorted(
            {t for pats in engine._patterns.values() for t, _ in pats}, key=len, reverse=True
        )
        self._screen = re.compile("|".join(re.escape(t) for t in all_terms), re.IGNORECASE)

    def shortlist(
        self, candidates: list[Candidate], spec: dict
    ) -> tuple[list[Candidate], dict[str, float]]:
        """Returns (candidates worth deep scoring, dense scores for everyone)."""
        dense_scores: dict[str, float] = {}
        if self.dense is not None:
            dense_scores = self.dense.scores(candidates, spec) or {}

        keep: list[Candidate] = []
        dense_rank = {}
        if dense_scores:
            ordered = sorted(dense_scores, key=dense_scores.get, reverse=True)
            dense_rank = {cid: i for i, cid in enumerate(ordered[:DENSE_TOP_K])}

        for c in candidates:
            if (
                c.cid in dense_rank
                or SAFETY_TITLE_RE.search(c.title)
                or self._screen.search(c.narrative_lc)
            ):
                keep.append(c)
        return keep, dense_scores
