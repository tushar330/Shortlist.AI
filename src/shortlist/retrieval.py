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
from .evidence import EvidenceEngine, _facet_regex
from .loader import Candidate

# Titles that always survive to full scoring regardless of channel hits —
# the cost of scoring a few thousand extra candidates is trivial, the cost of
# dropping one hidden gem is not.
SAFETY_TITLE_RE = re.compile(
    r"ml|machine learning|\bai\b|data scien|recommendation|search|nlp|applied scien|deep learning",
    re.IGNORECASE,
)

DENSE_TOP_K = 2000


def spec_must_haves(spec: dict) -> list[dict]:
    return spec["requirements"]["must_have"]

# Facets whose vocabulary is too generic to indicate domain fit on its own
# ("production", "python", "github" appear in most tech narratives). They
# still contribute to scoring — they just don't earn a deep-scoring slot.
GENERIC_FACETS = frozenset(
    {"production_shipping", "python_engineering", "open_source", "hrtech_marketplace"}
)


class HybridRetriever:
    def __init__(self, engine: EvidenceEngine, dense: DenseChannel | None):
        self.engine = engine
        self.dense = dense
        # One combined alternation over the MUST-HAVE distinctive facet terms.
        # Nice-to-have vocabulary cannot earn a deep-scoring slot on its own:
        # pool measurement showed "rag"/"llm" course-dabbling mentions and
        # kubernetes/kafka ops talk wave half the pool through. Word boundaries
        # matter too — bare re.escape lets "rag" hit "storage"/"average".
        must_facets = {r["facet"] for r in spec_must_haves(engine.spec)}
        terms = {
            t
            for facet in must_facets
            if facet not in GENERIC_FACETS and facet in engine.active_facets
            for t in engine.ontology["facets"][facet]["terms"]
        }
        self._screen = _facet_regex(sorted(terms))
        self._screen_terms = {t.lower() for t in terms}

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
                or self._assessment_hook(c)
            ):
                keep.append(c)
        return keep, dense_scores

    def _assessment_hook(self, c: Candidate) -> bool:
        """Platform assessments in must-have vocabulary keep a candidate in —
        the paraphrased-profile recovery channel must survive retrieval too."""
        return any(
            k.lower() in self._screen_terms for k in c.sig.get("skill_assessment_scores", {})
        )
