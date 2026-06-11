"""Cross-encoder re-ranking — the precision pass over the head of the ranking.

A cross-encoder reads the JD query and the candidate narrative *together*
(full token-level attention) and is markedly better at fine-grained ordering
than any bi-encoder cosine — exactly the "LLM-based re-ranking" tier of the
architecture the JD itself sketches, scaled to fit a CPU-only 5-minute budget:

  - applied only to the top RERANK_POOL candidates after ensemble ranking;
  - hard wall-clock budget with mid-flight checks: if inference is slower than
    expected, the pass aborts and the ensemble order stands;
  - blended (not trusted blindly): the cross-encoder refines local order, the
    evidence-based score keeps the global structure.

Model: ms-marco-MiniLM-L-6-v2 (22M params, ONNX int8 via fastembed).
"""

from __future__ import annotations

import time

import numpy as np

from .dense import narrative_text
from .loader import Candidate

MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANK_POOL = 200
TIME_BUDGET_S = 75.0
BLEND_CE = 0.35  # cross-encoder share of the blended head score


def _load_model():
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(MODEL_NAME)
    except Exception:
        return None


def _queries(spec: dict) -> list[str]:
    out = []
    if spec.get("ideal_candidate_paragraph"):
        out.append(spec["ideal_candidate_paragraph"][:1000])
    out.extend(q[:300] for q in spec.get("facet_queries", [])[:2])
    return out or ["relevant candidate for this role"]


class CrossEncoderReranker:
    def __init__(self):
        self._model = None
        self._tried = False

    def _ensure(self):
        if not self._tried:
            self._tried = True
            self._model = _load_model()
        return self._model

    def scores(
        self, spec: dict, candidates: list[Candidate], budget_s: float = TIME_BUDGET_S
    ) -> dict[str, float] | None:
        """cid -> [0,1] score, or None if unavailable / over budget."""
        model = self._ensure()
        if model is None:
            return None
        queries = _queries(spec)
        docs = [narrative_text(c) for c in candidates]
        t0 = time.time()
        per_query: list[np.ndarray] = []
        for qi, q in enumerate(queries):
            scores = np.fromiter(model.rerank(q, docs), dtype=np.float32, count=len(docs))
            per_query.append(scores)
            elapsed = time.time() - t0
            remaining = len(queries) - (qi + 1)
            if remaining and elapsed + (elapsed / (qi + 1)) > budget_s:
                break  # keep what we have; partial signal is still signal
        if not per_query:
            return None
        raw = np.mean(per_query, axis=0)
        lo, hi = float(raw.min()), float(raw.max())
        norm = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
        return {c.cid: float(norm[i]) for i, c in enumerate(candidates)}


def blend_head(
    ordered_cids: list[str],
    base_scores: dict[str, float],
    ce_scores: dict[str, float],
) -> list[str]:
    """Re-order the head: blended = (1-w)·minmax(base) + w·cross-encoder."""
    head = [cid for cid in ordered_cids if cid in ce_scores]
    if not head:
        return ordered_cids
    base = np.array([base_scores[cid] for cid in head])
    lo, hi = float(base.min()), float(base.max())
    base_n = (base - lo) / (hi - lo) if hi > lo else np.zeros_like(base)
    blended = (1 - BLEND_CE) * base_n + BLEND_CE * np.array([ce_scores[c] for c in head])
    head_sorted = [cid for _, cid in sorted(zip(-blended, head))]
    tail = [cid for cid in ordered_cids if cid not in ce_scores]
    return head_sorted + tail
