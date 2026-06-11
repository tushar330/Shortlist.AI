"""Dense semantic channel — the recall safety net for paraphrased profiles.

Embeds candidate narratives with BGE-small-en-v1.5 (ONNX, CPU) and scores them
against the compiled JD's facet queries plus its ideal-candidate paragraph.
This is what catches the plain-language tier-5 candidate who writes "built a
system that suggests items users might like" without ever saying "recsys".

Operationally:
  - `scripts/precompute_embeddings.py` embeds the official pool once, offline,
    and stores doc vectors AND JD query vectors in one .npz cache.
  - The ranking step only loads the cache: no model, no network, milliseconds.
  - Unseen candidates (sandbox uploads) are embedded on the fly if the model
    is available locally; otherwise the pipeline reports the channel as
    unavailable and the scorer runs lexical-only. Never a hard failure.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .loader import Candidate

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "artifacts" / "embeddings_cache.npz"
MAX_CHARS = 2000  # ~512 tokens; narratives beyond this add little signal


def narrative_text(c: Candidate) -> str:
    parts = [c.headline, c.summary] + [j.description for j in c.jobs]
    return " ".join(p for p in parts if p)[:MAX_CHARS]


def spec_queries(spec: dict) -> list[str]:
    queries = list(spec.get("facet_queries", []))
    ideal = spec.get("ideal_candidate_paragraph")
    if ideal:
        queries.append(ideal)
    return queries


def queries_fingerprint(spec: dict) -> str:
    return hashlib.sha256("\n".join(spec_queries(spec)).encode("utf-8")).hexdigest()[:16]


def _load_model():
    try:
        from fastembed import TextEmbedding

        return TextEmbedding(MODEL_NAME)
    except Exception:
        return None


class DenseChannel:
    def __init__(self, cache_path: str | Path | None = DEFAULT_CACHE):
        self._doc_vecs: dict[str, np.ndarray] = {}
        self._query_vecs: np.ndarray | None = None
        self._query_fp: str | None = None
        self._model = None
        self._model_tried = False
        if cache_path and Path(cache_path).exists():
            data = np.load(cache_path, allow_pickle=False)
            ids = data["doc_ids"]
            vecs = data["doc_vecs"].astype(np.float32)
            self._doc_vecs = {cid: vecs[i] for i, cid in enumerate(ids)}
            if "query_vecs" in data:
                self._query_vecs = data["query_vecs"].astype(np.float32)
                self._query_fp = str(data["query_fp"])

    # ----------------------------------------------------------------- model
    def _ensure_model(self):
        if not self._model_tried:
            self._model_tried = True
            self._model = _load_model()
        return self._model

    def _embed_queries(self, spec: dict) -> np.ndarray | None:
        fp = queries_fingerprint(spec)
        if self._query_vecs is not None and self._query_fp == fp:
            return self._query_vecs
        model = self._ensure_model()
        if model is None:
            return None
        vecs = np.array(list(model.query_embed(spec_queries(spec))), dtype=np.float32)
        self._query_vecs, self._query_fp = vecs, fp
        return vecs

    def _embed_docs(self, missing: list[Candidate]) -> bool:
        model = self._ensure_model()
        if model is None:
            return False
        texts = [narrative_text(c) for c in missing]
        for c, vec in zip(missing, model.embed(texts, batch_size=64)):
            self._doc_vecs[c.cid] = np.asarray(vec, dtype=np.float32)
        return True

    # ---------------------------------------------------------------- public
    def available_for(self, candidates: list[Candidate], spec: dict) -> bool:
        missing = [c for c in candidates if c.cid not in self._doc_vecs]
        if missing and not self._embed_docs(missing):
            return False
        return self._embed_queries(spec) is not None

    def scores(self, candidates: list[Candidate], spec: dict) -> dict[str, float] | None:
        """cid -> dense similarity in [0,1], or None if channel unavailable.

        Raw score = 0.5 * cosine(ideal-candidate paragraph)
                  + 0.5 * mean of top-3 facet-query cosines,
        min-max normalized over the scored pool (deterministic for a fixed pool).
        """
        if not self.available_for(candidates, spec):
            return None
        q = self._embed_queries(spec)
        docs = np.stack([self._doc_vecs[c.cid] for c in candidates])
        sims = docs @ q.T  # vectors are L2-normalized by the model
        ideal = sims[:, -1]
        facet = sims[:, :-1] if sims.shape[1] > 1 else sims
        k = min(3, facet.shape[1])
        top_k = np.mean(np.sort(facet, axis=1)[:, -k:], axis=1)
        raw = 0.5 * ideal + 0.5 * top_k
        lo, hi = float(raw.min()), float(raw.max())
        norm = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
        return {c.cid: float(norm[i]) for i, c in enumerate(candidates)}
