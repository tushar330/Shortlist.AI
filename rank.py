"""Shortlist.AI — produce the ranked submission CSV from a candidate pool.

Usage:
    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv [--audit]

Runs CPU-only and offline: the dense channel reads the precomputed embedding
cache (see scripts/precompute_embeddings.py); if neither cache nor local model
is available the pipeline degrades to lexical-only mode and still completes.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from shortlist.cross_encoder import CrossEncoderReranker, RERANK_POOL, blend_head
from shortlist.dense import DenseChannel, DEFAULT_CACHE
from shortlist.evidence import EvidenceEngine
from shortlist.integrity import audit as integrity_audit
from shortlist.loader import load_candidates, load_candidates_json
from shortlist.reasoning import reasoning_for
from shortlist.retrieval import HybridRetriever
from shortlist.scoring import Scorer, ensemble_rank
from shortlist.signals import SignalModel
from shortlist.trust import TrustModel

TOP_N = 100


def run(
    candidates_path: str,
    out_path: str,
    spec_path: str,
    use_dense: bool,
    audit: bool,
    use_rerank: bool = True,
) -> None:
    t0 = time.time()
    mark = lambda msg: print(f"[{time.time() - t0:6.1f}s] {msg}")

    spec = yaml.safe_load(open(spec_path, encoding="utf-8"))
    engine = EvidenceEngine(spec)
    trust = TrustModel(spec)
    signals = SignalModel(spec)
    scorer = Scorer(spec, engine, trust, signals)
    mark("spec + engines ready")

    if candidates_path.endswith(".json"):
        pool = load_candidates_json(candidates_path)
    else:
        pool = load_candidates(candidates_path)
    mark(f"loaded {len(pool):,} candidates")

    clean, flagged = [], []
    for c in pool:
        flags = integrity_audit(c)
        (flagged if flags else clean).append((c, flags) if flags else c)
    mark(f"integrity gate: {len(flagged)} flagged, {len(clean):,} clean")

    dense = DenseChannel(DEFAULT_CACHE if use_dense else None) if use_dense else None
    retriever = HybridRetriever(engine, dense)
    shortlist, dense_scores = retriever.shortlist(clean, spec)
    dense_available = bool(dense_scores)
    mark(
        f"retrieval: {len(shortlist):,} shortlisted "
        f"(dense channel {'on' if dense_available else 'OFF — lexical-only mode'})"
    )

    breakdowns = [scorer.breakdown(c, dense_scores.get(c.cid) if dense_available else None) for c in shortlist]
    mark(f"scored {len(breakdowns):,} candidates")

    ordered, stability = ensemble_rank(breakdowns, dense_available)
    mark("ensemble ranking complete")

    by_cid = {c.cid: c for c in shortlist}
    by_bid = {b.cid: b for b in ordered}
    if use_rerank:
        head = [by_cid[b.cid] for b in ordered[:RERANK_POOL]]
        ce = CrossEncoderReranker().scores(spec, head)
        if ce:
            new_order = blend_head(
                [b.cid for b in ordered], {b.cid: b.final for b in ordered}, ce
            )
            ordered = [by_bid[cid] for cid in new_order]
            mark(f"cross-encoder re-ranked top {len(ce)}")
        else:
            mark("cross-encoder unavailable or over budget — ensemble order stands")
    top = ordered[:TOP_N]

    rows = []
    prev = 1.0
    for i, b in enumerate(top):
        score = min(prev - 1e-6, round(max(b.final, 1e-6), 6))
        prev = score
        rows.append(
            {
                "candidate_id": b.cid,
                "rank": i + 1,
                "score": f"{score:.6f}",
                "reasoning": reasoning_for(b, by_cid[b.cid], i + 1, spec),
            }
        )
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        w.writeheader()
        w.writerows(rows)
    mark(f"wrote {out_path}")

    overlap = {b.cid for b in top} & {c.cid for c, _ in flagged}
    if overlap:
        print(f"WARNING: integrity-flagged candidates in top-{TOP_N}: {overlap}")

    if audit:
        from shortlist.report import write_audit

        write_audit(
            top, stability, by_cid, spec,
            Path(__file__).parent / "artifacts" / "audit_report.md",
            dense_available=dense_available,
        )
        mark("audit report written")

    mark("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--spec", default=str(Path(__file__).parent / "jd" / "job_spec.yaml"))
    ap.add_argument("--no-dense", action="store_true", help="force lexical-only mode")
    ap.add_argument("--no-rerank", action="store_true", help="skip cross-encoder pass")
    ap.add_argument("--audit", action="store_true", help="write artifacts/audit_report.md")
    args = ap.parse_args()
    run(
        args.candidates,
        args.out,
        args.spec,
        use_dense=not args.no_dense,
        audit=args.audit,
        use_rerank=not args.no_rerank,
    )


if __name__ == "__main__":
    main()
