"""One-time offline pre-computation of the dense-channel cache.

Embeds every candidate narrative with BGE-small-en-v1.5 (ONNX, CPU) plus the
compiled JD's query set, and stores everything in artifacts/embeddings_cache.npz.
The ranking step then needs no model and no network — it just loads this cache.

This step may exceed the 5-minute ranking budget; the submission spec
explicitly allows documented pre-computation (Section 10.3).

Usage:
    python scripts/precompute_embeddings.py <candidates.jsonl> [--spec jd/job_spec.yaml]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shortlist.dense import MODEL_NAME, DEFAULT_CACHE, narrative_text, queries_fingerprint, spec_queries
from shortlist.loader import iter_candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--spec", default=str(Path(__file__).resolve().parents[1] / "jd" / "job_spec.yaml"))
    ap.add_argument("--out", default=str(DEFAULT_CACHE))
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    from fastembed import TextEmbedding

    t0 = time.time()
    model = TextEmbedding(MODEL_NAME)
    print(f"model {MODEL_NAME} ready in {time.time() - t0:.1f}s")

    ids: list[str] = []
    texts: list[str] = []
    for c in iter_candidates(args.candidates):
        ids.append(c.cid)
        texts.append(narrative_text(c))
    print(f"loaded {len(ids):,} narratives in {time.time() - t0:.1f}s")

    # Expect roughly 8 docs/s/laptop-core-set on full-length narratives:
    # ~3.5 hours for the 100K pool. One-time cost; the rank step only loads
    # the finished cache.
    vecs = np.empty((len(ids), 384), dtype=np.float16)
    done = 0
    for vec in model.embed(texts, batch_size=args.batch):
        vecs[done] = vec.astype(np.float16)
        done += 1
        if done % 2000 == 0:
            rate = done / (time.time() - t0)
            print(
                f"  embedded {done:,}/{len(ids):,}  ({rate:.0f}/s, "
                f"eta {(len(ids) - done) / rate / 60:.1f} min)",
                flush=True,
            )

    spec = yaml.safe_load(open(args.spec, encoding="utf-8"))
    query_vecs = np.array(list(model.query_embed(spec_queries(spec))), dtype=np.float16)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        doc_ids=np.array(ids),
        doc_vecs=vecs,
        query_vecs=query_vecs,
        query_fp=np.array(queries_fingerprint(spec)),
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
