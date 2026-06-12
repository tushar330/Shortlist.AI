"""Pre-download the ONNX models into the local cache (one-time, needs network).

Run this once before any offline/no-network execution (e.g. when building the
Stage-3 Docker image or the sandbox). After warming:
  - the dense bi-encoder never loads at rank time for the official pool
    (doc + query vectors come from artifacts/embeddings_cache.npz),
  - the cross-encoder loads from the local cache with no network.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shortlist.cross_encoder import MODEL_NAME as CE_MODEL
from shortlist.dense import MODEL_NAME as BI_MODEL


def main() -> None:
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    print(f"warming {BI_MODEL} ...")
    TextEmbedding(BI_MODEL)
    print(f"warming {CE_MODEL} ...")
    TextCrossEncoder(CE_MODEL)
    print("model caches ready - offline runs are now possible")


if __name__ == "__main__":
    main()
