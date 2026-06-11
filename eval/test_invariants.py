"""Persona ordering invariants — the offline gate every change must pass.

Run: pytest eval/ -q   (from the repo root)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shadow_personas import build_shadow_pool
from shortlist.evidence import EvidenceEngine
from shortlist.integrity import audit as integrity_audit
from shortlist.scoring import Scorer, ensemble_rank
from shortlist.signals import SignalModel
from shortlist.trust import TrustModel

SPEC_PATH = Path(__file__).resolve().parents[1] / "jd" / "job_spec.yaml"


@pytest.fixture(scope="module")
def ranked():
    spec = yaml.safe_load(open(SPEC_PATH, encoding="utf-8"))
    engine = EvidenceEngine(spec)
    scorer = Scorer(spec, engine, TrustModel(spec), SignalModel(spec))
    pool = build_shadow_pool()
    labels = {c.cid: tier for c, tier in pool}

    clean = [c for c, _ in pool if not integrity_audit(c)]
    breakdowns = [scorer.breakdown(c, None) for c in clean]
    ordered, _ = ensemble_rank(breakdowns, dense_available=False)
    ranks = {b.cid: i for i, b in enumerate(ordered)}
    finals = {b.cid: b.final for b in ordered}
    return {"labels": labels, "ranks": ranks, "finals": finals, "pool": pool}


def test_honeypot_is_gate_flagged():
    pool = build_shadow_pool()
    honeypot = next(c for c, t in pool if c.cid == "SHDW_0000008")
    flags = integrity_audit(honeypot)
    assert flags, "honeypot with impossible profile must be flagged"
    assert {f.check for f in flags} & {
        "stated_experience_contradicts_history",
        "expert_skills_never_used",
    }


def test_tier5_above_everything_below_tier3(ranked):
    tier5 = [cid for cid, t in ranked["labels"].items() if t == 5 and cid in ranked["ranks"]]
    low = [cid for cid, t in ranked["labels"].items() if t <= 2 and cid in ranked["ranks"]]
    for hi in tier5:
        for lo in low:
            assert ranked["ranks"][hi] < ranked["ranks"][lo], f"{hi} must outrank {lo}"


def test_paraphrased_tier5_found_lexical_only(ranked):
    """Even without embeddings, the de-buzzworded strong profile must stay
    ahead of every tier<=2 persona (the dense channel then lifts it further)."""
    assert ranked["ranks"]["SHDW_0000002"] <= 4


def test_paraphrased_tier5_top3_with_dense():
    """With the dense channel on, the paraphrased tier-5 must reach the top 3."""
    pytest.importorskip("fastembed")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from shortlist.dense import DenseChannel

    spec = yaml.safe_load(open(SPEC_PATH, encoding="utf-8"))
    engine = EvidenceEngine(spec)
    scorer = Scorer(spec, engine, TrustModel(spec), SignalModel(spec))
    pool = build_shadow_pool()
    clean = [c for c, _ in pool if not integrity_audit(c)]

    dense = DenseChannel(cache_path=None)
    scores = dense.scores(clean, spec)
    if scores is None:
        pytest.skip("embedding model not available locally")
    breakdowns = [scorer.breakdown(c, scores[c.cid]) for c in clean]
    ordered, _ = ensemble_rank(breakdowns, dense_available=True)
    ranks = {b.cid: i for i, b in enumerate(ordered)}
    assert ranks["SHDW_0000002"] <= 2, f"paraphrased tier-5 at {ranks['SHDW_0000002']}"


def test_stuffer_buried(ranked):
    """Keyword stuffer must rank below every tier>=3 candidate."""
    stuffer_rank = ranked["ranks"]["SHDW_0000007"]
    for cid, tier in ranked["labels"].items():
        if tier >= 3 and cid in ranked["ranks"]:
            assert ranked["ranks"][cid] < stuffer_rank


def test_dormant_twin_below_active_twin(ranked):
    active, dormant = ranked["ranks"]["SHDW_0000001"], ranked["ranks"]["SHDW_0000012"]
    assert active < dormant
    # and the gap must be material, not a coin flip
    assert ranked["finals"]["SHDW_0000001"] > ranked["finals"]["SHDW_0000012"] * 1.15


def test_detector_personas_below_tier4(ranked):
    """Research-only, CV-only and consulting-only sit below every tier>=4."""
    for persona in ("SHDW_0000009", "SHDW_0000010", "SHDW_0000011"):
        for cid, tier in ranked["labels"].items():
            if tier >= 4 and cid in ranked["ranks"]:
                assert ranked["ranks"][cid] < ranked["ranks"][persona]


def test_shadow_ndcg_at_5(ranked):
    """NDCG@5 against shadow labels must stay near-perfect."""
    order = sorted(ranked["ranks"], key=ranked["ranks"].get)
    gains = [ranked["labels"][cid] for cid in order]
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:5]))
    ideal = sorted((ranked["labels"][cid] for cid in order), reverse=True)
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:5]))
    assert dcg / idcg >= 0.93, f"shadow NDCG@5 = {dcg / idcg:.3f} (lexical-only floor)"
