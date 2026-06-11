"""Expected-hire scorer — turns evidence into a defensible ranking.

    FinalScore = CoreFit × (1 − capped JD penalties) × behavioral multiplier

CoreFit blends five interpretable components (lexical evidence, trusted-skill
coverage, dense similarity, experience-band fit, career quality). Component
values are computed once; the weight blend is then evaluated under K jittered
weight configurations and aggregated by mean rank (Borda), so the final order
reflects the evidence, not one arbitrary choice of weights.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

import numpy as np

from .evidence import EvidenceEngine, FacetEvidence, HANDS_ON_RE
from .loader import Candidate
from .signals import BehavioralProfile, SignalModel
from .trust import TrustModel

BLEND_WEIGHTS = {
    "lexical": 0.30,
    "trust": 0.25,
    "dense": 0.15,
    "band": 0.10,
    "career": 0.20,
}
PENALTY_PER_DETECTOR = 0.25
PENALTY_CAP = 0.60
ENSEMBLE_K = 25
ENSEMBLE_SIGMA = 0.15
ENSEMBLE_SEED = 20260615  # REF_DATE — deterministic across runs

DOMAIN_FACETS = ("nlp_ir", "embeddings_retrieval", "ranking_recsys", "vector_infra")


@dataclass(slots=True)
class ScoreBreakdown:
    cid: str
    components: dict[str, float]
    core_fit: float
    penalties: dict[str, float]
    penalty_factor: float
    behavioral: BehavioralProfile
    final: float
    facet_evidence: dict[str, FacetEvidence] = field(repr=False, default=None)
    trust: dict = field(repr=False, default=None)


def _weighted_facet_mean(values: dict[str, float], spec: dict) -> float:
    total, norm = 0.0, 0.0
    for level in ("must_have", "nice_to_have"):
        for r in spec["requirements"][level]:
            w = r["weight"]
            total += w * values.get(r["facet"], 0.0)
            norm += w
    return total / norm if norm else 0.0


def _band_fit(yoe: float, spec: dict) -> float:
    exp = spec["experience"]
    lo, hi = exp.get("min_years"), exp.get("max_years")
    if lo is None or hi is None:
        return 0.7
    ilo, ihi = exp.get("ideal_min") or lo, exp.get("ideal_max") or hi
    if ilo <= yoe <= ihi:
        return 1.0
    if lo <= yoe <= hi:
        return 0.92
    if yoe < lo:
        return max(0.0, 0.92 - (lo - yoe) * 0.25)
    return max(0.30, 0.92 - (yoe - hi) * 0.12)


def _career_quality(c: Candidate, facet_ev: dict[str, FacetEvidence], engine: EvidenceEngine) -> float:
    if not c.jobs:
        return 0.3
    consulting = sum(
        1
        for j in c.jobs
        if any(k in j.company.lower() for k in engine._consulting) or j.industry == "IT Services"
    )
    product_share = 1.0 - consulting / len(c.jobs)

    current = next((j for j in c.jobs if j.is_current), c.jobs[0])
    hands_on_recent = 1.0 if HANDS_ON_RE.search(current.description) else 0.4

    domain = max(facet_ev.get(f, FacetEvidence(0)).score for f in DOMAIN_FACETS)

    stints = [j.duration_months for j in c.jobs if j.duration_months > 0]
    med = statistics.median(stints) if stints else 0
    tenure = min(1.0, med / 24.0)

    return 0.35 * product_share + 0.25 * hands_on_recent + 0.25 * domain + 0.15 * tenure


class Scorer:
    def __init__(self, spec: dict, engine: EvidenceEngine, trust: TrustModel, signals: SignalModel):
        self.spec = spec
        self.engine = engine
        self.trust = trust
        self.signals = signals

    def breakdown(self, c: Candidate, dense_score: float | None) -> ScoreBreakdown:
        facet_ev = self.engine.facet_evidence(c)
        trust_cov = self.trust.coverage(c, facet_ev)
        negatives = self.engine.negative_signals(c, facet_ev)
        behavioral = self.signals.profile(c)

        components = {
            "lexical": _weighted_facet_mean({f: e.score for f, e in facet_ev.items()}, self.spec),
            "trust": _weighted_facet_mean({f: t.score for f, t in trust_cov.items()}, self.spec),
            "dense": dense_score if dense_score is not None else 0.0,
            "band": _band_fit(c.yoe, self.spec),
            "career": _career_quality(c, facet_ev, self.engine),
        }
        weights = dict(BLEND_WEIGHTS)
        if dense_score is None:  # channel unavailable: renormalize without it
            weights.pop("dense")
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
            components["dense"] = 0.0
            core = sum(components[k] * w for k, w in weights.items())
        else:
            core = sum(components[k] * w for k, w in weights.items())

        penalty_total = min(PENALTY_CAP, sum(s * PENALTY_PER_DETECTOR for s in negatives.values()))
        penalty_factor = 1.0 - penalty_total
        final = core * penalty_factor * behavioral.multiplier

        return ScoreBreakdown(
            cid=c.cid,
            components={k: round(v, 4) for k, v in components.items()},
            core_fit=round(core, 4),
            penalties={k: round(v, 3) for k, v in negatives.items()},
            penalty_factor=round(penalty_factor, 4),
            behavioral=behavioral,
            final=round(final, 6),
            facet_evidence=facet_ev,
            trust=trust_cov,
        )


def ensemble_rank(
    breakdowns: list[ScoreBreakdown], dense_available: bool
) -> tuple[list[ScoreBreakdown], dict[str, float]]:
    """Borda aggregation over K weight-jittered blends.

    Component values, penalties and multipliers are fixed evidence; only the
    blend weights are jittered (lognormal, sigma=0.15). Returns breakdowns in
    final order plus a stability map (fraction of jitters agreeing the
    candidate belongs in the top 10).
    """
    names = [k for k in BLEND_WEIGHTS if dense_available or k != "dense"]
    comp = np.array([[b.components[k] for k in names] for b in breakdowns])
    mods = np.array([b.penalty_factor * b.behavioral.multiplier for b in breakdowns])
    base = np.array([BLEND_WEIGHTS[k] for k in names])

    rng = np.random.default_rng(ENSEMBLE_SEED)
    rank_sum = np.zeros(len(breakdowns))
    top10_hits = np.zeros(len(breakdowns))
    for _ in range(ENSEMBLE_K):
        w = base * rng.lognormal(0.0, ENSEMBLE_SIGMA, size=len(names))
        w /= w.sum()
        scores = (comp @ w) * mods
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype=int)
        ranks[order] = np.arange(len(scores))
        rank_sum += ranks
        top10_hits[order[:10]] += 1

    mean_rank = rank_sum / ENSEMBLE_K
    # Mean rank decides; the candidate's own base score breaks ties, then cid.
    order = sorted(
        range(len(breakdowns)),
        key=lambda i: (mean_rank[i], -breakdowns[i].final, breakdowns[i].cid),
    )
    stability = {breakdowns[i].cid: top10_hits[i] / ENSEMBLE_K for i in order[:30]}
    return [breakdowns[i] for i in order], stability
