"""Trusted-skill coverage — listed skills weighted by corroboration.

A listed skill is a claim, not evidence. Pool analysis showed keyword stuffers
list JD-relevant skills with plausible durations but near-zero endorsements,
no assessments, and zero narrative support — while genuine candidates carry
heavy endorsements and describe the work in their career history. So:

  - claimed proficiency is ignored entirely (free to fake),
  - durations contribute almost nothing (stuffers fake them),
  - endorsements, platform assessments, and narrative corroboration dominate.

Assessment keys are also used to *recover* canonical skills for paraphrased
profiles (e.g. a candidate listing "Vector Representations" whose assessment
key says "Embeddings") — those candidates are strong profiles in disguise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .evidence import FacetEvidence
from .jd_compiler import load_ontology
from .loader import Candidate


@dataclass(slots=True)
class FacetTrust:
    score: float
    best_skill: str = ""
    corroboration: list[str] = field(default_factory=list)


def _matches(term: str, skill_name: str) -> bool:
    if len(term) < 3:
        return term == skill_name
    return term == skill_name or (len(term) >= 4 and term in skill_name) or skill_name in term


class TrustModel:
    def __init__(self, spec: dict, ontology: dict | None = None):
        self.ontology = ontology or load_ontology()
        reqs = spec["requirements"]
        self.active_facets = [
            r["facet"]
            for level in ("must_have", "nice_to_have", "contextual")
            for r in reqs[level]
        ]
        self._facet_terms = {
            f: [t.lower() for t in self.ontology["facets"][f]["terms"]]
            for f in self.active_facets
        }

    def coverage(
        self, c: Candidate, facet_ev: dict[str, FacetEvidence]
    ) -> dict[str, FacetTrust]:
        assess = {k.lower(): v for k, v in c.sig.get("skill_assessment_scores", {}).items()}
        out: dict[str, FacetTrust] = {}

        for facet, terms in self._facet_terms.items():
            best = FacetTrust(0.0)
            narrative_score = facet_ev.get(facet, FacetEvidence(0)).score

            for s in c.skills:
                name = s["name"].lower()
                if not any(_matches(t, name) for t in terms):
                    continue
                weight, why = 0.15, ["listed"]
                endo = s.get("endorsements", 0)
                if endo >= 30:
                    weight += 0.25
                    why.append(f"{endo} endorsements")
                elif endo >= 10:
                    weight += 0.15
                    why.append(f"{endo} endorsements")
                if s.get("duration_months", 0) >= 24:
                    weight += 0.10
                    why.append(f"{s['duration_months']}m use")
                score = assess.get(name)
                if score is not None:
                    weight += 0.30 if score >= 60 else 0.15
                    why.append(f"assessment {score:.0f}")
                if narrative_score > 0:
                    weight += 0.45
                    why.append("described in career history")
                if weight > best.score:
                    best = FacetTrust(min(1.0, round(weight, 3)), s["name"], why)

            # Canonical-skill recovery: assessment exists for a facet term the
            # candidate never listed (paraphrased profile).
            for key, score in assess.items():
                if any(_matches(t, key) for t in terms):
                    weight = 0.55 + (0.2 if score >= 70 else 0.0)
                    if narrative_score > 0:
                        weight += 0.2
                    if weight > best.score:
                        best = FacetTrust(
                            min(1.0, round(weight, 3)),
                            key,
                            [f"platform assessment {score:.0f}"],
                        )
            out[facet] = best
        return out
