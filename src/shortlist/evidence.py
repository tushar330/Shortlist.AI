"""Evidence engine — what a candidate has demonstrably *done*.

Reads the same ontology as the JD compiler and scores each job-spec facet from
the candidate's narrative (career descriptions > summary > headline). The
skills array is deliberately not trusted here — uncorroborated skill lists are
the keyword-stuffer trap; they are handled separately by the trust module.

Also evaluates the negative detectors the JD activated (research-only careers,
CV-without-IR, consulting-only, title-chasers, management drift, thin recent
LLM-wrapper experience).
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from .jd_compiler import load_ontology, _term_pattern
from .loader import Candidate

# Section weights: claims in job descriptions outrank the self-summary, which
# outranks the one-line headline. Listed skills score zero here by design.
SECTION_WEIGHTS = {"desc": 1.0, "summary": 0.7, "headline": 0.5}

CV_SPEECH_TERMS = (
    "computer vision", "object detection", "image classification", "image segmentation",
    "opencv", "yolo", "speech recognition", "asr", "text-to-speech", "tts",
    "robotics", "slam", "autonomous", "diffusion model", "gan",
)
RESEARCH_TITLE_RE = re.compile(r"research|postdoc|academic|phd", re.IGNORECASE)
RESEARCH_INDUSTRY_RE = re.compile(r"research|academia|university", re.IGNORECASE)
HANDS_ON_RE = re.compile(
    r"\b(built|implemented|developed|shipped|deployed|trained|wrote|coded|coding|"
    r"debugged|optimi[sz]ed|refactored|integrated|automated)\b",
    re.IGNORECASE,
)
MGMT_TITLE_RE = re.compile(
    r"\b(architect|head of|vp\b|vice president|director|engineering manager|tech lead|principal(?! engineer))",
    re.IGNORECASE,
)
SENIOR_TITLE_RE = re.compile(r"\b(senior|staff|principal|lead)\b", re.IGNORECASE)
WRAPPER_RE = re.compile(r"langchain|openai api|gpt-?[345w]|chatgpt|prompt", re.IGNORECASE)
CORE_ML_RE = re.compile(
    r"\b(trained|fine-?tun|embedding|ranking model|xgboost|lightgbm|production model|"
    r"feature engineering|offline eval)\b",
    re.IGNORECASE,
)

AI_FACETS = (
    "embeddings_retrieval", "vector_infra", "ranking_recsys", "llm_engineering",
    "nlp_ir", "learning_to_rank",
)


def _saturation(n_terms: int) -> float:
    """Distinct-term saturation: 1 term is a mention, 3+ is fluency."""
    if n_terms <= 0:
        return 0.0
    return min(1.0, 0.55 + 0.225 * (n_terms - 1))


@dataclass(slots=True)
class FacetEvidence:
    score: float
    terms: list[str] = field(default_factory=list)
    section: str = ""


class EvidenceEngine:
    def __init__(self, spec: dict, ontology: dict | None = None):
        self.spec = spec
        self.ontology = ontology or load_ontology()
        reqs = spec["requirements"]
        self.active_facets = {
            r["facet"]: r["weight"]
            for level in ("must_have", "nice_to_have", "contextual")
            for r in reqs[level]
        }
        self._patterns: dict[str, list[tuple[str, re.Pattern]]] = {
            facet: [(t, _term_pattern(t)) for t in self.ontology["facets"][facet]["terms"]]
            for facet in self.active_facets
        }
        self._cv_patterns = [(t, _term_pattern(t)) for t in CV_SPEECH_TERMS]
        self._consulting = tuple(self.ontology["consulting_employers"])
        self.active_detectors = {d["type"] for d in spec.get("disqualifiers", [])}

    # ------------------------------------------------------------- positive
    def facet_evidence(self, c: Candidate) -> dict[str, FacetEvidence]:
        sections = {
            "desc": " ".join(j.description for j in c.jobs).lower(),
            "summary": c.summary.lower(),
            "headline": c.headline.lower(),
        }
        out: dict[str, FacetEvidence] = {}
        for facet, patterns in self._patterns.items():
            best = FacetEvidence(0.0)
            for name, weight in SECTION_WEIGHTS.items():
                text = sections[name]
                terms = [t for t, p in patterns if p.search(text)]
                score = weight * _saturation(len(terms))
                if score > best.score:
                    best = FacetEvidence(round(score, 4), terms[:6], name)
            out[facet] = best
        return out

    # ------------------------------------------------------------- negative
    def negative_signals(
        self, c: Candidate, facet_ev: dict[str, FacetEvidence]
    ) -> dict[str, float]:
        """Detector -> severity 0..1, only for detectors the JD activated."""
        out: dict[str, float] = {}
        desc_all = " ".join(j.description for j in c.jobs)

        if "research_only" in self.active_detectors and c.jobs:
            researchy = sum(
                1
                for j in c.jobs
                if (RESEARCH_TITLE_RE.search(j.title) and "data scientist" not in j.title.lower())
                or RESEARCH_INDUSTRY_RE.search(j.industry)
            )
            share = researchy / len(c.jobs)
            production = facet_ev.get("production_shipping", FacetEvidence(0)).score
            if share > 0:
                out["research_only"] = round(share * (1.0 - production), 3)

        if "cv_speech_only" in self.active_detectors:
            cv_terms = [t for t, p in self._cv_patterns if p.search(c.narrative_lc)]
            ir_score = max(
                facet_ev.get(f, FacetEvidence(0)).score
                for f in ("nlp_ir", "embeddings_retrieval", "ranking_recsys", "vector_infra")
            )
            if len(cv_terms) >= 2 and ir_score < 0.3:
                out["cv_speech_only"] = 0.9
            elif cv_terms and ir_score < 0.3 and re.search(r"vision|speech", c.title, re.I):
                out["cv_speech_only"] = 0.6

        if "consulting_only" in self.active_detectors and c.jobs:
            def is_consulting(j):
                comp = j.company.lower()
                return any(k in comp for k in self._consulting) or j.industry == "IT Services"
            if all(is_consulting(j) for j in c.jobs):
                out["consulting_only"] = 1.0 if len(c.jobs) >= 2 else 0.6

        if "title_chaser" in self.active_detectors and len(c.jobs) >= 3:
            # The JD's complaint is hopping *for title escalation* (~every 1.5
            # years), not ordinary startup mobility — so escalation is required.
            ended = [j.duration_months for j in c.jobs if not j.is_current]
            companies = {j.company for j in c.jobs}
            if ended and len(companies) >= 3 and statistics.median(ended) <= 16:
                first, last = c.jobs[-1], c.jobs[0]  # history is newest-first
                escalated = bool(SENIOR_TITLE_RE.search(last.title)) and not bool(
                    SENIOR_TITLE_RE.search(first.title)
                )
                if escalated:
                    out["title_chaser"] = 0.8

        if "non_coding_architect" in self.active_detectors and c.jobs:
            current = next((j for j in c.jobs if j.is_current), c.jobs[0])
            if MGMT_TITLE_RE.search(c.title) or MGMT_TITLE_RE.search(current.title):
                hands_on = bool(HANDS_ON_RE.search(current.description))
                out["non_coding_architect"] = 0.3 if hands_on else 0.8

        if "recent_wrapper_only" in self.active_detectors and c.jobs:
            ai_jobs = [
                j
                for j in c.jobs
                if any(
                    p.search(j.description.lower())
                    for f in AI_FACETS
                    if f in self._patterns
                    for _, p in self._patterns[f]
                )
            ]
            if ai_jobs:
                only_current = all(j.is_current for j in ai_jobs)
                current_tenure = max((j.computed_months or 0) for j in ai_jobs)
                if only_current and current_tenure < 12 and WRAPPER_RE.search(desc_all):
                    core_hits = len(set(CORE_ML_RE.findall(desc_all)))
                    out["recent_wrapper_only"] = 0.8 if core_hits <= 1 else 0.4

        if "closed_source_no_validation" in self.active_detectors:
            gh = c.sig.get("github_activity_score", -1)
            oss = facet_ev.get("open_source", FacetEvidence(0)).score
            if c.yoe >= 5 and gh == -1 and oss == 0:
                out["closed_source_no_validation"] = 0.3

        # framework_enthusiast is evidence-thin in this dataset; subsumed by
        # recent_wrapper_only. Kept inactive deliberately.
        return out
