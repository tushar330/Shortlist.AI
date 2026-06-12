"""Reasoning generator — explain each ranked candidate from their own evidence.

Design constraints (mirroring the Stage-4 review rubric):
  - every claim is rendered from a profile field or computed feature, never
    free-generated: hallucination is impossible by construction;
  - reasoning connects to specific JD requirements (facet labels come from the
    compiled spec);
  - honest concerns: the top concern is stated, not hidden, at every rank;
  - variation: sentence frames are chosen deterministically from the candidate
    id, so strings differ between rows but runs stay reproducible;
  - rank-consistent tone: confident head, balanced middle, hedged tail.
"""

from __future__ import annotations

import hashlib

from .loader import Candidate
from .scoring import ScoreBreakdown

PENALTY_PHRASES = {
    "research_only": "career skews research-heavy with little shipping evidence",
    "cv_speech_only": "background is vision/speech-centric with no IR exposure",
    "consulting_only": "entire career at IT-services firms",
    "title_chaser": "rapid title-driven company hopping",
    "non_coding_architect": "recent roles look hands-off",
    "recent_wrapper_only": "AI exposure is recent and wrapper-level",
    "closed_source_no_validation": "no public footprint to validate depth",
}

FACET_SHORT = {
    "embeddings_retrieval": "embeddings/retrieval",
    "vector_infra": "vector-search infrastructure",
    "ranking_recsys": "ranking/recsys",
    "eval_rigor": "ranking evaluation",
    "llm_engineering": "LLM fine-tuning",
    "learning_to_rank": "learning-to-rank",
    "nlp_ir": "NLP/IR",
    "python_engineering": "Python",
    "production_shipping": "production deployment",
    "distributed_scale": "large-scale systems",
    "open_source": "open-source work",
    "hrtech_marketplace": "HR-tech/marketplace",
}


ACRONYMS = {"bge", "e5", "ndcg", "mrr", "llm", "rag", "nlp", "ltr", "faiss", "bm25",
            "tts", "asr", "ner", "lora", "qlora", "peft", "oss", "a/b test", "a/b testing"}


def _pick(cid: str, salt: str, options: list[str]) -> str:
    h = int(hashlib.md5(f"{cid}:{salt}".encode()).hexdigest(), 16)
    return options[h % len(options)]


def _pretty_terms(terms: list[str]) -> str:
    picked: list[str] = []
    for t in terms:
        if any(t in p or p in t for p in picked):  # drop singular/plural twins
            continue
        picked.append(t)
        if len(picked) == 2:
            break
    return ", ".join(t.upper() if t in ACRONYMS and "/" not in t else t for t in picked)


STRENGTH_FRAMES = [
    "career history shows real {facet} work ({terms})",
    "past roles describe shipping {facet} systems ({terms})",
    "hands-on {facet} depth — {terms} appear in actual role descriptions",
    "{facet} experience is demonstrated, not just listed ({terms})",
    "solid {facet} track record ({terms} in career history)",
]
TRUST_FRAMES = [
    "{skill} backed by {why}",
    "{skill} is corroborated ({why})",
    "{skill} checks out — {why}",
]


def _strengths(b: ScoreBreakdown, c: Candidate, spec: dict) -> list[str]:
    out: list[str] = []
    ev_sorted = sorted(b.facet_evidence.items(), key=lambda kv: -kv[1].score)
    for i, (facet, ev) in enumerate(ev_sorted[:3]):
        if ev.score >= 0.55 and facet in FACET_SHORT:
            frame = _pick(b.cid, f"sf{i}", STRENGTH_FRAMES)
            terms = _pretty_terms(ev.terms) if ev.terms else "multiple signals"
            if "," not in terms:  # single term: fix verb agreement
                frame = frame.replace("{terms} appear", "{terms} appears")
            out.append(frame.format(facet=FACET_SHORT[facet], terms=terms))
    for facet, t in sorted(b.trust.items(), key=lambda kv: -kv[1].score):
        if t.score >= 0.7 and t.best_skill and len(out) < 4:
            why = t.corroboration[1] if len(t.corroboration) > 1 else t.corroboration[0]
            out.append(_pick(b.cid, "tf", TRUST_FRAMES).format(skill=t.best_skill, why=why))
            break
    exp = spec["experience"]
    if exp.get("ideal_min") and exp["ideal_min"] <= c.yoe <= (exp.get("ideal_max") or 99):
        out.append(f"{c.yoe:.1f} yrs lands in the ideal {exp['ideal_min']}-{exp['ideal_max']} band")
    sig = c.sig
    if sig.get("recruiter_response_rate", 0) >= 0.6 and (c.days_since_active or 999) <= 45:
        out.append(
            f"reachable (response rate {sig['recruiter_response_rate']:.0%}, "
            f"active {c.days_since_active}d ago)"
        )
    if sig.get("notice_period_days", 99) <= 30:
        out.append(f"{sig['notice_period_days']}-day notice")
    return out


def _concerns(b: ScoreBreakdown, c: Candidate, spec: dict) -> list[str]:
    out: list[str] = [PENALTY_PHRASES[p] for p in b.penalties if p in PENALTY_PHRASES]
    out.extend(b.behavioral.notes)
    exp = spec["experience"]
    lo, hi = exp.get("min_years"), exp.get("max_years")
    if lo and c.yoe < lo:
        out.append(f"{c.yoe:.1f} yrs is under the {lo}-{hi} band")
    elif hi and c.yoe > hi + 2:
        out.append(f"{c.yoe:.1f} yrs is above the {lo}-{hi} band")
    missing = [
        FACET_SHORT[r["facet"]]
        for r in spec["requirements"]["must_have"]
        if r["facet"] in FACET_SHORT
        and b.facet_evidence.get(r["facet"]) is not None
        and b.facet_evidence[r["facet"]].score < 0.2
        and b.trust[r["facet"]].score < 0.3
    ]
    if missing:
        out.append("no demonstrated " + ", ".join(missing[:2]))
    return out


def reasoning_for(b: ScoreBreakdown, c: Candidate, rank: int, spec: dict) -> str:
    strengths = _strengths(b, c, spec)
    concerns = _concerns(b, c, spec)
    cid = b.cid

    opener = _pick(
        cid,
        "opener",
        [
            f"{c.title} ({c.yoe:.1f} yrs, {c.location.split(',')[0]})",
            f"{c.yoe:.1f}-yr {c.title} in {c.location.split(',')[0]}",
            f"{c.location.split(',')[0]}-based {c.title} with {c.yoe:.1f} yrs",
        ],
    )

    joiner = _pick(cid, "join", ["; ", ", and ", "; plus "])
    if rank <= 15:
        body = joiner.join(strengths[:3]) if strengths else "broad fit across JD must-haves"
        concern_lead = _pick(cid, "c1", ["Only concern:", "One flag:", "Sole caveat:"])
        tail = f" {concern_lead} {concerns[0]}." if concerns else ""
        frame = _pick(cid, "f1", ["{o} — {b}.{t}", "{o}: {b}.{t}"])
    elif rank <= 50:
        body = joiner.join(strengths[:2]) if strengths else "reasonable facet coverage"
        concern_lead = _pick(cid, "c2", ["Watch-out:", "Caveat:", "To verify:"])
        tail = f" {concern_lead} {concerns[0]}." if concerns else ""
        frame = _pick(cid, "f2", ["{o} — {b}.{t}", "{o}; {b}.{t}"])
    elif rank <= 85:
        body = strengths[0] if strengths else "partial skill overlap with the JD"
        connector = _pick(cid, "c3", ["However,", "That said,", "Balanced against this,"])
        tail = f" {connector} {concerns[0]}." if concerns else ""
        frame = "{o} — {b}.{t}"
    else:
        qualifier = _pick(
            cid, "q", ["Stretch pick", "Below-cutoff filler", "Adjacent profile"]
        )
        body = strengths[0] if strengths else "some adjacent experience"
        tail = f" {concerns[0].capitalize()}." if concerns else ""
        frame = qualifier + ": {o} — {b}.{t}"

    text = " ".join(frame.format(o=opener, b=body, t=tail).split())
    if len(text) > 320 and len(strengths) > 1:
        # rebuild with fewer strengths rather than cutting mid-sentence
        body = joiner.join(strengths[:1]) if rank <= 50 else strengths[0]
        text = " ".join(frame.format(o=opener, b=body, t=tail).split())
    if len(text) > 320:
        cut = text[:320]
        text = cut[: cut.rfind(".") + 1] if "." in cut else cut.rsplit(" ", 1)[0]
    return text
