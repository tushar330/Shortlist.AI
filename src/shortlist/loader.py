"""Streaming candidate loader.

Parses the organizer JSONL (plain or gzipped) into compact slotted records so
the full 100K pool stays well under the 16 GB budget. Date arithmetic is done
once here; every downstream stage works on plain ints/floats.
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterator

try:
    import orjson as _json

    def _loads(line: str | bytes):
        return _json.loads(line)
except ImportError:  # pragma: no cover - orjson is in requirements
    import json as _json

    def _loads(line: str | bytes):
        return _json.loads(line)

# Fixed reference date so every run (including Stage-3 reproduction months from
# now) computes identical tenures and recency decays.
REF_DATE = date(2026, 6, 15)


def _months_between(start: str | None, end: str | None) -> int | None:
    """Whole months between two ISO dates; end=None means 'present' (REF_DATE)."""
    if not start:
        return None
    try:
        s = date.fromisoformat(start)
        e = date.fromisoformat(end) if end else REF_DATE
    except ValueError:
        return None
    return (e.year - s.year) * 12 + (e.month - s.month)


def _days_before_ref(iso: str | None) -> int | None:
    if not iso:
        return None
    try:
        return (REF_DATE - date.fromisoformat(iso)).days
    except ValueError:
        return None


@dataclass(slots=True)
class Job:
    company: str
    title: str
    start_date: str | None
    end_date: str | None
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str
    computed_months: int | None  # from start/end date arithmetic


@dataclass(slots=True)
class Candidate:
    cid: str
    headline: str
    summary: str
    location: str
    country: str
    yoe: float
    title: str
    company: str
    company_size: str
    industry: str
    jobs: list[Job]
    education: list[dict]
    skills: list[dict]
    certs: list[str]
    sig: dict
    narrative_lc: str = field(default="")
    days_since_active: int | None = field(default=None)

    @property
    def history_months(self) -> int:
        """Total career months by date arithmetic (jobs may overlap)."""
        return sum(max(j.computed_months or 0, 0) for j in self.jobs)


def _to_candidate(raw: dict) -> Candidate:
    p = raw["profile"]
    jobs = [
        Job(
            company=j.get("company", ""),
            title=j.get("title", ""),
            start_date=j.get("start_date"),
            end_date=j.get("end_date"),
            duration_months=j.get("duration_months", 0),
            is_current=bool(j.get("is_current")),
            industry=j.get("industry", ""),
            company_size=j.get("company_size", ""),
            description=j.get("description", ""),
            computed_months=_months_between(j.get("start_date"), j.get("end_date")),
        )
        for j in raw.get("career_history", [])
    ]
    sig = raw.get("redrob_signals", {})
    narrative = " ".join(
        [p.get("headline", ""), p.get("summary", "")] + [j.description for j in jobs]
    )
    return Candidate(
        cid=raw["candidate_id"],
        headline=p.get("headline", ""),
        summary=p.get("summary", ""),
        location=p.get("location", ""),
        country=p.get("country", ""),
        yoe=float(p.get("years_of_experience", 0.0)),
        title=p.get("current_title", ""),
        company=p.get("current_company", ""),
        company_size=p.get("current_company_size", ""),
        industry=p.get("current_industry", ""),
        jobs=jobs,
        education=raw.get("education", []),
        skills=raw.get("skills", []),
        certs=[c.get("name", "") for c in raw.get("certifications", [])],
        sig=sig,
        narrative_lc=narrative.lower(),
        days_since_active=_days_before_ref(sig.get("last_active_date")),
    )


def iter_candidates(path: str | Path) -> Iterator[Candidate]:
    path = Path(path)
    if path.suffix == ".gz":
        opener = lambda: io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    else:
        opener = lambda: open(path, encoding="utf-8")
    with opener() as f:
        for line in f:
            if line.strip():
                yield _to_candidate(_loads(line))


def load_candidates(path: str | Path) -> list[Candidate]:
    return list(iter_candidates(path))


def load_candidates_json(path: str | Path) -> list[Candidate]:
    """Load a pretty-printed JSON array (e.g. sample_candidates.json)."""
    raw = _loads(Path(path).read_bytes())
    return [_to_candidate(r) for r in raw]
