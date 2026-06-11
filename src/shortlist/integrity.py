"""Forensic integrity gate — the honeypot firewall.

The dataset plants ~80 honeypot candidates with subtly impossible profiles
(forced to relevance tier 0; ranking them is heavily penalized). A recruiter
sanity-checks a résumé before shortlisting; this module does the same, with one
discipline: every check targets a LOGICAL IMPOSSIBILITY, never an improbability.

Pool-frequency analysis showed why this matters: plausible-sounding checks like
"last_active before signup" (7,496 hits) or "skill duration exceeds career
length" (2,845 hits) are generator noise that would poison the gate with false
positives. The checks below each fire on <0.05% of the pool.
"""

from __future__ import annotations

from dataclasses import dataclass

from .loader import Candidate

# Calibrated thresholds (see scripts/calibrate_integrity.py for pool counts).
EXPERT_UNUSED_MIN = 3      # >=3 'expert' skills never used a single month
YOE_MISMATCH_YEARS = 4.0   # stated experience vs career-history date math
DURATION_MISMATCH_MONTHS = 3  # job's own duration field vs its start/end dates


@dataclass(slots=True)
class IntegrityFlag:
    check: str
    detail: str


def check_expert_unused(c: Candidate) -> IntegrityFlag | None:
    hits = [
        s["name"]
        for s in c.skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    ]
    if len(hits) >= EXPERT_UNUSED_MIN:
        return IntegrityFlag(
            "expert_skills_never_used",
            f"{len(hits)} skills claimed at expert proficiency with 0 months of use: "
            + ", ".join(hits[:5]),
        )
    return None


def check_yoe_vs_history(c: Candidate) -> IntegrityFlag | None:
    if not c.jobs:
        return None
    computed_years = c.history_months / 12.0
    if abs(computed_years - c.yoe) > YOE_MISMATCH_YEARS:
        return IntegrityFlag(
            "stated_experience_contradicts_history",
            f"profile claims {c.yoe:.1f} yrs but career history sums to "
            f"{computed_years:.1f} yrs",
        )
    return None


def check_duration_vs_dates(c: Candidate) -> IntegrityFlag | None:
    for j in c.jobs:
        if j.computed_months is None:
            continue
        if abs(j.computed_months - j.duration_months) > DURATION_MISMATCH_MONTHS:
            return IntegrityFlag(
                "job_duration_contradicts_dates",
                f"{j.title} at {j.company}: dates span {j.computed_months} months "
                f"but duration_months says {j.duration_months}",
            )
    return None


def check_current_flag(c: Candidate) -> IntegrityFlag | None:
    for j in c.jobs:
        if j.is_current and j.end_date:
            return IntegrityFlag(
                "current_job_has_end_date",
                f"{j.title} at {j.company} marked current but ends {j.end_date}",
            )
        if not j.is_current and not j.end_date:
            return IntegrityFlag(
                "ended_job_missing_end_date",
                f"{j.title} at {j.company} marked not-current with no end date",
            )
    return None


CHECKS = (
    check_expert_unused,
    check_yoe_vs_history,
    check_duration_vs_dates,
    check_current_flag,
)


def audit(c: Candidate) -> list[IntegrityFlag]:
    """All integrity violations for a candidate; empty list = clean."""
    return [flag for check in CHECKS if (flag := check(c)) is not None]


def is_honeypot(c: Candidate) -> bool:
    return any(check(c) is not None for check in CHECKS)
