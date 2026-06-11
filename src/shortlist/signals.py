"""Behavioral signal model — is this candidate actually hireable?

Converts the 23 Redrob platform signals into four interpretable components and
one multiplier applied to fit. The framing is decision-theoretic: a ranking for
a recruiter should approximate expected hire value,

    fit  ×  P(reachable)  ×  P(completes the process)  ×  logistics feasibility

so a perfect-on-paper candidate who is dormant with a 5% response rate (the
JD's own example) sinks decisively, while no single weak signal can destroy a
strong profile (multiplier floor 0.55).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .jd_compiler import load_ontology
from .loader import Candidate

COMPONENT_WEIGHTS = {
    "availability": 0.40,
    "responsiveness": 0.30,
    "logistics": 0.20,
    "credibility": 0.10,
}
MULTIPLIER_FLOOR, MULTIPLIER_CEIL = 0.55, 1.10


@dataclass(slots=True)
class BehavioralProfile:
    availability: float
    responsiveness: float
    logistics: float
    credibility: float
    multiplier: float
    notes: list[str]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class SignalModel:
    def __init__(self, spec: dict, ontology: dict | None = None):
        self.spec = spec
        self.ontology = ontology or load_ontology()
        cities = self.ontology["india_cities"]
        loc = spec["locations"]
        self._preferred_aliases = [
            a for c in loc.get("preferred_cities", []) for a in cities.get(c, [])
        ]
        self._welcome_aliases = [
            a for c in loc.get("welcome_cities", []) for a in cities.get(c, [])
        ]
        self._country = (loc.get("country_preference") or "").lower()
        self._floor = spec["behavioral"].get("response_rate_floor") or 0.0
        notice = spec.get("notice", {})
        self._notice_loved = notice.get("loved_max_days") or 30
        self._notice_buyout = self._notice_loved + (notice.get("buyout_max_days") or 0)

    # ------------------------------------------------------------ components
    def _availability(self, c: Candidate, notes: list[str]) -> float:
        sig = c.sig
        recency = math.exp(-(c.days_since_active or 365) / 45.0)
        open_flag = 1.0 if sig.get("open_to_work_flag") else 0.25
        notice = sig.get("notice_period_days", 90)
        if notice <= self._notice_loved:
            notice_fit = 1.0
        elif notice <= self._notice_buyout:
            notice_fit = 0.8
        elif notice <= 90:
            notice_fit = 0.55
        else:
            notice_fit = 0.35
        if (c.days_since_active or 0) > 120:
            notes.append(f"inactive for {c.days_since_active} days")
        if notice > self._notice_buyout:
            notes.append(f"{notice}-day notice period")
        return _clamp(0.45 * recency + 0.30 * open_flag + 0.25 * notice_fit)

    def _responsiveness(self, c: Candidate, notes: list[str]) -> float:
        sig = c.sig
        rate = _clamp(sig.get("recruiter_response_rate", 0.0))
        speed = math.exp(-sig.get("avg_response_time_hours", 96.0) / 48.0)
        completion = _clamp(sig.get("interview_completion_rate", 0.5))
        score = 0.55 * rate + 0.15 * speed + 0.30 * completion
        if rate < self._floor:
            score *= 0.5
            notes.append(f"recruiter response rate only {rate:.0%}")
        return _clamp(score)

    def _logistics(self, c: Candidate, notes: list[str]) -> float:
        sig = c.sig
        loc = c.location.lower()
        in_country = self._country and self._country in c.country.lower()
        reloc = bool(sig.get("willing_to_relocate"))
        if any(a in loc for a in self._preferred_aliases):
            place = 1.0
        elif any(a in loc for a in self._welcome_aliases):
            place = 0.9
        elif in_country:
            place = 0.75 if reloc else 0.6
        elif reloc:
            place = 0.4
            notes.append(f"based in {c.country}, willing to relocate")
        else:
            place = 0.15
            notes.append(f"based in {c.country}, not willing to relocate")

        mode = sig.get("preferred_work_mode", "flexible")
        job_mode = self.spec["locations"].get("work_mode") or "hybrid"
        if mode in ("flexible", job_mode):
            mode_fit = 1.0
        elif job_mode == "hybrid" and mode == "onsite":
            mode_fit = 0.9
        else:
            mode_fit = 0.6
            notes.append(f"prefers {mode} work")
        return _clamp(0.8 * place + 0.2 * mode_fit)

    def _credibility(self, c: Candidate, notes: list[str]) -> float:
        sig = c.sig
        verified = 0.5 * bool(sig.get("verified_email")) + 0.5 * bool(sig.get("verified_phone"))
        completeness = _clamp(sig.get("profile_completeness_score", 0) / 100.0)
        gh = sig.get("github_activity_score", -1)
        github = _clamp(gh / 100.0) if gh >= 0 else 0.3
        linked = 1.0 if sig.get("linkedin_connected") else 0.5
        return _clamp(0.3 * verified + 0.3 * completeness + 0.25 * github + 0.15 * linked)

    # ----------------------------------------------------------------- public
    def profile(self, c: Candidate) -> BehavioralProfile:
        notes: list[str] = []
        comp = {
            "availability": self._availability(c, notes),
            "responsiveness": self._responsiveness(c, notes),
            "logistics": self._logistics(c, notes),
            "credibility": self._credibility(c, notes),
        }
        blended = sum(comp[k] * w for k, w in COMPONENT_WEIGHTS.items())
        multiplier = MULTIPLIER_FLOOR + (MULTIPLIER_CEIL - MULTIPLIER_FLOOR) * blended
        return BehavioralProfile(
            **{k: round(v, 4) for k, v in comp.items()},
            multiplier=round(multiplier, 4),
            notes=notes,
        )
