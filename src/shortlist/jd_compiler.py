"""JD compiler: turns a free-text job description into a structured job spec.

The spec is the contract consumed by every downstream stage (evidence engine,
scorer, reasoning). Each extracted requirement carries the JD sentence it came
from, so the system can always show *why* it believes the job needs something.
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

import yaml

ONTOLOGY_PATH = Path(__file__).parent / "ontology.yaml"

# Scope headers — short lines that switch what the following text means.
SCOPE_CUES = {
    "must": ("absolutely need", "must have", "must-have", "you need", "hard requirements"),
    "nice": ("like you to have", "won't reject", "nice to have", "nice-to-have", "bonus", "good to have"),
    "negative": ("do not want", "don't want", "not want"),
    "ideal": ("ideal candidate", "between the lines"),
    "logistics": ("location, comp", "logistics", "on location", "notice period"),
}

# Sentence-level cues that mark a disqualifier wherever it appears.
NEGATIVE_SENTENCE_CUES = (
    "will not move forward",
    "not move forward",
    "we're not a fit",
    "not a fit",
    "we will probably not",
)

WEIGHTS = {"must": 1.0, "nice": 0.4, "contextual": 0.15}


def load_ontology(path: Path = ONTOLOGY_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _term_pattern(term: str) -> re.Pattern:
    # Word boundaries where the term edges are alphanumeric; substring otherwise
    # (handles "a/b test", "precision@", "map@").
    escaped = re.escape(term)
    prefix = r"\b" if term[0].isalnum() else ""
    suffix = r"\b" if term[-1].isalnum() else ""
    return re.compile(prefix + escaped + suffix, re.IGNORECASE)


class JDCompiler:
    def __init__(self, ontology: dict | None = None):
        self.ontology = ontology or load_ontology()
        self._facet_patterns = {
            facet: [(t, _term_pattern(t)) for t in spec["terms"]]
            for facet, spec in self.ontology["facets"].items()
        }

    # ------------------------------------------------------------------ scopes
    def _scoped_lines(self, text: str) -> list[tuple[str, str]]:
        """Tag each line with the scope set by the most recent header-like line."""
        scoped, current = [], "body"
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            for scope, cues in SCOPE_CUES.items():
                if any(c in low for c in cues) and len(line) < 90:
                    current = scope
                    break
            scoped.append((current, line))
        return scoped

    # ------------------------------------------------------------- extraction
    def _extract_facets(self, scoped: list[tuple[str, str]]) -> dict[str, dict]:
        """Map facet -> {requirement level, matched terms, evidence sentences}."""
        found: dict[str, dict] = {}
        rank = {"must": 3, "nice": 2, "contextual": 1}
        for scope, line in scoped:
            level = scope if scope in ("must", "nice") else "contextual"
            for sentence in _split_sentences(line):
                for facet, patterns in self._facet_patterns.items():
                    terms = [t for t, p in patterns if p.search(sentence)]
                    if not terms:
                        continue
                    entry = found.setdefault(
                        facet, {"level": level, "terms": [], "evidence": []}
                    )
                    if rank[level] > rank[entry["level"]]:
                        entry["level"] = level
                    entry["terms"] = sorted(set(entry["terms"]) | set(terms))
                    if len(entry["evidence"]) < 3 and sentence not in entry["evidence"]:
                        entry["evidence"].append(sentence[:240])
        return found

    def _extract_disqualifiers(self, scoped: list[tuple[str, str]]) -> list[dict]:
        detectors = self.ontology["disqualifier_detectors"]
        active: dict[str, dict] = {}
        for scope, line in scoped:
            low = line.lower()
            is_negative = scope == "negative" or any(
                c in low for c in NEGATIVE_SENTENCE_CUES
            )
            if not is_negative:
                continue
            for name, det in detectors.items():
                if any(cue in low for cue in det["jd_cues"]):
                    entry = active.setdefault(
                        name, {"type": name, "label": det["label"], "evidence": []}
                    )
                    if len(entry["evidence"]) < 2:
                        entry["evidence"].append(line.strip()[:240])
        return list(active.values())

    def _extract_experience(self, text: str, scoped) -> dict:
        band = {"min_years": None, "max_years": None, "ideal_min": None, "ideal_max": None}
        rng = re.compile(r"(\d{1,2})\s*[–\-—]\s*(\d{1,2})\s+years?", re.IGNORECASE)
        m = rng.search(text)
        if m:
            band["min_years"], band["max_years"] = int(m.group(1)), int(m.group(2))
        ideal_text = " ".join(l for s, l in scoped if s == "ideal")
        m = rng.search(ideal_text)
        if m:
            band["ideal_min"], band["ideal_max"] = int(m.group(1)), int(m.group(2))
        band["hard_requirement"] = "range, not a requirement" not in text.lower()
        return band

    def _extract_locations(self, text: str) -> dict:
        low = text.lower()
        cities = self.ontology["india_cities"]
        loc_line = ""
        for line in text.splitlines():
            if line.lower().startswith("location"):
                loc_line = line.lower()
                break
        preferred = [c for c, aliases in cities.items() if any(a in loc_line for a in aliases)]
        welcome_lines = [l.lower() for l in text.splitlines() if "welcome" in l.lower()]
        welcome = sorted(
            {
                c
                for c, aliases in cities.items()
                for l in welcome_lines
                if any(a in l for a in aliases)
            }
            - set(preferred)
        )
        return {
            "country_preference": "India" if "india" in low else None,
            "preferred_cities": preferred,
            "welcome_cities": welcome,
            "relocation_ok": "relocat" in low,
            "visa_sponsorship": not (
                "don't sponsor" in low or "do not sponsor" in low or "no visa" in low
            ),
            "work_mode": next(
                (m for m in ("hybrid", "remote", "onsite") if m in low), None
            ),
        }

    def _extract_notice(self, text: str) -> dict:
        low = text.lower()
        notice = {"loved_max_days": None, "buyout_max_days": None, "soft_max_days": None}
        m = re.search(r"sub-(\d+)[- ]day notice", low)
        if m:
            notice["loved_max_days"] = int(m.group(1))
        m = re.search(r"buy out up to (\d+)", low)
        if m:
            notice["buyout_max_days"] = int(m.group(1))
        m = re.search(r"(\d+)\+\s*day notice", low)
        if m:
            notice["soft_max_days"] = int(m.group(1))
        return notice

    def _extract_behavioral(self, text: str) -> dict:
        low = text.lower()
        cues = ("hasn't logged in", "response rate", "active on", "actually available")
        wants_active = any(c in low for c in cues)
        return {
            "requires_active_candidate": wants_active,
            "response_rate_floor": 0.10 if wants_active else None,
            "open_to_work_preferred": "job market" in low or "open to" in low,
        }

    # ------------------------------------------------------------------ public
    def compile(self, jd_text: str, jd_name: str = "job_description") -> dict:
        scoped = self._scoped_lines(jd_text)
        facets = self._extract_facets(scoped)
        ont_facets = self.ontology["facets"]

        requirements = {"must_have": [], "nice_to_have": [], "contextual": []}
        bucket = {"must": "must_have", "nice": "nice_to_have", "contextual": "contextual"}
        for facet, info in sorted(facets.items()):
            requirements[bucket[info["level"]]].append(
                {
                    "facet": facet,
                    "label": ont_facets[facet]["label"],
                    "weight": WEIGHTS[info["level"]],
                    "jd_terms_matched": info["terms"],
                    "jd_evidence": info["evidence"],
                }
            )

        ideal_lines = [l for s, l in scoped if s == "ideal"]
        title_line = next((l for _, l in scoped if "job description" in l.lower()), "")

        facet_queries = [
            f"{r['label']}. {r['jd_evidence'][0]}"
            for r in requirements["must_have"] + requirements["nice_to_have"]
            if r["jd_evidence"]
        ]

        return {
            "meta": {
                "source": jd_name,
                "title": title_line.split(":", 1)[-1].strip() or None,
                "compiled_on": date.today().isoformat(),
                "compiler_version": 1,
            },
            "experience": self._extract_experience(jd_text, scoped),
            "locations": self._extract_locations(jd_text),
            "notice": self._extract_notice(jd_text),
            "behavioral": self._extract_behavioral(jd_text),
            "requirements": requirements,
            "disqualifiers": self._extract_disqualifiers(scoped),
            "facet_queries": facet_queries,
            "ideal_candidate_paragraph": " ".join(ideal_lines)[:1200] or None,
        }


def summarize(spec: dict) -> str:
    lines = [f"Compiled JD spec — {spec['meta'].get('title') or spec['meta']['source']}"]
    exp = spec["experience"]
    lines.append(
        f"  experience: {exp['min_years']}-{exp['max_years']} yrs"
        f" (ideal {exp['ideal_min']}-{exp['ideal_max']})"
    )
    loc = spec["locations"]
    lines.append(
        f"  location: {loc['country_preference']} | preferred {loc['preferred_cities']}"
        f" | welcome {loc['welcome_cities']} | mode {loc['work_mode']}"
        f" | visa={loc['visa_sponsorship']}"
    )
    lines.append(f"  notice: {spec['notice']}")
    for level in ("must_have", "nice_to_have", "contextual"):
        names = [r["facet"] for r in spec["requirements"][level]]
        lines.append(f"  {level} ({len(names)}): {', '.join(names)}")
    lines.append(
        f"  disqualifiers ({len(spec['disqualifiers'])}): "
        + ", ".join(d["type"] for d in spec["disqualifiers"])
    )
    lines.append(f"  behavioral: {spec['behavioral']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Compile a JD into a structured job spec")
    ap.add_argument("jd_file", help="Path to the job description (text/markdown)")
    ap.add_argument("-o", "--out", help="Output YAML path", default=None)
    ap.add_argument("--summary", action="store_true", help="Print a human summary")
    args = ap.parse_args()

    text = Path(args.jd_file).read_text(encoding="utf-8")
    spec = JDCompiler().compile(text, jd_name=Path(args.jd_file).stem)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True, width=100)
        print(f"spec written to {args.out}")
    if args.summary or not args.out:
        print(summarize(spec))


if __name__ == "__main__":
    main()
