"""Audit report — the glass box around the ranking.

Writes artifacts/audit_report.md with:
  - top-100 composition (titles, countries, experience, engagement),
  - per-candidate evidence cards for the head of the ranking: score waterfall,
    penalties, behavioral notes, top-10 stability under weight jitter,
  - a counterfactual per card ("what single change would move them most"),
plus artifacts/top100_breakdown.csv with every component for offline analysis.
"""

from __future__ import annotations

import csv
from bisect import bisect_left
from collections import Counter
from pathlib import Path

from .loader import Candidate
from .scoring import ScoreBreakdown

CARDS = 20


def _bar(x: float, width: int = 20) -> str:
    return "#" * round(x * width) or "."


def _counterfactual(b: ScoreBreakdown, finals_desc: list[float], rank: int) -> str | None:
    candidates = []
    mult = b.behavioral.multiplier
    if mult < 0.95:
        candidates.append(("behavioral signals at full strength", b.core_fit * b.penalty_factor * 1.10))
    if b.penalty_factor < 1.0:
        candidates.append(("JD penalty removed", b.core_fit * mult))
    if not candidates:
        return None
    fix, new_final = max(candidates, key=lambda kv: kv[1])
    asc = finals_desc[::-1]
    new_rank = len(finals_desc) - bisect_left(asc, new_final)
    if rank - new_rank < 5:
        return None
    return f"counterfactual: with {fix}, score {b.final:.3f} -> {new_final:.3f} (~rank {max(1, new_rank)})"


def write_audit(
    top: list[ScoreBreakdown],
    stability: dict[str, float],
    by_cid: dict[str, Candidate],
    spec: dict,
    out_path: Path,
    dense_available: bool = True,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cands = [by_cid[b.cid] for b in top]
    finals = [b.final for b in top]

    lines = [
        "# Shortlist.AI — ranking audit report",
        "",
        f"Job: {spec['meta'].get('title')}  |  candidates ranked: {len(top)}  |  "
        f"dense channel: {'on' if dense_available else 'off (lexical-only)'}",
        "",
        "## Top-100 composition",
        "",
        "| dimension | distribution |",
        "|---|---|",
        f"| titles | {dict(Counter(c.title for c in cands).most_common(8))} |",
        f"| countries | {dict(Counter(c.country for c in cands).most_common(5))} |",
        f"| experience | min {min(c.yoe for c in cands):.1f} / median "
        f"{sorted(c.yoe for c in cands)[len(cands)//2]:.1f} / max {max(c.yoe for c in cands):.1f} |",
        f"| active <=45d | {sum(1 for c in cands if (c.days_since_active or 999) <= 45)} / {len(cands)} |",
        f"| open to work | {sum(1 for c in cands if c.sig.get('open_to_work_flag'))} / {len(cands)} |",
        f"| notice <=30d | {sum(1 for c in cands if c.sig.get('notice_period_days', 99) <= 30)} / {len(cands)} |",
        "",
        f"## Evidence cards (top {CARDS})",
        "",
    ]

    for i, b in enumerate(top[:CARDS]):
        c = by_cid[b.cid]
        lines += [
            f"### #{i + 1}  {b.cid} — {c.title} @ {c.company} ({c.location}, {c.yoe:.1f} yrs)",
            "",
            "```",
        ]
        for name, val in b.components.items():
            lines.append(f"  {name:<8} {val:5.2f}  {_bar(val)}")
        lines.append(
            f"  core {b.core_fit:.3f} x penalty {b.penalty_factor:.2f} "
            f"x behavioral {b.behavioral.multiplier:.2f} = {b.final:.4f}"
        )
        lines.append("```")
        if b.penalties:
            lines.append(f"- penalties: {b.penalties}")
        if b.behavioral.notes:
            lines.append(f"- behavioral notes: {'; '.join(b.behavioral.notes)}")
        if b.cid in stability:
            lines.append(f"- top-10 stability across weight jitter: {stability[b.cid]:.0%}")
        cf = _counterfactual(b, finals, i + 1)
        if cf:
            lines.append(f"- {cf}")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    csv_path = out_path.parent / "top100_breakdown.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["rank", "candidate_id", "title", "country", "yoe", "lexical", "trust",
             "dense", "band", "career", "core_fit", "penalty_factor", "multiplier",
             "final", "penalties"]
        )
        for i, b in enumerate(top):
            c = by_cid[b.cid]
            w.writerow(
                [i + 1, b.cid, c.title, c.country, c.yoe,
                 b.components["lexical"], b.components["trust"], b.components["dense"],
                 b.components["band"], b.components["career"], b.core_fit,
                 b.penalty_factor, b.behavioral.multiplier, b.final,
                 ";".join(b.penalties) or "-"]
            )
