"""Calibrate the integrity gate against the full candidate pool.

Reports per-check fire counts (each must stay impossibility-rare) and writes
artifacts/honeypots_found.csv listing every flagged candidate with the specific
contradiction that betrayed it.

Usage:
    python scripts/calibrate_integrity.py <path-to-candidates.jsonl>
"""

import csv
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shortlist.integrity import audit
from shortlist.loader import iter_candidates


def main(candidates_path: str) -> None:
    t0 = time.time()
    per_check = Counter()
    flagged: list[tuple[str, str, str, list]] = []
    n = 0

    for c in iter_candidates(candidates_path):
        n += 1
        flags = audit(c)
        if flags:
            for f in flags:
                per_check[f.check] += 1
            flagged.append((c.cid, c.title, c.country, flags))

    print(f"scanned {n:,} candidates in {time.time() - t0:.1f}s")
    print(f"unique flagged: {len(flagged)}  ({100 * len(flagged) / n:.3f}% of pool)")
    for check, count in per_check.most_common():
        print(f"  {check}: {count}  ({100 * count / n:.3f}%)")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "honeypots_found.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "current_title", "country", "checks", "details"])
        for cid, title, country, flags in sorted(flagged):
            w.writerow(
                [
                    cid,
                    title,
                    country,
                    "; ".join(fl.check for fl in flags),
                    " | ".join(fl.detail for fl in flags),
                ]
            )
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
