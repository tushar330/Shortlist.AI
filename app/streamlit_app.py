"""Shortlist.AI sandbox — run the ranking engine on a small candidate sample.

Deployable to Streamlit Cloud / HuggingFace Spaces (free CPU tier). Lets the
organizers (or anyone) verify the system end-to-end:

  1. upload a candidates JSONL/JSON sample (<=100 rows) or use the bundled one,
  2. optionally paste a DIFFERENT job description - the JD compiler builds a
     new spec on the fly, demonstrating the engine is JD-agnostic,
  3. get the ranked shortlist with per-candidate evidence cards and the CSV.

Run locally:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shortlist.dense import DenseChannel  # noqa: E402
from shortlist.evidence import EvidenceEngine  # noqa: E402
from shortlist.integrity import audit as integrity_audit  # noqa: E402
from shortlist.jd_compiler import JDCompiler  # noqa: E402
from shortlist.loader import _to_candidate  # noqa: E402
from shortlist.reasoning import reasoning_for  # noqa: E402
from shortlist.scoring import Scorer, ensemble_rank  # noqa: E402
from shortlist.signals import SignalModel  # noqa: E402
from shortlist.trust import TrustModel  # noqa: E402

st.set_page_config(page_title="Shortlist.AI", page_icon=":mag:", layout="wide")
st.title("Shortlist.AI — evidence-first candidate ranking")
st.caption(
    "Verify > understand > weigh > explain. CPU-only, no LLM calls at ranking time. "
    "Every score decomposes into named evidence."
)

MAX_CANDIDATES = 100


@st.cache_resource
def default_spec() -> dict:
    return yaml.safe_load(open(ROOT / "jd" / "job_spec.yaml", encoding="utf-8"))


@st.cache_resource
def dense_channel() -> DenseChannel:
    return DenseChannel(cache_path=None)  # sandbox embeds on the fly


with st.sidebar:
    st.header("1 - Job description")
    jd_mode = st.radio("Spec source", ["Bundled Senior AI Engineer JD", "Paste a different JD"])
    spec = default_spec()
    if jd_mode == "Paste a different JD":
        jd_text = st.text_area("Paste the JD text", height=260)
        if jd_text.strip():
            spec = JDCompiler().compile(jd_text, jd_name="pasted_jd")
            st.success(
                f"Compiled: {len(spec['requirements']['must_have'])} must-have facets, "
                f"{len(spec['disqualifiers'])} disqualifiers"
            )
    st.header("2 - Candidates")
    upload = st.file_uploader("JSONL or JSON array (<=100 candidates)", type=["jsonl", "json"])
    use_dense = st.toggle("Dense semantic channel", value=True)

raw_records: list[dict] = []
if upload is not None:
    data = upload.read().decode("utf-8")
    try:
        parsed = json.loads(data)
        raw_records = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        raw_records = [json.loads(l) for l in data.splitlines() if l.strip()]
else:
    sample = next(
        (p for p in (ROOT / "app" / "sample_candidates.json",
                     ROOT / "data" / "sample_candidates.json") if p.exists()),
        None,
    )
    if sample:
        raw_records = json.loads(sample.read_text(encoding="utf-8"))
        st.info(f"Using bundled sample ({len(raw_records)} candidates). Upload a file to replace it.")

if not raw_records:
    st.warning("Upload a candidate sample to begin.")
    st.stop()

raw_records = raw_records[:MAX_CANDIDATES]
candidates = [_to_candidate(r) for r in raw_records]

engine = EvidenceEngine(spec)
scorer = Scorer(spec, engine, TrustModel(spec), SignalModel(spec))

flagged = {c.cid: integrity_audit(c) for c in candidates}
clean = [c for c in candidates if not flagged[c.cid]]

dense_scores = None
if use_dense:
    with st.spinner("Embedding narratives (CPU)..."):
        dense_scores = dense_channel().scores(clean, spec)

breakdowns = [
    scorer.breakdown(c, dense_scores.get(c.cid) if dense_scores else None) for c in clean
]
ordered, stability = ensemble_rank(breakdowns, dense_available=bool(dense_scores))
by_cid = {c.cid: c for c in candidates}

n_flagged = sum(1 for v in flagged.values() if v)
col1, col2, col3 = st.columns(3)
col1.metric("Candidates", len(candidates))
col2.metric("Integrity-flagged (honeypots)", n_flagged)
col3.metric("Dense channel", "on" if dense_scores else "off")

if n_flagged:
    with st.expander(f"Honeypot firewall caught {n_flagged} candidate(s)"):
        for cid, flags in flagged.items():
            if flags:
                st.markdown(f"**{cid}** — " + "; ".join(f.detail for f in flags))

st.subheader("Ranked shortlist")
rows = []
for i, b in enumerate(ordered):
    c = by_cid[b.cid]
    rows.append(
        {
            "rank": i + 1,
            "candidate_id": b.cid,
            "title": c.title,
            "location": f"{c.location} ({c.country})",
            "yoe": c.yoe,
            "score": b.final,
            "reasoning": reasoning_for(b, c, i + 1, spec),
        }
    )
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False)
st.download_button("Download ranked CSV", csv_buf.getvalue(), "shortlist_ranked.csv", "text/csv")

st.subheader("Evidence cards")
for i, b in enumerate(ordered[:10]):
    c = by_cid[b.cid]
    with st.expander(f"#{i + 1}  {b.cid} — {c.title}, {c.location} ({c.yoe:.1f} yrs)"):
        wcol, pcol = st.columns([2, 3])
        with wcol:
            st.markdown("**Score waterfall**")
            comp_df = pd.DataFrame(
                {"component": list(b.components), "value": list(b.components.values())}
            )
            st.bar_chart(comp_df, x="component", y="value", height=200)
            st.markdown(
                f"core `{b.core_fit:.3f}` x penalties `{b.penalty_factor:.2f}` "
                f"x behavioral `{b.behavioral.multiplier:.2f}` = **`{b.final:.4f}`**"
            )
        with pcol:
            st.markdown("**Why**")
            st.write(reasoning_for(b, c, i + 1, spec))
            if b.penalties:
                st.markdown(f"**JD penalties:** {b.penalties}")
            if b.behavioral.notes:
                st.markdown("**Behavioral notes:** " + "; ".join(b.behavioral.notes))
            ev = sorted(b.facet_evidence.items(), key=lambda kv: -kv[1].score)[:4]
            st.markdown(
                "**Top facet evidence:** "
                + ", ".join(f"{f} {e.score:.2f} ({', '.join(e.terms[:2])})" for f, e in ev if e.score > 0)
            )
