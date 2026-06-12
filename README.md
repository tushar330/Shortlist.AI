# Shortlist.AI

An evidence-first, JD-agnostic candidate ranking engine built for the Redrob
**Intelligent Candidate Discovery & Ranking Challenge**.

Shortlist.AI reads a job description, compiles it into a structured job spec,
and ranks a 100,000-candidate pool by **expected hire value** — not keyword
similarity:

```
FinalScore = CoreFit × (1 − JD penalties) × behavioral multiplier
           ≈ fit × P(claims are real) × P(reachable & completes the process)
```

Every score decomposes into named evidence, so the system is fast (no LLM
calls at ranking time), trap-resistant, and fully explainable. See
[ARCHITECTURE.md](ARCHITECTURE.md) for every design decision and its rationale.

## Results at a glance

| | |
|---|---|
| Full 100K ranking run | **~2 minutes** wall-clock (budget: 5 min), CPU-only, offline |
| Peak memory | well under 16 GB (slotted records, streaming parse) |
| Honeypots in top-100 | **0** (66 caught pool-wide, each logged with its contradiction) |
| Offline invariant suite | 8/8 green (`pytest eval/`) |
| Top-100 composition | 94% India, median 6.4 yrs experience, all ML/search/recsys profiles |

## Pipeline

```
any JD ──► JD compiler ──► job_spec.yaml
                               │
candidates.jsonl ──► loader ──► forensic integrity gate ──► hybrid retrieval
                     (7 s)      (honeypot firewall, 66)     (sparse ∪ dense ∪ safety nets)
                                                                   │
                                              expected-hire scorer + 25-config
                                              weight-jitter ensemble (Borda)
                                                                   │
                                              cross-encoder re-rank (top-200,
                                              budget-gated, CPU)
                                                                   │
                                    reasoning + audit report + submission.csv
```

## Reproduce the submission

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use .venv/bin/activate on Linux
pip install -r requirements.txt

# One command. The repo ships the precomputed embedding cache, so this runs
# fully offline and finishes in ~2 minutes:
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv --audit
```

Pre-computation (only needed to regenerate the shipped cache, or for a new
candidate pool; one-time, ~3.5 h on a laptop CPU, network for model download):

```bash
python scripts/warm_models.py                                  # model caches
python scripts/precompute_embeddings.py ./data/candidates.jsonl
```

The ranking step itself loads the cache and never touches the network. Without
the cache and models the pipeline still runs in lexical-only mode (`--no-dense
--no-rerank` to force it).

## Verify

```bash
pytest eval/ -q                                  # persona ordering invariants
python scripts/validate_submission.py submission.csv   # official format validator
python scripts/calibrate_integrity.py ./data/candidates.jsonl  # honeypot audit
```

`artifacts/audit_report.md` (written by `--audit`) shows the top-100
composition, per-candidate score waterfalls, jitter stability and
counterfactuals. `artifacts/honeypots_found.csv` lists every excluded honeypot
with the specific impossibility that betrayed it.

## Sandbox app

```bash
pip install -r requirements-dev.txt
streamlit run app/streamlit_app.py
```

Upload a ≤100-candidate JSONL/JSON sample (the 50-candidate organizer sample is
preloaded), or paste a **different JD** to watch the spec compile and the
ranking reorder — the engine is JD-agnostic by construction.

## Repository layout

| Path | Purpose |
|------|---------|
| `rank.py` | Single-command entry point producing the submission CSV |
| `src/shortlist/` | Pipeline modules: jd_compiler, loader, integrity, evidence, trust, retrieval, dense, cross_encoder, signals, scoring, reasoning, report |
| `src/shortlist/ontology.yaml` | Shared vocabulary: JD facets, disqualifier detectors, reference data |
| `jd/` | Released JD + compiled `job_spec.yaml` |
| `scripts/` | Pre-computation, model warming, integrity calibration, official validator |
| `eval/` | Shadow ground-truth personas + 8 pytest ordering invariants |
| `app/streamlit_app.py` | Hosted sandbox (Streamlit Cloud / HF Spaces, free CPU tier) |
| `artifacts/` | Embedding cache (shipped), honeypot log, audit report |
| `ARCHITECTURE.md` | Design decisions with their rationale |
| `submission_metadata.yaml` | Portal metadata mirror |

## Compute-constraint compliance

| Constraint | Status |
|---|---|
| ≤ 5 min ranking runtime | ~2 min measured on an 8-core laptop |
| ≤ 16 GB RAM | ~2–3 GB peak |
| CPU only | ONNX int8 models; no torch, no GPU code paths |
| No network during ranking | doc+query vectors read from the shipped cache; models from local cache |
| ≤ 5 GB disk | cache is 71 MB |
