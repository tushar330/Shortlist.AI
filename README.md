# Shortlist.AI

An evidence-first, JD-agnostic candidate ranking engine built for the Redrob
**Intelligent Candidate Discovery & Ranking Challenge**.

Shortlist.AI reads a job description, compiles it into a structured job spec, and
ranks a 100,000-candidate pool by **expected hire value** — not keyword similarity.
Every score is decomposable into named evidence from the candidate's profile, so
the system is fast (no LLM calls at ranking time), trap-resistant, and fully
explainable.

## Pipeline

```
any JD ──► JD compiler ──► job_spec.yaml ─┐
                                          ▼
candidates.jsonl ──► loader ──► forensic integrity gate ──► hybrid retrieval
                                 (honeypot firewall)         (sparse + dense)
                                                                   ▼
                                            cross-encoder re-rank (top-200)
                                                                   ▼
                       expected-hire scorer  =  fit × reachability × process-completion
                                                                   ▼
                          explanations: reasoning + score waterfall + counterfactuals
                                                                   ▼
                              submission.csv + audit_report.md
```

## Reproduce the submission

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows (use bin/activate on Linux)
pip install -r requirements.txt
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

The ranking step runs CPU-only, offline, in under 5 minutes within 16 GB RAM.
Dense-channel embeddings are precomputed once by `scripts/precompute_embeddings.py`
(documented pre-computation; the ranking step only loads the cache).

## Repository layout

| Path | Purpose |
|------|---------|
| `rank.py` | Single-command entry point that produces the submission CSV |
| `src/shortlist/` | Pipeline modules (loader, integrity, evidence, retrieval, scoring, reasoning) |
| `jd/` | Job description + compiled `job_spec.yaml` |
| `scripts/` | Offline pre-computation (embeddings cache) |
| `eval/` | Shadow ground-truth simulator, persona invariant tests, ablations |
| `app/` | Streamlit sandbox (upload candidates / paste a JD → ranked output) |
| `artifacts/` | Generated: honeypot log, audit report, embedding cache |

Status: under active development — see commit history for phase-by-phase progress.
