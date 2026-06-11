# Shortlist.AI — architecture and design decisions

This document explains how the system works and *why* each decision was made.
It is written to be defensible: every section ends with the question we expect
a reviewer to ask, and the answer we would give.

## The problem, restated

Rank 100,000 candidates for one job description, top-100 out, under hard
production constraints: ≤5 minutes, CPU-only, no network, 16 GB. The dataset is
adversarial — keyword stuffers, paraphrased strong profiles, behavioral twins,
and ~80 honeypot résumés with impossible facts (>10% of them in the top-100
disqualifies the submission).

The deeper observation: this is not a similarity problem, it is a *credibility
and decision* problem. What a recruiter wants is not "most similar profile" but
"highest expected value of reaching out". That reframing drives the whole
architecture:

    FinalScore = CoreFit × (1 − JD penalties) × behavioral multiplier
               ≈ fit × P(claims are real) × P(reachable & completes process)

## Pipeline

```
any JD ──► JD compiler ──► job_spec.yaml
                               │
candidates ──► loader ──► integrity gate ──► hybrid retrieval ──► scorer ──► explanations
               (3.5s)     (honeypot           (sparse ∪ dense)    (ensemble)   (reasoning,
                           firewall)                                            waterfalls,
                                                                                counterfactuals)
```

### 1. JD compiler (`src/shortlist/jd_compiler.py`)

Free-text JD → structured spec: must-have / nice-to-have facets (with the JD
sentence each came from), disqualifier detectors activated by the JD's own
"do not want" language, experience bands, locations, notice thresholds,
behavioral expectations. The vocabulary lives in `ontology.yaml`, shared with
the candidate-side evidence engine — both sides speak the same language.

> *Why not hardcode the one JD?* Because the system is a platform: paste a
> different JD in the sandbox and the ranking changes accordingly. The released
> JD is just the first input. (Try it: the repo includes a second test JD.)

### 2. Forensic integrity gate (`src/shortlist/integrity.py`)

Four impossibility-grade checks (≥3 "expert" skills with zero months of use;
stated experience contradicting career-history date math by >4 years; a job's
duration field contradicting its own dates by >3 months; `is_current`
contradictions). 66 of 100,000 candidates flagged — each logged with the
specific contradiction in `artifacts/honeypots_found.csv`.

> *Why these checks and not more?* We tested the tempting ones against pool
> frequency. "Last active before signup" fires on 7,496 candidates and "skill
> duration exceeds career length" on 2,845 — that's generator noise, and
> flagging it would exclude thousands of legitimate candidates. Every check we
> kept fires on <0.05% of the pool. Impossibility, not improbability.

### 3. Three evidence channels (`evidence.py`, `trust.py`, `dense.py`)

- **Lexical facet evidence** — curated ontology terms matched in the narrative,
  weighted by section: career descriptions > summary > headline ≫ skills list.
- **Trusted-skill coverage** — a listed skill counts only with corroboration:
  endorsements, platform assessments, narrative support. Claimed proficiency is
  ignored entirely; durations are nearly worthless (stuffers fake both).
  Assessment keys also *recover* canonical skills for paraphrased profiles.
- **Dense semantic similarity** — BGE-small embeddings (precomputed cache; the
  ranking run is pure numpy) against the JD's facet queries. This is the recall
  safety net for the candidate who writes "built the system that decides what
  users see first" without one buzzword.

> *Why is the skills array trusted least?* Pool evidence: a keyword stuffer
> (Graphic Designer) lists Pinecone/RAG/FAISS with plausible durations but 0–4
> endorsements, no assessments, and a narrative about stakeholder management.
> An honest strong profile carries 30–60 endorsements and describes the work.
> The discriminator is corroboration, so corroboration is the model.

### 4. Negative detectors (JD-activated)

Research-only careers without shipping evidence, CV/speech specialists without
IR exposure, consulting-only careers, title-chasers (escalation required — not
ordinary startup mobility), management drift, recent thin LLM-wrapper
experience. Each maps to an explicit "we do not want" sentence in the JD; a JD
without that language leaves the detector off.

### 5. Behavioral signal model (`signals.py`)

Availability (activity recency, open-to-work, notice fit), responsiveness
(response rate with the JD's own 10% floor, interview completion), logistics
(location/relocation/work-mode against the spec), credibility (verifications,
GitHub). Blended into a multiplier ∈ [0.55, 1.10].

> *Why a multiplier instead of an additive term?* The signals doc frames
> engagement as a modifier on fit, and multiplication has the right semantics:
> it cannot rescue a weak profile (0 × anything = 0) but decisively separates
> behavioral twins — identical paper profiles with different engagement.

### 6. Scoring + jitter ensemble (`scoring.py`)

CoreFit = 0.30 lexical + 0.25 trust + 0.15 dense + 0.10 experience-band +
0.20 career quality. The blend is evaluated under 25 lognormal weight
perturbations and aggregated by mean rank (Borda).

> *Why an ensemble?* With no leaderboard and three submissions, we cannot tune
> weights against feedback. The ensemble makes the ranking robust to the one
> thing we cannot validate — our exact weight choice — and the reported top-10
> stability tells us where the ordering is evidence-driven vs. weight-driven.

### 7. Cross-encoder re-rank (`cross_encoder.py`)

ms-marco-MiniLM-L-6-v2 (22M params, ONNX) over the top-200 only, blended at
35%, with a hard wall-clock budget and automatic abort. This is the
"re-ranking" tier of the architecture the JD itself sketches — scaled to CPU.

### 8. Explanations (`reasoning.py`, `report.py`)

Reasoning strings are rendered exclusively from profile fields and computed
features — hallucination is impossible by construction. Tone follows rank band;
phrasing varies deterministically per candidate id; the top concern is always
stated. The audit report adds score waterfalls, behavioral notes, jitter
stability, and a counterfactual per candidate ("with behavioral signals at
full strength: rank ~27 → ~12").

## Validation without a leaderboard

`eval/` contains a shadow ground truth: one synthetic labeled candidate per
trap persona, plus ordering invariants run as pytest (8 tests): tier-5s above
everything ≤ tier-2; the paraphrased tier-5 found; the stuffer buried; the
dormant twin materially below its active twin; the honeypot gate-flagged;
shadow NDCG floors. Every change to weights or vocabulary must keep these
green.

## Performance budget (measured)

| step | time |
|---|---|
| parse 100K candidates (465 MB) | ~4 s (orjson, slotted records) |
| integrity gate | ~1 s |
| retrieval (cache load + numpy cosine + screen) | ~10 s |
| deep scoring of shortlist | ~1–2 min |
| cross-encoder top-200 | ≤75 s (budget-gated) |
| reasoning + CSV + audit | ~5 s |

Pre-computation (embeddings, one-time, documented): ~30 min CPU.

## What we deliberately did NOT do

- **No LLM API calls at ranking time** — banned by the rules, and the right
  call anyway: per-candidate LLM scoring cannot scale to a 200K pool.
- **No local-LLM re-ranker** — a 7B model on CPU breaks the budget and adds
  opacity exactly where the evaluation demands explainability.
- **No trained learning-to-rank model** — there are no labels; pseudo-labels
  would launder our own heuristics through an opaque model.
- **No vector database service** — network is banned; numpy over 100K×384
  takes milliseconds.
- **No agent frameworks** — nothing to chain.
