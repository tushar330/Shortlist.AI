"""Shadow ground truth — synthetic labeled personas for offline evaluation.

There is no leaderboard and only three submissions, so we reverse-engineer the
dataset's persona design and synthesize one labeled candidate per trap class.
If the ranker cannot order THESE correctly, it has no business submitting.

Labels follow the challenge's tier scheme: 5 = ideal hire ... 0 = honeypot.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shortlist.loader import Candidate, Job

ACTIVE_SIG = {
    "profile_completeness_score": 92.0,
    "signup_date": "2023-04-10",
    "last_active_date": "2026-06-01",
    "open_to_work_flag": True,
    "profile_views_received_30d": 40,
    "applications_submitted_30d": 6,
    "recruiter_response_rate": 0.85,
    "avg_response_time_hours": 6.0,
    "skill_assessment_scores": {},
    "connection_count": 420,
    "endorsements_received": 180,
    "notice_period_days": 15,
    "expected_salary_range_inr_lpa": {"min": 35, "max": 55},
    "preferred_work_mode": "hybrid",
    "willing_to_relocate": True,
    "github_activity_score": 70.0,
    "search_appearance_30d": 80,
    "saved_by_recruiters_30d": 12,
    "interview_completion_rate": 0.95,
    "offer_acceptance_rate": 0.8,
    "verified_email": True,
    "verified_phone": True,
    "linkedin_connected": True,
}

RETRIEVAL_DESC = (
    "Designed and shipped the company's hybrid retrieval system combining BM25 with "
    "dense embeddings over 20M documents. Owned embedding model selection, index "
    "refresh pipelines, and the offline evaluation harness (NDCG, MRR) plus online "
    "A/B tests. Built the learning-to-rank re-ranker with LightGBM in production."
)
PARAPHRASED_DESC = (
    "Built and shipped the matching engine that decides which profiles appear when "
    "someone searches. Combined classic keyword scoring with vector representations "
    "of text so the system finds people who describe the same work in different "
    "words. Set up the measurement framework that tells us offline whether a new "
    "version actually orders results better, then confirmed with live experiments."
)
RESEARCH_DESC = (
    "Published research on neural ranking architectures at SIGIR and EMNLP. "
    "Investigated novel attention mechanisms for retrieval in an academic lab "
    "setting. Built experimental prototypes for paper evaluations."
)
CV_DESC = (
    "Developed object detection and image segmentation pipelines with YOLO and "
    "OpenCV for autonomous inspection. Trained GANs for data augmentation and "
    "deployed computer vision models to edge devices."
)
GENERIC_DESC = (
    "Responsible for stakeholder communication, team coordination and project "
    "delivery across multiple client engagements."
)
JAVA_DESC = (
    "Built and deployed Spring Boot microservices in production, owned REST API "
    "design, wrote integration tests, and optimized SQL queries for reporting."
)


def _job(title: str, company: str, months: int, start: str, end: str | None,
         desc: str, industry: str = "Technology", size: str = "201-500") -> Job:
    return Job(
        company=company, title=title, start_date=start, end_date=end,
        duration_months=months, is_current=end is None, industry=industry,
        company_size=size, description=desc,
        computed_months=months,
    )


def make(cid: str, title: str, country: str, location: str, yoe: float,
         jobs: list[Job], skills: list[dict], summary: str,
         sig_overrides: dict | None = None, education: list | None = None) -> Candidate:
    sig = dict(ACTIVE_SIG)
    if sig_overrides:
        sig.update(sig_overrides)
    narrative = " ".join([f"{title} | {summary}"] + [j.description for j in jobs])
    days = (
        (2026 - int(sig["last_active_date"][:4])) * 365
        + 166 - int(sig["last_active_date"][5:7]) * 30
    )
    return Candidate(
        cid=cid, headline=f"{title} | {summary[:60]}", summary=summary,
        location=location, country=country, yoe=yoe, title=title,
        company=jobs[0].company if jobs else "", company_size="201-500",
        industry=jobs[0].industry if jobs else "", jobs=jobs,
        education=education or [{"institution": "IIT", "degree": "B.Tech",
                                 "field_of_study": "CS", "start_year": 2014,
                                 "end_year": 2018, "tier": "tier_1"}],
        skills=skills, certs=[], sig=sig,
        narrative_lc=narrative.lower(),
        days_since_active=max(0, (2026 * 12 + 6) * 30 + 15 - (int(sig["last_active_date"][:4]) * 12 + int(sig["last_active_date"][5:7])) * 30 - int(sig["last_active_date"][8:10])),
    )


SKILLS_STRONG = [
    {"name": "Embeddings", "proficiency": "expert", "endorsements": 45, "duration_months": 60},
    {"name": "FAISS", "proficiency": "advanced", "endorsements": 30, "duration_months": 40},
    {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 84},
    {"name": "Learning-to-Rank", "proficiency": "advanced", "endorsements": 25, "duration_months": 36},
]
SKILLS_PARAPHRASED = [
    {"name": "Vector Representations", "proficiency": "advanced", "endorsements": 28, "duration_months": 48},
    {"name": "Search & Discovery", "proficiency": "expert", "endorsements": 35, "duration_months": 60},
    {"name": "Python", "proficiency": "expert", "endorsements": 41, "duration_months": 80},
]
SKILLS_STUFFED = [
    {"name": n, "proficiency": "advanced", "endorsements": e, "duration_months": d}
    for n, e, d in [
        ("RAG", 2, 14), ("Pinecone", 1, 11), ("Embeddings", 3, 16), ("FAISS", 0, 9),
        ("Fine-tuning LLMs", 2, 12), ("Information Retrieval", 1, 15),
        ("Semantic Search", 2, 13), ("Vector Databases", 0, 10),
    ]
]
SKILLS_JAVA = [
    {"name": "Java", "proficiency": "expert", "endorsements": 40, "duration_months": 90},
    {"name": "Spring Boot", "proficiency": "advanced", "endorsements": 25, "duration_months": 70},
    {"name": "SQL", "proficiency": "advanced", "endorsements": 30, "duration_months": 90},
]


def build_shadow_pool() -> list[tuple[Candidate, int]]:
    """Returns (candidate, relevance_tier) pairs."""
    pool: list[tuple[Candidate, int]] = []

    tier5 = make(
        "SHDW_0000001", "Senior ML Engineer", "India", "Pune, Maharashtra", 7.0,
        [_job("Senior ML Engineer", "Flipkart", 30, "2023-12-15", None, RETRIEVAL_DESC,
              "E-commerce", "10001+"),
         _job("ML Engineer", "Meesho", 36, "2020-12-01", "2023-12-01", RETRIEVAL_DESC,
              "E-commerce", "1001-5000"),
         _job("Software Engineer", "Zeta", 18, "2019-06-01", "2020-12-01", JAVA_DESC)],
        SKILLS_STRONG,
        "ML engineer focused on search, retrieval and ranking systems in production.",
    )
    pool.append((tier5, 5))

    paraphrased = make(
        "SHDW_0000002", "Senior Software Engineer", "India", "Hyderabad, Telangana", 6.5,
        [_job("Senior Software Engineer", "Swiggy", 28, "2024-02-01", None, PARAPHRASED_DESC,
              "Food Delivery", "5001-10000"),
         _job("Software Engineer", "Razorpay", 40, "2020-10-01", "2024-02-01", PARAPHRASED_DESC,
              "Fintech", "1001-5000")],
        SKILLS_PARAPHRASED,
        "Engineer who builds the systems that decide what users see first.",
        sig_overrides={"skill_assessment_scores": {"Embeddings": 88.0, "NDCG": 80.0}},
    )
    pool.append((paraphrased, 5))

    tier4 = make(
        "SHDW_0000003", "ML Engineer", "India", "Mumbai, Maharashtra", 5.5,
        [_job("ML Engineer", "Nykaa", 36, "2023-06-01", None, RETRIEVAL_DESC,
              "E-commerce", "1001-5000"),
         _job("Data Scientist", "Fractal", 30, "2020-12-01", "2023-06-01", RETRIEVAL_DESC,
              "Analytics", "5001-10000")],
        SKILLS_STRONG,
        "ML engineer with production ranking experience.",
        sig_overrides={"notice_period_days": 75, "recruiter_response_rate": 0.5},
    )
    pool.append((tier4, 4))

    tier3 = make(
        "SHDW_0000004", "Data Engineer", "India", "Bengaluru, Karnataka", 6.0,
        [_job("Data Engineer", "PhonePe", 48, "2022-06-01", None,
              "Built feature pipelines feeding the recommendation models; owned Spark "
              "jobs and the data quality framework. Worked closely with the ranking "
              "team on offline metrics.", "Fintech", "1001-5000")],
        [{"name": "Spark", "proficiency": "expert", "endorsements": 38, "duration_months": 70},
         {"name": "Python", "proficiency": "advanced", "endorsements": 30, "duration_months": 72}],
        "Data engineer adjacent to ML ranking teams.",
    )
    pool.append((tier3, 3))

    java = make(
        "SHDW_0000005", "Java Developer", "India", "Chennai, Tamil Nadu", 7.0,
        [_job("Java Developer", "Freshworks", 60, "2021-06-01", None, JAVA_DESC,
              "SaaS", "1001-5000")],
        SKILLS_JAVA,
        "Backend developer building microservices.",
    )
    pool.append((java, 2))

    nontech = make(
        "SHDW_0000006", "HR Manager", "India", "Delhi", 8.0,
        [_job("HR Manager", "Randstad", 70, "2020-08-01", None, GENERIC_DESC,
              "Staffing", "10001+")],
        [{"name": "Recruitment", "proficiency": "expert", "endorsements": 30, "duration_months": 90}],
        "HR professional managing talent acquisition.",
    )
    pool.append((nontech, 1))

    stuffer = make(
        "SHDW_0000007", "Marketing Manager", "India", "Gurgaon, Haryana", 7.5,
        [_job("Marketing Manager", "BrandCo", 60, "2021-06-01", None, GENERIC_DESC,
              "Marketing", "201-500")],
        SKILLS_STUFFED,
        "Marketing professional with 7.5+ years of experience driving outcomes in my domain.",
    )
    pool.append((stuffer, 1))

    honeypot = make(
        "SHDW_0000008", "Senior AI Engineer", "India", "Noida, Uttar Pradesh", 9.0,
        [_job("Senior AI Engineer", "DeepHire", 14, "2025-04-01", None, RETRIEVAL_DESC)],
        [{"name": n, "proficiency": "expert", "endorsements": 5, "duration_months": 0}
         for n in ("RAG", "Pinecone", "Embeddings", "FAISS")] + SKILLS_STRONG[2:],
        "Senior AI engineer with deep retrieval experience.",
    )  # claims 9 yrs; history shows 14 months; 4 expert skills never used
    pool.append((honeypot, 0))

    research = make(
        "SHDW_0000009", "AI Research Engineer", "India", "Bengaluru, Karnataka", 6.0,
        [_job("AI Research Engineer", "IISc AI Lab", 40, "2023-02-01", None, RESEARCH_DESC,
              "Research", "51-200"),
         _job("Research Assistant", "IIT Madras", 36, "2020-02-01", "2023-02-01", RESEARCH_DESC,
              "Academia", "10001+")],
        SKILLS_STRONG,
        "Researcher exploring neural information retrieval.",
    )
    pool.append((research, 2))

    cv_eng = make(
        "SHDW_0000010", "Computer Vision Engineer", "India", "Pune, Maharashtra", 6.0,
        [_job("Computer Vision Engineer", "SightTech", 48, "2022-06-01", None, CV_DESC)],
        [{"name": "OpenCV", "proficiency": "expert", "endorsements": 40, "duration_months": 60},
         {"name": "Python", "proficiency": "expert", "endorsements": 45, "duration_months": 72}],
        "Computer vision specialist for industrial inspection.",
    )
    pool.append((cv_eng, 2))

    consulting = make(
        "SHDW_0000011", "Software Engineer", "India", "Hyderabad, Telangana", 6.5,
        [_job("Software Engineer", "TCS", 40, "2023-02-01", None, JAVA_DESC, "IT Services", "10001+"),
         _job("Systems Engineer", "Infosys", 38, "2019-12-01", "2023-02-01", JAVA_DESC,
              "IT Services", "10001+")],
        SKILLS_JAVA,
        "Software engineer delivering enterprise client projects.",
    )
    pool.append((consulting, 2))

    dormant_twin = make(
        "SHDW_0000012", "Senior ML Engineer", "India", "Pune, Maharashtra", 7.0,
        [_job("Senior ML Engineer", "Flipkart", 30, "2023-12-15", None, RETRIEVAL_DESC,
              "E-commerce", "10001+"),
         _job("ML Engineer", "Meesho", 36, "2020-12-01", "2023-12-01", RETRIEVAL_DESC,
              "E-commerce", "1001-5000")],
        SKILLS_STRONG,
        "ML engineer focused on search, retrieval and ranking systems in production.",
        sig_overrides={
            "last_active_date": "2025-09-20",
            "recruiter_response_rate": 0.04,
            "open_to_work_flag": False,
            "avg_response_time_hours": 160.0,
            "interview_completion_rate": 0.4,
        },
    )
    pool.append((dormant_twin, 3))  # same paper profile as tier5; signals say unreachable

    return pool
