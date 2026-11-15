from app.interviewer import InterviewContext
from app.planner import _fake_plan, retrieve
from app.rag import BM25, chunk_text

JD = """Requirements:
- Strong PostgreSQL: schema design, indexing, query tuning
- Caching and queues with Redis and BullMQ; idempotent background jobs
- Observability: structured logging, tracing, on-call incident response"""
RESUME = """Arjun Mehta — Backend Engineer
Reduced p95 latency of a report query from 9s to 400ms with composite PostgreSQL indexes.
Introduced Redis-backed idempotency keys for retries in BullMQ workers.
Led incident response for a queue backlog outage and added alerting."""
BANK = """1. How would you find and fix a slow PostgreSQL query in production?
2. How do you make a BullMQ job idempotent when it can be retried?"""


def test_chunks_are_line_based_and_strip_bullets():
    chunks = chunk_text(BANK, "question_bank")
    assert [c.text[:6] for c in chunks] == ["How wo", "How do"]


def test_bm25_finds_matching_resume_line():
    index = BM25(chunk_text(RESUME, "resume"))
    top, _ = index.search("PostgreSQL indexing query tuning", k=1)[0]
    assert "PostgreSQL indexes" in top.text


def test_retrieve_pairs_requirements_with_evidence():
    reqs = retrieve({"jd": JD, "resume": RESUME, "question_bank": BANK})
    assert all("Requirements" not in r.text for r in reqs)
    pg = next(r for r in reqs if "PostgreSQL" in r.text)
    assert "PostgreSQL" in pg.resume_evidence[0] and "PostgreSQL" in pg.bank_questions[0]


def test_fake_plan_has_no_duplicates_and_uses_sources():
    ctx = InterviewContext(interview_id="i", candidate_name="Arjun", company_name="Acme", job_title="Backend", question_count=4)
    plan = _fake_plan(ctx, retrieve({"jd": JD, "resume": RESUME, "question_bank": BANK}))
    questions = [q.question for q in plan]
    assert len(questions) == len(set(questions)) == 4
    assert {"question_bank", "resume"} <= {q.source for q in plan}
