"""Builds a per-candidate question plan with retrieval over the JD, resume and question banks."""

import logging
import re

from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .config import get_settings
from .interviewer import InterviewContext, PlannedQuestion
from .rag import BM25, Chunk, chunk_text
from .resume import fetch_document

log = logging.getLogger(__name__)


class Requirement(BaseModel):
    text: str
    resume_evidence: list[str]
    bank_questions: list[str]


class _Plan(BaseModel):
    questions: list[PlannedQuestion] = Field(description="Ordered interview plan")


async def gather_documents(ctx: InterviewContext) -> dict[str, str]:
    jd = ctx.job_description
    if ctx.jd_document_key:
        jd_file = await run_in_threadpool(fetch_document, ctx.jd_document_key)
        if jd_file:
            jd = f"{jd}\n\n{jd_file}".strip()
    banks = []
    for key in ctx.question_bank_keys:
        text = await run_in_threadpool(fetch_document, key)
        if text:
            banks.append(text)
    return {"jd": jd or ctx.job_title, "resume": ctx.resume_text, "question_bank": "\n\n".join(banks)}


def retrieve(docs: dict[str, str], *, max_requirements: int = 8) -> list[Requirement]:
    """Each JD chunk is a requirement; pull the resume lines and bank questions that best match it."""
    # Skip headings like "Requirements:" — they are not requirements themselves.
    jd_chunks = [c for c in chunk_text(docs["jd"], "jd", max_words=45, overlap=10) if len(c.tokens) >= 3]
    corpus: list[Chunk] = chunk_text(docs["resume"], "resume", max_words=60) + chunk_text(
        docs["question_bank"], "question_bank", max_words=50
    )
    index = BM25(corpus) if corpus else None
    reqs: list[Requirement] = []
    for jc in jd_chunks[: max_requirements * 2]:
        resume_hits = index.search(jc.text, k=2, source="resume") if index else []
        bank_hits = index.search(jc.text, k=2, source="question_bank") if index else []
        reqs.append(
            Requirement(
                text=jc.text,
                resume_evidence=[c.text for c, _ in resume_hits],
                bank_questions=[c.text for c, _ in bank_hits],
            )
        )
    # Prefer requirements the resume can actually speak to, then the rest.
    reqs.sort(key=lambda r: (not r.resume_evidence, not r.bank_questions))
    return reqs[:max_requirements]


async def build_plan(ctx: InterviewContext) -> list[PlannedQuestion]:
    docs = await gather_documents(ctx)
    reqs = retrieve(docs)
    if get_settings().fake:
        return _fake_plan(ctx, reqs)

    from .llm import structured  # local import keeps llm <-> planner acyclic

    blocks = []
    for i, r in enumerate(reqs, 1):
        ev = "\n".join(f"  - resume: {e}" for e in r.resume_evidence) or "  - resume: (no matching evidence)"
        bq = "\n".join(f"  - bank: {q}" for q in r.bank_questions)
        blocks.append(f"R{i}. {r.text}\n{ev}\n{bq}".rstrip())
    try:
        plan = await structured(
            system=(
                "You design structured first-round interviews. Using ONLY the retrieved material below, write an ordered "
                f"plan of exactly {ctx.question_count} main questions for this candidate and role.\n"
                "- Each question targets one job requirement and, where possible, a specific claim in the resume "
                "(quote it in evidence). Prefer the company's own bank questions when they fit, adapted to the candidate.\n"
                "- Mix: background/motivation first, then depth on core skills, one scenario or design question, one "
                "behavioural question" + (", and one problem-solving question talked through verbally" if ctx.coding_enabled else "") + ".\n"
                "- Questions must be answerable by speaking in 2-3 minutes. Give 1-2 probing follow-ups each.\n"
                "- source is 'resume' if anchored on a resume claim, 'question_bank' if adapted from the bank, else 'jd'.\n"
                "- Never ask about age, family, religion, nationality, health or other protected topics."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Role: {ctx.job_title}\nCandidate: {ctx.candidate_name}\n\n"
                        f"<job>\n{(ctx.job_description or docs['jd'])[:4000]}\n</job>\n\n"
                        f"<retrieved>\n{chr(10).join(blocks) or '(no retrieval — use job description)'}\n</retrieved>\n\n"
                        f"<resume_full>\n{docs['resume'][:6000] or '(no resume — ask candidate to describe background)'}\n</resume_full>"
                    ),
                }
            ],
            schema=_Plan,
            effort="low",
        )
        return [q.model_copy(update={"id": f"q{i + 1}"}) for i, q in enumerate(plan.questions)]
    except Exception as err:
        log.warning("LLM plan failed (%s), using retrieval fallback", err)
        return _fake_plan(ctx, reqs)


def _fake_plan(ctx: InterviewContext, reqs: list[Requirement]) -> list[PlannedQuestion]:
    """Deterministic plan from retrieval alone, so the flow is testable without Claude."""
    questions = [
        PlannedQuestion(
            id="q1",
            competency="Background & motivation",
            question=f"Walk me through your background and what draws you to the {ctx.job_title} role.",
            rationale="Opener",
            follow_ups=["What would you want to learn in your first six months?"],
            source="jd",
            evidence=[],
        )
    ]
    used: set[str] = set()
    for r in reqs:
        if len(questions) >= ctx.question_count:
            break
        topic = _topic(r.text)
        bank = next((b for b in r.bank_questions if "?" in b and b not in used), None)
        claim_src = next((e for e in r.resume_evidence if e not in used), None)
        if bank:
            q = bank.split("?")[0].strip() + "?"
            source, evidence = "question_bank", [bank]
            used.add(bank)
        elif claim_src:
            used.add(claim_src)
            claim = claim_src[:140]
            q = f'Your resume mentions "{claim}". How did you approach {topic} there, and what would you do differently today?'
            source, evidence = "resume", [claim_src]
        else:
            q = f"The role needs {topic}. Tell me about a time you worked on that and the trade-offs you made."
            source, evidence = "jd", []
        questions.append(
            PlannedQuestion(
                id=f"q{len(questions) + 1}",
                competency=topic.capitalize(),
                question=q,
                rationale=f"JD requirement: {r.text[:120]}",
                follow_ups=["Can you give a concrete example with numbers?", "What was the hardest trade-off?"],
                source=source,
                evidence=evidence,
            )
        )
    return questions


def _topic(text: str) -> str:
    """Short name for a JD line: the part before a colon, else its first few words."""
    head = text.split(":", 1)[0] if ":" in text[:40] else " ".join(text.split()[:5])
    return head.strip().rstrip(".,;").lower() or "this area"
