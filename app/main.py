"""hiremind-ai HTTP service. The backend calls these with a short-lived internal JWT."""

import logging

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from . import llm
from .planner import build_plan
from .config import get_settings
from .interviewer import InterviewContext, Turn
from .resume import parse_resume

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = FastAPI(title="hiremind-ai", version="0.1.0")


def require_backend(authorization: str = Header(default="")) -> dict:
    token = authorization.removeprefix("Bearer ").strip()
    try:
        claims = jwt.decode(token, get_settings().jwt_internal_secret, algorithms=["HS256"], audience="ai-service")
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid internal token: {e}") from e
    return claims


@app.exception_handler(llm.LLMRefusal)
async def on_refusal(_, exc: llm.LLMRefusal):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=422, content={"message": f"AI declined: {exc}"})


@app.get("/health")
def health():
    s = get_settings()
    return {
        "status": "ok",
        "service": "hiremind-ai",
        "fakeMode": s.fake,
        "llmProvider": s.llm_provider,
        "model": s.llm_model,
        # The backend only offers the video call when the agent has everything it needs.
        "voiceReady": s.voice_ready,
    }


# ---------- Contract used by hiremind-backend's AiClientService ----------


class ParseRequest(BaseModel):
    companyId: str
    resumeId: str
    storageKey: str
    jobId: str | None = None


@app.post("/v1/resume/parse", dependencies=[Depends(require_backend)])
async def resume_parse(req: ParseRequest):
    return {"text": await run_in_threadpool(parse_resume, req.storageKey)}


class FitRequest(BaseModel):
    companyId: str
    candidateJobId: str = ""
    jobDescription: str
    resumeText: str


@app.post("/v1/fit/score", dependencies=[Depends(require_backend)])
async def fit_score(req: FitRequest):
    result = await llm.score_fit(req.jobDescription, req.resumeText)
    return result.model_dump()


class ChatRequest(BaseModel):
    context: InterviewContext
    history: list[Turn]


@app.post("/v1/interview/chat", dependencies=[Depends(require_backend)])
async def interview_chat(req: ChatRequest):
    turn = await llm.next_turn(req.context, req.history)
    return turn.model_dump()


class PlanRequest(BaseModel):
    context: InterviewContext


@app.post("/v1/interview/plan", dependencies=[Depends(require_backend)])
async def interview_plan(req: PlanRequest):
    questions = await build_plan(req.context)
    return {"questions": [q.model_dump() for q in questions]}


class EvaluateRequest(BaseModel):
    context: InterviewContext
    transcript: list[Turn]
    # Proctoring event types seen during the interview (e.g. AI_SUSPECTED_READING, PASTE), for the authenticity check.
    integrity_events: list[str] = []


@app.post("/v1/interview/evaluate", dependencies=[Depends(require_backend)])
async def interview_evaluate(req: EvaluateRequest):
    result = await llm.evaluate(req.context, req.transcript, req.integrity_events)
    out = result.model_dump()
    # Backend's report UI reads details.dimensions as {name: score}.
    out["dimensions"] = {d.name: d.score for d in result.dimensions}
    out["evidence"] = {d.name: d.evidence for d in result.dimensions}
    return out
