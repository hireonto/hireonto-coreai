"""LLM calls (Gemini, Groq or Claude) that return validated JSON (Pydantic) — fit scoring, chat turns, evaluation."""

import logging
import random
from typing import Literal, TypeVar

import anthropic
from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from .authenticity import analyze_answer, summarize
from .config import get_settings
from .guardrails import classify_candidate_turn, scrub_self_disclosure
from .interviewer import InterviewContext, Turn, build_instructions, fake_reply

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# On a policy decline the API retries on a fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMRefusal(RuntimeError):
    pass


_client: anthropic.AsyncAnthropic | None = None
_gemini: genai.Client | None = None
_groq: AsyncOpenAI | None = None


def client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=get_settings().anthropic_api_key)
    return _client


def gemini() -> genai.Client:
    global _gemini
    if _gemini is None:
        _gemini = genai.Client(api_key=get_settings().gemini_api_key, http_options=genai_types.HttpOptions(timeout=120_000))
    return _gemini


def groq() -> AsyncOpenAI:
    global _groq
    if _groq is None:
        _groq = AsyncOpenAI(api_key=get_settings().groq_api_key, base_url="https://api.groq.com/openai/v1")
    return _groq


async def structured(
    *,
    system: str,
    messages: list[dict],
    schema: type[T],
    effort: Literal["low", "medium", "high"] = "medium",
    max_tokens: int = 16000,
) -> T:
    """Tries the primary provider, then each configured fallback (e.g. Gemini 503 -> Groq)."""
    providers = get_settings().llm_chain
    for i, provider in enumerate(providers):
        call = {"groq": _structured_groq, "gemini": _structured_gemini}.get(provider, _structured_claude)
        try:
            return await call(system=system, messages=messages, schema=schema, effort=effort, max_tokens=max_tokens)
        except LLMRefusal:
            raise
        except Exception as exc:
            if i == len(providers) - 1:
                raise
            log.warning("%s failed (%s: %s); falling back to %s", provider, type(exc).__name__, str(exc)[:200], providers[i + 1])
    raise RuntimeError("No LLM provider configured")


_GEMINI_THINKING = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}


def _gemini_thinking(effort: str) -> genai_types.ThinkingConfig | None:
    level = _GEMINI_THINKING.get(effort.lower())
    return genai_types.ThinkingConfig(thinking_level=level) if level else None


async def _structured_gemini(*, system: str, messages: list[dict], schema: type[T], effort: str, max_tokens: int) -> T:
    contents = [
        genai_types.Content(role="model" if m["role"] == "assistant" else "user", parts=[genai_types.Part(text=m["content"])])
        for m in messages
    ]
    base = dict(
        system_instruction=system,
        response_mime_type="application/json",
        response_json_schema=_inline_refs(schema.model_json_schema()),
        max_output_tokens=max_tokens,
    )
    thinking = _gemini_thinking(effort)
    config = genai_types.GenerateContentConfig(**base, thinking_config=thinking) if thinking else genai_types.GenerateContentConfig(**base)
    try:
        response = await gemini().aio.models.generate_content(
            model=get_settings().gemini_model,
            contents=contents,
            config=config,
        )
    except Exception as err:
        if thinking and "thinking" in str(err).lower():
            log.warning("Gemini thinking_config rejected, retrying without: %s", err)
            response = await gemini().aio.models.generate_content(
                model=get_settings().gemini_model,
                contents=contents,
                config=genai_types.GenerateContentConfig(**base),
            )
        else:
            raise
    candidate = response.candidates[0] if response.candidates else None
    if not response.text:
        reason = getattr(candidate, "finish_reason", None) or getattr(response.prompt_feedback, "block_reason", None)
        if str(reason).endswith(("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST")):
            raise LLMRefusal(f"Model declined the request ({reason})")
        raise RuntimeError(f"No structured output (finish_reason={reason})")
    return schema.model_validate_json(response.text)


async def _structured_groq(*, system: str, messages: list[dict], schema: type[T], effort: str, max_tokens: int) -> T:
    s = get_settings()
    response = await groq().chat.completions.create(
        model=s.groq_model,
        messages=[{"role": "system", "content": system}, *messages],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema.__name__.lstrip("_"), "schema": _inline_refs(schema.model_json_schema())},
        },
        max_completion_tokens=max_tokens,
        # reasoning_effort is only accepted by the reasoning models (gpt-oss, qwen3).
        **({"reasoning_effort": effort} if "gpt-oss" in s.groq_model else {}),
    )
    choice = response.choices[0]
    if not choice.message.content:
        if getattr(choice.message, "refusal", None):
            raise LLMRefusal(choice.message.refusal)
        raise RuntimeError(f"No structured output (finish_reason={choice.finish_reason})")
    return schema.model_validate_json(choice.message.content)


def _inline_refs(schema: dict) -> dict:
    """Expand $ref/$defs so nested models work with Gemini's JSON-schema subset."""
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].rsplit("/", 1)[-1]])
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


async def _structured_claude(*, system: str, messages: list[dict], schema: type[T], effort: str, max_tokens: int) -> T:
    s = get_settings()
    response = await client().beta.messages.parse(
        model=s.claude_model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        output_format=schema,
        output_config={"effort": effort},
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise LLMRefusal(getattr(response.stop_details, "explanation", None) or "Model declined the request")
    if response.parsed_output is None:
        raise RuntimeError(f"No structured output (stop_reason={response.stop_reason})")
    return response.parsed_output


# ---------- Fit scoring ----------


class FitScore(BaseModel):
    score: float = Field(ge=0, le=1, description="0-1 fit of the resume to the job")
    explanation: str = Field(description="2-3 sentences: strongest matches and the main gaps")


async def score_fit(job_description: str, resume_text: str) -> FitScore:
    if get_settings().fake:
        return FitScore(
            score=round(0.6 + random.random() * 0.35, 2),
            explanation="FAKE_MODE: Relevant backend experience; depth in the listed stack should be verified in interview.",
        )
    return await structured(
        system=(
            "You screen resumes for recruiters. Score how well the resume fits the job from 0 to 1 "
            "(0.9+ exceptional, 0.7 solid, 0.5 partial, <0.4 poor). Judge evidence of skills and impact, "
            "not keyword counts. Ignore name, gender, age, nationality, photos and any other protected attributes."
        ),
        messages=[
            {
                "role": "user",
                "content": f"<job>\n{job_description}\n</job>\n\n<resume>\n{resume_text}\n</resume>",
            }
        ],
        schema=FitScore,
        effort="low",
    )


# ---------- Text-chat interview turn ----------


class ChatTurn(BaseModel):
    reply: str = Field(description="The interviewer's next message to the candidate")
    done: bool = Field(description="True only when the interview is finished and this is the closing message")


async def next_turn(ctx: InterviewContext, history: list[Turn]) -> ChatTurn:
    if get_settings().fake:
        reply, done = fake_reply(ctx, history)
        return ChatTurn(reply=reply, done=done)
    messages: list[dict] = [{"role": "user", "content": "(The candidate has joined the chat. Begin the interview.)"}]
    for t in history:
        role = "assistant" if t.role == "interviewer" else "user"
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n\n" + t.text
        else:
            messages.append({"role": role, "content": t.text})
    if messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": "(no response)"})
    # Guardrails and authenticity checks steer the next reply through a note on the latest candidate turn.
    last = history[-1].text if history and history[-1].role == "candidate" else ""
    notes = []
    if hit := classify_candidate_turn(last):
        notes.append(hit.note)
    if analyze_answer(last).suspicious:
        notes.append(
            "This answer reads like polished written text. Without accusing them, ask one follow-up that needs a "
            "personal, specific detail in their own words."
        )
    if notes:
        messages[-1]["content"] += "\n\n<interviewer_note>" + " ".join(notes) + "</interviewer_note>"
    turn = await structured(
        system=build_instructions(ctx, voice=False),
        messages=messages,
        schema=ChatTurn,
        effort="low",
        max_tokens=4000,
    )
    turn.reply = scrub_self_disclosure(turn.reply, ctx.company_name)
    return turn


# ---------- Evaluation ----------


class Dimension(BaseModel):
    name: Literal["technical_depth", "problem_solving", "communication", "role_fit", "ownership"]
    score: float = Field(ge=0, le=1)
    evidence: str = Field(description="One sentence citing what the candidate actually said")


class Authenticity(BaseModel):
    human_likelihood: float = Field(ge=0, le=1, description="1 = clearly the candidate's own spontaneous words")
    verdict: Literal["likely_human", "uncertain", "likely_ai_assisted"]
    signals: list[str] = Field(description="Concrete reasons, citing answers")
    flagged_answers: list[str] = Field(description="Short quotes of answers that look read from an AI tool")


class Evaluation(BaseModel):
    overallScore: float = Field(ge=0, le=1)
    recommendation: Literal["strong_hire", "hire", "hold", "no_hire"]
    summary: str = Field(description="3-5 sentence summary for the hiring manager")
    strengths: list[str]
    concerns: list[str]
    dimensions: list[Dimension]
    authenticity: Authenticity


def _heuristic_authenticity(transcript: list[Turn], integrity_events: list[str]) -> Authenticity:
    per_answer = [analyze_answer(t.text) for t in transcript if t.role == "candidate"]
    human, signals = summarize(per_answer)
    live = [e for e in integrity_events if "AI_SUSPECTED" in e or "PASTE" in e]
    if live:
        human = round(max(0.0, human - 0.15 * len(live)), 2)
        signals.append(f"{len(live)} live flag(s) during the interview")
    verdict = "likely_human" if human >= 0.7 else "likely_ai_assisted" if human < 0.4 else "uncertain"
    flagged = [t.text[:140] for t, a in zip([t for t in transcript if t.role == "candidate"], per_answer) if a.suspicious]
    return Authenticity(human_likelihood=human, verdict=verdict, signals=signals, flagged_answers=flagged)


async def evaluate(ctx: InterviewContext, transcript: list[Turn], integrity_events: list[str] | None = None) -> Evaluation:
    events = integrity_events or []
    if get_settings().fake or not transcript:
        return Evaluation(
            authenticity=_heuristic_authenticity(transcript, events),
            overallScore=0.74,
            recommendation="hire",
            summary="FAKE_MODE: Clear communicator with relevant project experience; system design depth was moderate.",
            strengths=["Explained past projects concretely", "Structured answers"],
            concerns=["Scaling answer lacked specifics"],
            dimensions=[
                Dimension(name="technical_depth", score=0.72, evidence="FAKE_MODE"),
                Dimension(name="problem_solving", score=0.7, evidence="FAKE_MODE"),
                Dimension(name="communication", score=0.82, evidence="FAKE_MODE"),
                Dimension(name="role_fit", score=0.75, evidence="FAKE_MODE"),
                Dimension(name="ownership", score=0.7, evidence="FAKE_MODE"),
            ],
        )
    lines = "\n".join(f"{'INTERVIEWER' if t.role == 'interviewer' else 'CANDIDATE'}: {t.text}" for t in transcript)
    heuristic = _heuristic_authenticity(transcript, events)
    return await structured(
        system=(
            "You are a fair, rigorous hiring panelist. Evaluate the candidate ONLY on what they said in the transcript, "
            "against the job. Speech-to-text errors are possible; do not penalise transcription artefacts. "
            "Ignore protected attributes. Scores 0-1. 'strong_hire' needs clear evidence across most dimensions.\n"
            "Also judge authenticity: did the candidate answer in their own words, or read answers produced by an AI "
            "tool? Read-out answers tend to use polished written phrasing, 'firstly/secondly' structure, textbook "
            "definitions and generic claims with no personal specifics, and often fall apart under follow-up questions. "
            "Spoken, spontaneous answers have hesitations, self-corrections and concrete personal detail. Weigh the "
            "automatic signals provided, but they are hints, not proof. Be conservative: only say 'likely_ai_assisted' "
            "with clear evidence, and mention it in concerns."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Role: {ctx.job_title}\n<job>\n{ctx.job_description}\n</job>\n\n"
                    f"<resume>\n{ctx.resume_text}\n</resume>\n\n<transcript>\n{lines}\n</transcript>\n\n"
                    f"<automatic_authenticity_signals>\nhuman_likelihood={heuristic.human_likelihood}; "
                    f"signals={heuristic.signals}; live_integrity_events={events}\n</automatic_authenticity_signals>"
                ),
            }
        ],
        schema=Evaluation,
        effort="high",
    )
