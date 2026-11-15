"""LiveKit voice interviewer.

The backend creates a room per interview and dispatches this agent into it with
{"interviewId": ...} as job metadata. The agent loads the resume + job context,
runs the conversation (STT -> Gemini, Groq or Claude -> TTS, with Cartesia, Groq
or OpenAI speech), streams every turn back to the backend, and marks the interview complete
when it wraps up. Without an LLM key it speaks the RAG question plan as a script.

Run:  uv run python -m agent.worker dev
"""

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from livekit.agents import (  # noqa: E402
    Agent,
    room_io,
    stt as agents_stt,
    tts as agents_tts,
    AgentServer,
    AgentSession,
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    JobContext,
    RunContext,
    ModelSettings,
    TurnHandlingOptions,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    cli,
    function_tool,
    llm,
)
from livekit import rtc  # noqa: E402
from livekit.plugins import anthropic, cartesia, google, groq, openai, silero  # noqa: E402

# Imported up front: the turn model runs in LiveKit's inference process, which is set up at worker start.
from livekit.plugins.turn_detector.multilingual import MultilingualModel  # noqa: E402

from app.authenticity import analyze_answer  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.guardrails import classify_candidate_turn, scrub_self_disclosure  # noqa: E402
from app.interviewer import InterviewContext, Turn, build_instructions, fake_reply  # noqa: E402

from .answer_watch import AnswerTimer, JobKnowledge  # noqa: E402
from .backend import BackendClient  # noqa: E402
from .dtln import SAMPLE_RATE as DTLN_RATE, DTLNFilter  # noqa: E402

log = logging.getLogger("hiremind.agent")
settings = get_settings()
# Local prod mode otherwise pre-warms ~4 idle processes (~15s before accepting jobs).
server = AgentServer(num_idle_processes=1, initialize_process_timeout=45.0)

SAY_WARNINGS = {
    "CAMERA_OFF": "Sorry to interrupt, your camera seems to be off. Please turn it back on, it needs to stay on for the interview.",
    "NO_FACE": "Sorry to interrupt, I can't see your face. Please stay in front of the camera.",
    "MULTIPLE_FACES": "Just a reminder, this interview needs to be taken alone. It looks like someone else is in view.",
    "TAB_SWITCH": "Please stay on the interview tab for the rest of the interview.",
    "MIC_MUTED": "Your microphone seems to be muted. Please unmute so I can hear you.",
}

SPOKEN_WARNINGS = {
    "CAMERA_OFF": "ask the candidate to turn their camera back on — it must stay on for the whole interview.",
    "NO_FACE": "ask the candidate to stay in front of the camera so their face is visible.",
    "MULTIPLE_FACES": "remind the candidate that the interview must be taken alone — another person seems to be in view.",
    "TAB_SWITCH": "remind the candidate to stay on the interview tab.",
    "MIC_MUTED": "let the candidate know their microphone is muted.",
}


class VoiceLLM(anthropic.LLM):
    """Claude with low effort so spoken replies start quickly."""

    def chat(self, **kwargs: Any):
        extra = dict(kwargs.pop("extra_kwargs", None) or {})
        body = dict(extra.get("extra_body") or {})
        body.setdefault("output_config", {"effort": "low"})
        extra["extra_body"] = body
        return super().chat(extra_kwargs=extra, **kwargs)


class ScriptedLLM(llm.LLM):
    """Stand-in for Claude when no Anthropic key is set: speaks the question plan in order."""

    def __init__(self, ctx: InterviewContext, on_end) -> None:
        super().__init__()
        self._ctx = ctx
        self._on_end = on_end
        self._said: list[Turn] = []

    def chat(self, *, chat_ctx: llm.ChatContext, tools=None, conn_options=None, **_: Any) -> "llm.LLMStream":
        text, done = fake_reply(self._ctx, self._said)
        self._said.append(Turn(role="interviewer", text=text))
        if done:
            self._on_end("completed")
        return _ScriptedStream(self, text, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class _ScriptedStream(llm.LLMStream):
    def __init__(self, owner: ScriptedLLM, text: str, *, chat_ctx, tools, conn_options) -> None:
        from livekit.agents import DEFAULT_API_CONNECT_OPTIONS

        super().__init__(owner, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options or DEFAULT_API_CONNECT_OPTIONS)
        self._text = text

    async def _run(self) -> None:
        self._event_ch.send_nowait(
            llm.ChatChunk(id=uuid.uuid4().hex, delta=llm.ChoiceDelta(role="assistant", content=self._text))
        )


def _llm(provider: str):
    if provider == "groq":
        return groq.LLM(model=settings.groq_voice_model, api_key=settings.groq_api_key)
    if provider == "gemini":
        # gemini-3.8-flash rejects MINIMAL; omit thinking for fast, reliable voice turns.
        return google.LLM(model=settings.gemini_model, api_key=settings.gemini_api_key)
    return VoiceLLM(model=settings.claude_model, max_tokens=2048)


def _stt(provider: str):
    if provider == "groq":
        return groq.STT(model="whisper-large-v3-turbo", language="en", api_key=settings.groq_api_key)
    if provider == "openai":
        return openai.STT(model=settings.stt_model, language="en")
    model = (settings.cartesia_stt_model or "ink-2").strip()
    kwargs: dict[str, Any] = {
        "model": model,
        "language": "en",
        "api_key": settings.cartesia_api_key,
    }
    if not model.startswith("ink-whisper"):
        kwargs["turn_end_timeout_ms"] = settings.cartesia_turn_end_timeout_ms
    return cartesia.STT(**kwargs)


def _tts(provider: str):
    if provider == "openai":
        return openai.TTS(
            model=settings.tts_model,
            voice=settings.tts_voice,
            instructions="Speak like a friendly, professional interviewer: calm, clear, moderate pace.",
        )
    return cartesia.TTS(
        model=settings.cartesia_tts_model, voice=settings.cartesia_voice, language="en", api_key=settings.cartesia_api_key
    )


# Fallbacks follow voice-bot-poc's service-fallback idea: if a provider errors mid-call,
# the next one with a key takes over. LiveKit's FallbackAdapters switch per request.


def make_llm():
    chain = settings.llm_chain
    if len(chain) == 1:
        return _llm(chain[0])
    return llm.FallbackAdapter([_llm(p) for p in chain], attempt_timeout=10.0)  # Gemini rejects deadlines under 10s


def stt_uses_own_turns() -> bool:
    """Cartesia ink-2 detects end of turn itself; everything else needs VAD or the turn model."""
    return settings.stt_provider == "cartesia" and not (settings.cartesia_stt_model or "ink-2").startswith("ink-whisper")


def make_stt(vad):
    if stt_uses_own_turns():
        # STT-driven turn detection needs that one streaming model; no fallback here.
        return _stt("cartesia")
    chain = settings.chain(settings.stt_provider, settings.stt_fallbacks, settings._speech_key)
    if len(chain) <= 1:
        return _stt(settings.stt_provider)
    return agents_stt.FallbackAdapter([_stt(p) for p in chain], vad=vad)


def make_tts():
    chain = settings.chain(settings.tts_provider, settings.tts_fallbacks, settings._speech_key)
    if len(chain) <= 1:
        return _tts(settings.tts_provider)
    return agents_tts.FallbackAdapter([_tts(p) for p in chain])


def make_vad():
    return silero.VAD.load(
        min_speech_duration=settings.vad_start_secs,
        min_silence_duration=settings.vad_stop_secs,
        activation_threshold=settings.vad_confidence,
        sample_rate=16000,
    )


def turn_handling_for_stt() -> TurnHandlingOptions:
    """Preemptive LLM+TTS and VAD interruption everywhere; end of turn depends on the STT."""
    if stt_uses_own_turns():
        # Cartesia ink-2 already waits turn_end_timeout_ms, so endpointing stays tight.
        turn: TurnHandlingOptions = {"turn_detection": "stt", "endpointing": {"min_delay": 0.12, "max_delay": 0.55}}
    else:
        if settings.turn_detector:
            turn = {"turn_detection": MultilingualModel()}
        else:
            turn = {"turn_detection": "vad"}
        turn["endpointing"] = {"min_delay": settings.endpointing_min_delay, "max_delay": settings.endpointing_max_delay}
    turn["preemptive_generation"] = {"enabled": True, "preemptive_tts": True}
    turn["interruption"] = {"mode": "vad"}
    return turn


def room_options() -> room_io.RoomOptions:
    # Keep the session alive when the candidate drops; the entrypoint decides when to give up.
    if not settings.noise_filter:
        return room_io.RoomOptions(close_on_disconnect=False)
    # DTLN runs at 16 kHz, so take the candidate's mic at that rate (VAD and STT use 16 kHz anyway).
    return room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(sample_rate=DTLN_RATE, noise_cancellation=DTLNFilter()),
        close_on_disconnect=False,
    )


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

PROBE_NOTE = (
    "Interviewer note: this answer sounds read out (polished written phrasing, list structure or a long pause "
    "before a fluent answer). Do not accuse the candidate. Ask one short follow-up that needs a personal, "
    "specific detail in their own words, for example what they personally did or what went wrong."
)


class Interviewer(Agent):
    def __init__(
        self, ctx: InterviewContext, backend: BackendClient, on_end, timer: AnswerTimer, knowledge: JobKnowledge
    ) -> None:
        super().__init__(instructions=build_instructions(ctx, voice=True))
        self._ctx = ctx
        self._backend = backend
        self._on_end = on_end
        self._started = time.monotonic()
        self._timer = timer
        self._knowledge = knowledge

    def _event(self, event_type: str, detail: str) -> None:
        asyncio.create_task(self._backend.proctoring_event(self._ctx.interview_id, event_type, "warning", detail[:480]))

    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        """Guardrails and the authenticity check run on every candidate turn, before the reply is generated."""
        text = new_message.text_content or ""
        notes = []
        if hit := classify_candidate_turn(text):
            log.info("interview %s: guardrail %s: %s", self._ctx.interview_id, hit.category, text[:160])
            for category in hit.category.split(","):
                self._event(f"GUARD_{category.upper()}", f'Candidate said: "{text[:300]}"')
            notes.append(hit.note)
        # Judge the whole answer so far, not just this fragment, and flag each answer once.
        answer = self._timer.add_text(text)
        signals = analyze_answer(answer, latency_s=self._timer.latency, speech_s=self._timer.speech or None)
        if signals.suspicious and not self._timer.flagged:
            self._timer.flagged = True
            log.info("interview %s: answer looks read out (%.2f): %s", self._ctx.interview_id, signals.score, signals.signals)
            self._event("AI_SUSPECTED_READING", f"score {signals.score}: {'; '.join(signals.signals)}")
        # An answer often arrives in several turns; keep asking for the probe until Tovia actually replies.
        if self._timer.flagged:
            notes.append(PROBE_NOTE)
        if notes:
            turn_ctx.add_message(role="system", content=" ".join(notes))

    async def llm_node(self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings: ModelSettings):
        """Passes the model's reply through sentence by sentence, dropping any self-disclosure of model or vendor."""
        buf = ""
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            if isinstance(chunk, str):
                text = chunk
            elif isinstance(chunk, llm.ChatChunk) and chunk.delta and chunk.delta.content:
                text = chunk.delta.content
                if chunk.delta.tool_calls:
                    yield chunk.model_copy(update={"delta": chunk.delta.model_copy(update={"content": None})})
            else:
                yield chunk
                continue
            buf += text
            *done, buf = _SENTENCE_END.split(buf)
            for sentence in done:
                if out := scrub_self_disclosure(sentence, self._ctx.company_name):
                    yield out + " "
        if buf.strip() and (out := scrub_self_disclosure(buf, self._ctx.company_name)):
            yield out

    @function_tool
    async def search_job_knowledge(self, context: RunContext, query: str) -> str:
        """Search the documents the company uploaded for this role (question banks, role guides, technical docs).

        Use it when choosing a topic or a follow-up so questions match what the company cares about.
        Never read results aloud or tell the candidate expected answers.

        Args:
            query: The skill, topic or requirement to look up, e.g. "PostgreSQL indexing" or "on-call incidents".
        """
        hits = self._knowledge.search(query)
        log.info("interview %s: knowledge search %r -> %d hits", self._ctx.interview_id, query, len(hits))
        return "\n".join(hits) if hits else "No matching material in the uploaded documents."

    async def on_enter(self) -> None:
        first = (self._ctx.candidate_name or "there").split()[0]
        line = (
            f"Hello {first}, I'm Tovia from {self._ctx.company_name}. "
            "Thanks for joining today — how are you doing?"
        )
        log.info("interview %s: greeting candidate", self._ctx.interview_id)
        await self.session.say(line, allow_interruptions=True)

    @function_tool
    async def flag_integrity_concern(self, context: RunContext, concern: str, detail: str) -> str:
        """Record an integrity concern noticed in the conversation.

        Use when another voice seems to prompt the candidate, the candidate talks to someone else,
        or answers sound read aloud. Mention it politely to the candidate as well.

        Args:
            concern: One of "other_voice", "talking_to_someone", "reading_answers", "other".
            detail: One short sentence describing what you noticed.
        """
        log.info("interview %s: integrity flag %s: %s", self._ctx.interview_id, concern, detail)
        await self._backend.proctoring_event(
            self._ctx.interview_id, f"AI_FLAG_{concern.upper()}", "warning", detail
        )
        return "Noted. Continue the interview."

    @function_tool
    async def end_interview(self, context: RunContext, reason: str) -> str:
        """Call this once the interview is finished and you have said goodbye.

        Args:
            reason: Short reason, e.g. "completed" or "candidate asked to stop".
        """
        # Smaller models sometimes call this after the first answer; hold them to the plan
        # unless the candidate asked to stop or time is nearly up.
        asked = sum(1 for item in self.session.history.items if getattr(item, "role", None) == "assistant")
        needed = (len(self._ctx.question_plan) or self._ctx.question_count) + 1  # greeting + each question
        late = time.monotonic() - self._started > self._ctx.duration_minutes * 60 * 0.8
        by_candidate = any(w in reason.lower() for w in ("candidate", "stop", "time"))
        if asked < needed and not late and not by_candidate:
            log.info("interview %s: refused early end_interview (%s) after %d/%d turns", self._ctx.interview_id, reason, asked, needed)
            return (
                f"Do not end the interview yet: only {asked} of about {needed} interviewer turns are done. "
                "Acknowledge the answer briefly and ask the next question from your plan."
            )
        log.info("interview %s: end_interview (%s) after %d turns", self._ctx.interview_id, reason, asked)
        self._on_end(reason)
        return "The interview is over. Do not say anything else."


@server.rtc_session(agent_name=settings.agent_name)
async def entrypoint(job: JobContext) -> None:
    meta = json.loads(job.job.metadata or "{}")
    interview_id: str = meta["interviewId"]
    backend = BackendClient()
    ctx = await backend.context(interview_id)
    log.info("starting interview %s for %s (%s)", interview_id, ctx.candidate_name, ctx.job_title)
    if not (ctx.resume_text or "").strip():
        log.warning("interview %s: no resume text in context — use JD and question plan only", interview_id)
    if not (ctx.job_description or "").strip() and not ctx.jd_document_key:
        log.warning("interview %s: no JD text — rely on job title and plan", interview_id)
    if ctx.question_plan:
        log.info("interview %s: question plan has %d items", interview_id, len(ctx.question_plan))

    ending: dict[str, str] = {}
    completed = asyncio.Event()
    timer = AnswerTimer()
    try:
        knowledge = await JobKnowledge.load(ctx)
    except Exception:
        log.exception("interview %s: could not load job documents", interview_id)
        knowledge = JobKnowledge(None)

    async def finish(reason: str) -> None:
        if completed.is_set():
            return
        completed.set()
        try:
            await backend.complete(interview_id, reason)
        except Exception:
            log.exception("failed to mark interview %s complete", interview_id)

    end = lambda reason: ending.setdefault("reason", reason)  # noqa: E731
    scripted = settings.fake
    turn_handling = turn_handling_for_stt()
    vad = make_vad()
    session = AgentSession(
        vad=vad,
        stt=make_stt(vad),
        llm=ScriptedLLM(ctx, on_end=end) if scripted else make_llm(),
        tts=make_tts(),
        turn_handling=turn_handling,
        aec_warmup_duration=0,
    )
    td = turn_handling.get("turn_detection")
    log.info(
        "voice pipeline: stt=%s (turn=%s) llm=%s tts=%s noise_filter=%s",
        settings.chain(settings.stt_provider, settings.stt_fallbacks, settings._speech_key),
        td if isinstance(td, str) else type(td).__name__,
        "scripted" if scripted else settings.llm_chain,
        settings.chain(settings.tts_provider, settings.tts_fallbacks, settings._speech_key),
        settings.noise_filter,
    )

    def interject(instructions: str, line: str) -> None:
        # The scripted LLM can't follow instructions, so it speaks a fixed line instead.
        if scripted:
            session.say(line)
        else:
            session.generate_reply(instructions=instructions)

    @session.on("user_input_transcribed")
    def _on_user_stt(ev: UserInputTranscribedEvent) -> None:
        if ev.is_final and (ev.transcript or "").strip():
            log.info("interview %s: candidate said: %s", interview_id, ev.transcript.strip()[:200])

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        text = getattr(item, "text_content", None)
        role = getattr(item, "role", None)
        if not text or role not in ("assistant", "user"):
            return
        speaker = "interviewer" if role == "assistant" else "candidate"
        asyncio.create_task(backend.add_turn(interview_id, speaker, text))

    @session.on("user_state_changed")
    def _on_user_state(ev: UserStateChangedEvent) -> None:
        if ev.new_state == "speaking":
            timer.user_started()
        elif ev.old_state == "speaking":
            timer.user_stopped()

    @session.on("agent_state_changed")
    def _on_state(ev: AgentStateChangedEvent) -> None:
        if ev.new_state == "speaking":
            timer.agent_started()
        elif ev.old_state == "speaking":
            timer.agent_finished()
        # Once the goodbye has finished playing, close everything down.
        if ending and ev.new_state == "listening":
            asyncio.create_task(_wrap_up())

    async def _wrap_up() -> None:
        await finish(ending.get("reason", "completed"))
        await asyncio.sleep(1)
        job.delete_room()

    # Network drops: wait for the candidate to rejoin before completing the interview.
    grace: dict[str, asyncio.Task] = {}

    async def _give_up() -> None:
        await asyncio.sleep(settings.rejoin_grace_secs)
        log.info("interview %s: candidate did not rejoin within %ss", interview_id, settings.rejoin_grace_secs)
        end("disconnected")
        await _wrap_up()

    @job.room.on("participant_disconnected")
    def _on_left(p: rtc.RemoteParticipant) -> None:
        if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT or ending or "task" in grace:
            return
        log.info("interview %s: candidate left, waiting %ss for rejoin", interview_id, settings.rejoin_grace_secs)
        grace["task"] = asyncio.create_task(_give_up())

    @job.room.on("participant_connected")
    def _on_joined(p: rtc.RemoteParticipant) -> None:
        task = grace.pop("task", None)
        if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT or not task:
            return
        task.cancel()
        log.info("interview %s: candidate rejoined", interview_id)
        interject(
            "The candidate's connection dropped and they just rejoined. Welcome them back in one sentence "
            "and repeat your last question.",
            "Welcome back! Let's pick up where we left off.",
        )

    async def _time_guard() -> None:
        await asyncio.sleep(ctx.duration_minutes * 60)
        if ending:
            return
        if scripted:
            await session.say(
                "We're out of time. Thank you so much for your answers, the team will be in touch soon. Goodbye!",
                allow_interruptions=False,
            )
            end("time_up")
            await _wrap_up()
        else:
            session.generate_reply(
                instructions="Time is up. Thank the candidate, close the interview politely, then call end_interview."
            )

    async def _on_shutdown() -> None:
        await finish("disconnected")
        await backend.aclose()

    job.add_shutdown_callback(_on_shutdown)

    # The candidate's browser streams proctoring events on this topic; react to serious ones out loud.
    last_spoken: dict[str, float] = {}

    @job.room.on("data_received")
    def _on_data(packet: rtc.DataPacket) -> None:
        if packet.topic == "control" and not ending:
            # The candidate pressed "End interview"; complete now instead of waiting out the rejoin grace.
            end("candidate_ended")
            asyncio.create_task(_wrap_up())
            return
        if packet.topic != "proctoring" or ending:
            return
        try:
            event = json.loads(packet.data)
        except ValueError:
            return
        kind = str(event.get("eventType", ""))
        prompt = SPOKEN_WARNINGS.get(kind)
        now = time.monotonic()
        if not prompt or now - last_spoken.get(kind, 0) < 45:
            return
        last_spoken[kind] = now
        interject(f"Briefly and kindly interrupt: {prompt} Then continue where you left off.", SAY_WARNINGS[kind])

    await session.start(
        agent=Interviewer(ctx, backend, on_end=end, timer=timer, knowledge=knowledge),
        room=job.room,
        room_options=room_options(),
    )
    asyncio.create_task(_time_guard())


if __name__ == "__main__":
    cli.run_app(server)
