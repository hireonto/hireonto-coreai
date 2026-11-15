from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Shared with hiremind-backend (JWT_INTERNAL_SECRET) for service-to-service auth.
    jwt_internal_secret: str = "dev-internal-ai-secret-change-me"
    backend_url: str = "http://localhost:3001"

    # LLM: "gemini", "groq" or "claude". With no key for it the service runs in fake mode so the stack works offline.
    llm_provider: str = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.8-flash"
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"  # plan, scoring, evaluation, chat
    groq_voice_model: str = "openai/gpt-oss-20b"  # voice: fastest model this Groq key can use (low reasoning)
    anthropic_api_key: str | None = None
    claude_model: str = "claude-opus-5-5"
    ai_fake_mode: bool = False

    # Voice pipeline, used only by the agent worker. STT: "cartesia", "groq" or "openai"; TTS: "cartesia" or "openai".
    stt_provider: str = "groq"  # groq whisper + vad for low latency; cartesia ink-2 + stt turn also works
    tts_provider: str = "cartesia"
    cartesia_api_key: str | None = None
    cartesia_stt_model: str = "ink-2"  # ink-whisper lacks turn detection; use ink-2 + turn_detection=stt
    cartesia_turn_end_timeout_ms: int = 720
    cartesia_tts_model: str = "sonic-3"
    cartesia_voice: str = "f786b574-daa5-4673-aa0c-cbe3e8534c02"  # Katie, friendly
    openai_api_key: str | None = None
    stt_model: str = "gpt-4o-transcribe"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"
    # Fallbacks, tried in order after the primary when it errors (only providers with a key are used).
    llm_fallbacks: str = "groq,gemini,claude"
    stt_fallbacks: str = "cartesia,groq,openai"
    tts_fallbacks: str = "cartesia,openai"

    # Audio + turn taking (values follow voice-bot-poc; interview answers get a longer max wait).
    noise_filter: bool = True  # DTLN on the candidate's mic
    vad_start_secs: float = 0.2
    vad_stop_secs: float = 0.55
    vad_confidence: float = 0.6
    # For STTs without their own end-of-turn (groq, openai, ink-whisper): LiveKit's multilingual
    # end-of-turn model, like voice-bot-poc's smart turn. Waits up to max_delay while a thought is unfinished.
    turn_detector: bool = True
    endpointing_min_delay: float = 0.4
    endpointing_max_delay: float = 2.5
    # A dropped candidate (network, reload) can rejoin within this many seconds and carry on.
    rejoin_grace_secs: int = 60

    livekit_url: str | None = None
    livekit_api_key: str | None = None
    livekit_api_secret: str | None = None
    agent_name: str = "hiremind-interviewer"

    # Resume storage (MinIO / S3), same bucket the backend presigns uploads into.
    s3_endpoint: str | None = "http://localhost:9000"
    s3_bucket: str = "hiremind"
    s3_region: str = "us-east-1"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"

    def _speech_key(self, provider: str) -> str | None:
        return {"cartesia": self.cartesia_api_key, "groq": self.groq_api_key}.get(provider, self.openai_api_key)

    @property
    def voice_ready(self) -> bool:
        """LiveKit plus keys for the chosen STT and TTS. Without an LLM key the agent runs the scripted plan."""
        return bool(
            self.livekit_url
            and self.livekit_api_key
            and self.livekit_api_secret
            and self._speech_key(self.stt_provider)
            and self._speech_key(self.tts_provider)
        )

    def llm_key(self, provider: str) -> str | None:
        return {"gemini": self.gemini_api_key, "groq": self.groq_api_key, "claude": self.anthropic_api_key}.get(provider)

    def chain(self, primary: str, fallbacks: str, has_key) -> list[str]:
        """Primary first, then each fallback once, keeping only providers that have a key."""
        order = [primary, *(p.strip() for p in fallbacks.split(",") if p.strip())]
        return [p for i, p in enumerate(order) if p not in order[:i] and has_key(p)]

    @property
    def llm_chain(self) -> list[str]:
        return self.chain(self.llm_provider, self.llm_fallbacks, self.llm_key)

    @property
    def llm_model(self) -> str:
        return {"gemini": self.gemini_model, "groq": self.groq_model}.get(self.llm_provider, self.claude_model)

    @property
    def fake(self) -> bool:
        return self.ai_fake_mode or not self.llm_chain


@lru_cache
def get_settings() -> Settings:
    return Settings()
