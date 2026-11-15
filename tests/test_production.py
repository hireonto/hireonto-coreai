"""Production hardening: startup validation, internal auth, /health contract, body limit, R2 client, retries."""

import asyncio
import time

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from app.config import DEV_JWT_SECRET, Settings, get_settings
from app.main import app

SECRET = "x" * 40
PROD = {
    "APP_ENV": "production",
    "JWT_INTERNAL_SECRET": SECRET,
    "BACKEND_URL": "https://backend.example.com",
    "S3_ENDPOINT": "https://acct.r2.cloudflarestorage.com",
    "S3_BUCKET": "hiremind",
    "S3_REGION": "auto",
    "S3_ACCESS_KEY": "dummy-access",
    "S3_SECRET_KEY": "dummy-secret",
    "LLM_PROVIDER": "gemini",
    "GEMINI_API_KEY": "dummy-gemini",
}
ALL_VARS = [*PROD, "ENV", "AI_FAKE_MODE", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "LLM_FALLBACKS", "LIVEKIT_URL",
            "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "CARTESIA_API_KEY", "OPENAI_API_KEY", "MAX_BODY_BYTES"]


@pytest.fixture
def env(monkeypatch):
    """Clean environment (no .env file) with the given variables; returns a setter."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for var in ALL_VARS:
        monkeypatch.delenv(var, raising=False)

    def apply(values: dict) -> Settings:
        for k, v in values.items():
            monkeypatch.setenv(k, v)
        get_settings.cache_clear()
        return get_settings()

    yield apply
    get_settings.cache_clear()


def token(aud="ai-service", secret=SECRET, **extra) -> dict:
    now = int(time.time())
    claims = {"sub": "hiremind-backend", "aud": aud, "iat": now, "exp": now + 60, **extra}
    return {"Authorization": f"Bearer {jwt.encode(claims, secret, algorithm='HS256')}"}


# ---------- startup validation ----------


def test_valid_production_config_has_no_problems(env):
    assert env(PROD).production_problems() == []


def test_env_alias_turns_on_production(env):
    assert env({"ENV": "production"}).is_production


@pytest.mark.parametrize("var", ["JWT_INTERNAL_SECRET", "BACKEND_URL", "S3_ENDPOINT", "S3_BUCKET", "S3_ACCESS_KEY", "S3_SECRET_KEY"])
def test_missing_required_var_is_reported_by_name(env, var):
    problems = env({k: v for k, v in PROD.items() if k != var}).production_problems()
    assert any(p.startswith(var) for p in problems)


def test_dev_defaults_and_fake_mode_are_rejected(env):
    s = env({**PROD, "JWT_INTERNAL_SECRET": DEV_JWT_SECRET, "S3_ACCESS_KEY": "minioadmin",
             "BACKEND_URL": "http://localhost:3001", "AI_FAKE_MODE": "true"})
    text = " | ".join(s.production_problems())
    for expected in ("JWT_INTERNAL_SECRET still has", "S3_ACCESS_KEY still has", "BACKEND_URL points at", "AI_FAKE_MODE"):
        assert expected in text


def test_no_llm_key_is_rejected(env):
    problems = env({k: v for k, v in PROD.items() if k != "GEMINI_API_KEY"}).production_problems()
    assert any("no LLM key" in p and "GEMINI_API_KEY" in p for p in problems)


def test_problems_never_contain_secret_values(env):
    s = env({**PROD, "S3_ENDPOINT": "http://localhost:9000", "GEMINI_API_KEY": ""})
    assert not any(SECRET in p or "dummy" in p for p in s.production_problems())


def test_startup_refuses_without_jwt_secret(env):
    env({k: v for k, v in PROD.items() if k != "JWT_INTERNAL_SECRET"})
    with pytest.raises(RuntimeError, match="JWT_INTERNAL_SECRET"):
        with TestClient(app):
            pass


def test_startup_ok_with_production_config(env):
    env(PROD)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200


def test_llm_chain_skips_providers_without_key(env):
    s = env({**PROD, "GEMINI_API_KEY": "", "GROQ_API_KEY": "dummy-groq", "LLM_FALLBACKS": "groq,claude"})
    assert s.llm_chain == ["groq"] and not s.fake and s.production_problems() == []


# ---------- internal auth ----------


def test_health_shape_unchanged(env):
    env(PROD)
    body = TestClient(app).get("/health").json()
    assert set(body) == {"status", "service", "fakeMode", "llmProvider", "model", "voiceReady"}
    assert body["voiceReady"] is False and body["status"] == "ok"
    assert "dummy" not in str(body) and SECRET not in str(body)


ROUTES = ["/v1/resume/parse", "/v1/fit/score", "/v1/interview/chat", "/v1/interview/plan", "/v1/interview/evaluate"]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("headers", [{}, token(aud="hiremind-backend"), token(secret="y" * 40), token(exp=1)],
                         ids=["missing", "wrong-audience", "wrong-secret", "expired"])
def test_routes_reject_bad_tokens_with_generic_401(env, route, headers):
    env(PROD)
    res = TestClient(app).post(route, json={}, headers=headers)
    assert res.status_code == 401
    assert res.json() == {"detail": "Unauthorized"}


def test_body_size_limit(env):
    env({**PROD, "MAX_BODY_BYTES": "1000"})
    res = TestClient(app).post("/v1/fit/score", content=b"x" * 2000, headers={**token(), "content-type": "application/json"})
    assert res.status_code == 413


# ---------- S3 / R2 ----------


def test_r2_style_s3_client(env):
    from app import resume

    env(PROD)
    resume._client.cache_clear()
    meta = resume._s3().meta
    assert meta.endpoint_url == "https://acct.r2.cloudflarestorage.com"
    assert meta.region_name == "auto"
    assert meta.config.signature_version == "s3v4"
    assert meta.config.s3["addressing_style"] == "path"
    assert meta.config.connect_timeout == 5 and meta.config.read_timeout == 30
    assert meta.config.retries["mode"] == "standard" and meta.config.retries["total_max_attempts"] == 3
    assert meta.config.request_checksum_calculation == "when_required"


# ---------- voice agent ----------


def test_agent_exits_without_livekit(env, monkeypatch):
    from agent import worker

    monkeypatch.setattr(worker, "settings", env(PROD))
    with pytest.raises(SystemExit) as exc:
        worker.check_config()
    assert "LIVEKIT_URL" in str(exc.value) and "dummy" not in str(exc.value)


def test_backend_client_retries_get_but_not_post(env, monkeypatch):
    from agent import backend

    env(PROD)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(backend.asyncio, "sleep", lambda _: real_sleep(0))
    calls = {"GET": 0, "POST": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls[request.method] += 1
        claims = jwt.decode(request.headers["authorization"][7:], SECRET, algorithms=["HS256"], audience="hiremind-backend")
        assert claims["sub"] == "hiremind-ai"
        if request.method == "GET" and calls["GET"] < 3:
            return httpx.Response(503)
        return httpx.Response(502 if request.method == "POST" else 200, json={})

    async def run():
        client = backend.BackendClient()
        client._http = httpx.AsyncClient(base_url="https://backend.example.com", transport=httpx.MockTransport(handler))
        assert await client._req("GET", "/internal/interviews/1/context") == {}
        with pytest.raises(httpx.HTTPStatusError):
            await client.add_turn("1", "candidate", "hi")

    asyncio.run(run())
    assert calls == {"GET": 3, "POST": 1}
