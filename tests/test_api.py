import time

import jwt
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.interviewer import FAKE_SCRIPT, InterviewContext, build_instructions
from app.main import app
from app.resume import extract_text


@pytest.fixture(autouse=True)
def fake_mode(monkeypatch):
    monkeypatch.setenv("AI_FAKE_MODE", "true")
    monkeypatch.setenv("JWT_INTERNAL_SECRET", "test-secret")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def auth(aud: str = "ai-service") -> dict:
    now = int(time.time())
    token = jwt.encode({"sub": "hiremind-backend", "aud": aud, "iat": now, "exp": now + 60}, "test-secret", algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


CTX = {
    "interview_id": "iv1",
    "candidate_name": "Priya Sharma",
    "company_name": "Acme",
    "job_title": "Backend Engineer",
    "job_description": "Node.js, PostgreSQL",
    "resume_text": "Built payments APIs",
}

client = TestClient(app)


def test_rejects_missing_or_wrong_audience_token():
    assert client.post("/v1/fit/score", json={}).status_code == 401
    res = client.post(
        "/v1/fit/score",
        json={"companyId": "c", "jobDescription": "x", "resumeText": "y"},
        headers=auth("someone-else"),
    )
    assert res.status_code == 401


def test_fit_score_in_range():
    res = client.post("/v1/fit/score", json={"companyId": "c", "jobDescription": "x", "resumeText": "y"}, headers=auth())
    assert res.status_code == 200
    assert 0 <= res.json()["score"] <= 1


def test_chat_runs_full_scripted_interview():
    history: list[dict] = []
    for _ in range(len(FAKE_SCRIPT)):
        res = client.post("/v1/interview/chat", json={"context": CTX, "history": history}, headers=auth())
        assert res.status_code == 200
        turn = res.json()
        history += [{"role": "interviewer", "text": turn["reply"]}, {"role": "candidate", "text": "answer"}]
        if turn["done"]:
            break
    assert turn["done"] is True
    assert "Priya" in history[0]["text"]


def test_evaluate_returns_dimension_map():
    res = client.post(
        "/v1/interview/evaluate",
        json={"context": CTX, "transcript": [{"role": "interviewer", "text": "Q"}, {"role": "candidate", "text": "A"}]},
        headers=auth(),
    )
    body = res.json()
    assert res.status_code == 200
    assert body["recommendation"] in {"strong_hire", "hire", "hold", "no_hire"}
    assert set(body["dimensions"]) >= {"technical_depth", "communication"}


def test_instructions_include_resume_and_voice_rules():
    text = build_instructions(InterviewContext(**CTX), voice=True)
    assert "Built payments APIs" in text and "read aloud" in text


def test_extract_plain_text():
    assert extract_text(b"hello resume", "cv.txt") == "hello resume"
