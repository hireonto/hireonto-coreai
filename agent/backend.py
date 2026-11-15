"""Calls from the voice agent to hiremind-backend's internal interview endpoints."""

import time

import httpx
import jwt

from app.config import get_settings
from app.interviewer import InterviewContext


def _token() -> str:
    s = get_settings()
    now = int(time.time())
    return jwt.encode(
        {"sub": "hiremind-ai", "aud": "hiremind-backend", "iat": now, "exp": now + 300},
        s.jwt_internal_secret,
        algorithm="HS256",
    )


class BackendClient:
    def __init__(self) -> None:
        self._http = httpx.AsyncClient(base_url=get_settings().backend_url, timeout=15)

    async def _req(self, method: str, path: str, json: dict | None = None) -> dict:
        res = await self._http.request(method, path, json=json, headers={"Authorization": f"Bearer {_token()}"})
        res.raise_for_status()
        return res.json() if res.content else {}

    async def context(self, interview_id: str) -> InterviewContext:
        return InterviewContext.model_validate(await self._req("GET", f"/internal/interviews/{interview_id}/context"))

    async def add_turn(self, interview_id: str, role: str, text: str) -> None:
        await self._req("POST", f"/internal/interviews/{interview_id}/turns", {"role": role, "text": text})

    async def proctoring_event(self, interview_id: str, event_type: str, severity: str, detail: str) -> None:
        await self._req(
            "POST",
            f"/internal/interviews/{interview_id}/proctoring",
            {"eventType": event_type, "severity": severity, "detail": detail},
        )

    async def complete(self, interview_id: str, reason: str) -> None:
        await self._req("POST", f"/internal/interviews/{interview_id}/complete", {"reason": reason})

    async def aclose(self) -> None:
        await self._http.aclose()
