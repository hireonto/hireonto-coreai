"""Per-call helpers for the voice agent: answer timing and the job's knowledge base."""

import time

from app.interviewer import InterviewContext
from app.planner import gather_documents
from app.rag import BM25, chunk_text


class AnswerTimer:
    """Collects one answer: everything the candidate says between two interviewer turns.

    latency is the silence between Tovia finishing and the candidate starting to speak;
    speech is the time they actually spent speaking. Both feed the authenticity check.
    """

    def __init__(self) -> None:
        self._agent_done: float | None = None
        self._speaking_since: float | None = None
        self.reset()

    def reset(self) -> None:
        self.text = ""
        self.latency: float | None = None
        self.speech = 0.0
        self.flagged = False

    def agent_started(self) -> None:
        self.reset()
        self._agent_done = None

    def agent_finished(self) -> None:
        self._agent_done = time.monotonic()

    def user_started(self) -> None:
        now = time.monotonic()
        if self.latency is None and self._agent_done is not None:
            self.latency = now - self._agent_done
        self._speaking_since = now

    def user_stopped(self) -> None:
        if self._speaking_since is not None:
            self.speech += time.monotonic() - self._speaking_since
            self._speaking_since = None

    def add_text(self, text: str) -> str:
        self.text = f"{self.text} {text}".strip()
        return self.text


class JobKnowledge:
    """BM25 over the job description and the documents the company uploaded for the role."""

    def __init__(self, index: BM25 | None) -> None:
        self._index = index

    @classmethod
    async def load(cls, ctx: InterviewContext) -> "JobKnowledge":
        docs = await gather_documents(ctx)
        chunks = chunk_text(docs["question_bank"], "question_bank", max_words=80) + chunk_text(docs["jd"], "jd", max_words=80)
        return cls(BM25(chunks) if chunks else None)

    @property
    def available(self) -> bool:
        return self._index is not None

    def search(self, query: str, k: int = 4) -> list[str]:
        if not self._index:
            return []
        return [f"[{c.source}] {c.text}" for c, _ in self._index.search(query, k=k)]
