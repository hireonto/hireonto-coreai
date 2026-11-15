"""Small retrieval layer for interview planning: chunk documents, rank chunks with BM25.

The corpus per interview is tiny (a JD, a resume, a few question banks), so an
in-memory lexical index is fast, deterministic and needs no embedding provider.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-z0-9+#.]+")
_STOP = set(
    "a an the and or of to in on for with by at as is are was were be been this that these those it its "
    "we you your our they their i me my he she his her from into over under will would can could should "
    "have has had do does did not no yes than then so such about across within per via etc using use used".split()
)


def tokenize(text: str) -> list[str]:
    return [t.strip(".") for t in _WORD.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class Chunk:
    id: str
    source: str  # "jd" | "resume" | "question_bank"
    text: str
    tokens: list[str] = field(default_factory=list)


def chunk_text(text: str, source: str, *, max_words: int = 90, overlap: int = 20) -> list[Chunk]:
    """One chunk per line/bullet (resumes and JDs are line-oriented); long lines are windowed."""
    lines = [re.sub(r"^\s*([-•*]|\d+[.)])\s*", "", ln).strip() for ln in (text or "").splitlines()]
    chunks: list[Chunk] = []
    for line in filter(None, lines):
        words = line.split()
        step = max(1, max_words - overlap)
        for start in range(0, len(words), step):
            piece = " ".join(words[start : start + max_words])
            chunks.append(Chunk(id=f"{source}-{len(chunks)}", source=source, text=piece, tokens=tokenize(piece)))
            if start + max_words >= len(words):
                break
    return chunks


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.4, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.avgdl = sum(len(c.tokens) for c in chunks) / max(1, len(chunks))
        df: Counter[str] = Counter()
        for c in chunks:
            df.update(set(c.tokens))
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(c.tokens) for c in chunks]

    def search(self, query: str, *, k: int = 3, source: str | None = None) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scored: list[tuple[Chunk, float]] = []
        for chunk, tf in zip(self.chunks, self.tf):
            if source and chunk.source != source:
                continue
            dl = len(chunk.tokens) or 1
            score = sum(
                self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                for t in q
                if t in tf
            )
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
