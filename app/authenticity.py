"""Signals that a spoken answer was read out from an AI tool rather than given in the candidate's own words.

Nothing here is proof on its own; it adds up weak signals so the interviewer can probe and the
recruiter can review:
- written style: connectives and phrasing people rarely say aloud ("furthermore", "it is important to note")
- list structure: "firstly ... secondly ... finally"
- no natural speech markers ("I mean", "you know", "kind of") in a long answer
- a long silence before answering, then a long fluent answer (typing a prompt, waiting, reading)
- a fast, steady speaking rate over a long answer (reading pace)
"""

import re
from dataclasses import dataclass, field

_WRITTEN = re.compile(
    r"\b(furthermore|moreover|additionally|in conclusion|in summary|to summari[sz]e|overall,|it is (important|worth|essential) to"
    r"|leverag(e|ing)|robust|seamless(ly)?|comprehensive|delve|holistic|paramount|multifaceted|plays a (crucial|vital|key) role"
    r"|in today's|a (wide|broad) range of|ensures? that|facilitat(e|es|ing)|utiliz(e|es|ing)|thereby|whereas)\b",
    re.IGNORECASE,
)
_ENUMERATION = re.compile(r"\b(firstly|secondly|thirdly|lastly|first of all|second(ly)?,|third(ly)?,|finally,)", re.IGNORECASE)
_SPOKEN = re.compile(
    r"\b(um+|uh+|hmm+|erm|i mean|you know|kind of|sort of|like,|basically|so yeah|actually|i think|i guess|honestly|let me think|okay so)\b",
    re.IGNORECASE,
)
_DEFINITION = re.compile(r"\b(refers to|is defined as|is a (process|technique|method|concept) (that|of|for|where))\b", re.IGNORECASE)


@dataclass
class AnswerSignals:
    score: float  # 0 = sounds spontaneous, 1 = strongly looks read from AI output
    signals: list[str] = field(default_factory=list)

    @property
    def suspicious(self) -> bool:
        return self.score >= 0.6


def analyze_answer(text: str, *, latency_s: float | None = None, speech_s: float | None = None) -> AnswerSignals:
    """Scores one candidate answer. latency_s: silence before they started; speech_s: time spent speaking."""
    words = len(text.split())
    if words < 25:
        # Short answers carry too little signal to judge.
        return AnswerSignals(0.0)
    score, signals = 0.0, []

    written = len(_WRITTEN.findall(text))
    if written:
        score += min(0.4, 0.15 * written)
        signals.append(f"{written} written-style phrase(s)")
    if len(_ENUMERATION.findall(text)) >= 2:
        score += 0.2
        signals.append("list-like structure (firstly/secondly)")
    if _DEFINITION.search(text):
        score += 0.1
        signals.append("textbook-style definition")
    if words >= 60 and not _SPOKEN.search(text):
        score += 0.2
        signals.append("long answer with no natural speech markers")
    if latency_s is not None and latency_s >= 6 and words >= 50:
        score += 0.25
        signals.append(f"{latency_s:.0f}s silence before a long, fluent answer")
    if speech_s and speech_s > 0:
        rate = words / speech_s
        if rate >= 3.3 and words >= 60:
            score += 0.15
            signals.append(f"steady reading pace ({rate:.1f} words/s)")

    return AnswerSignals(round(min(1.0, score), 2), signals)


def summarize(answers: list[AnswerSignals]) -> tuple[float, list[str]]:
    """Human-likelihood for a whole interview (1 = clearly spontaneous) and its strongest signals."""
    judged = [a for a in answers if a.signals or a.score]
    if not judged:
        return 1.0, []
    flagged = [a for a in answers if a.suspicious]
    worst = max(a.score for a in answers)
    # A single read-out answer matters; several matter more.
    ai_likelihood = min(1.0, 0.6 * worst + 0.2 * len(flagged))
    signals = sorted({s for a in flagged or judged for s in a.signals})
    return round(1 - ai_likelihood, 2), signals[:6]
