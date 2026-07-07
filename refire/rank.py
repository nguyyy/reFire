"""Blend LLM + chat scores into a final ranked segment list."""
from __future__ import annotations

from typing import TypedDict


class Segment(TypedDict):
    start: float
    end: float
    score: float
    reason: str


def _chat_to_10(chat_z: float) -> float:
    """Map a chat z-score (~ -2..+4) onto 0-10, clamped."""
    return max(0.0, min(10.0, (chat_z + 2.0) * 2.0))


def rank_select(
    scored: list[dict],
    w_llm: float = 0.6,
    w_chat: float = 0.4,
    top_n: int | None = None,
    threshold: float | None = None,
) -> list[Segment]:
    """scored items carry start/end/chat_z/llm_score/reason.

    Final score = w_llm*llm_score + w_chat*normalized_chat. Sorted descending,
    then top_n and/or threshold applied.
    """
    segs: list[Segment] = []
    for s in scored:
        final = w_llm * s["llm_score"] + w_chat * _chat_to_10(s["chat_z"])
        segs.append({
            "start": s["start"], "end": s["end"],
            "score": round(final, 3), "reason": s.get("reason", ""),
        })

    segs.sort(key=lambda x: x["score"], reverse=True)
    if threshold is not None:
        segs = [x for x in segs if x["score"] >= threshold]
    if top_n is not None:
        segs = segs[:top_n]
    return segs


def _max_z(signal: list[tuple[float, float]] | None, a: float, b: float) -> float:
    """Max z of the (t_center, z) buckets whose center lands inside [a, b]."""
    return max((z for t, z in signal or [] if a <= t <= b), default=0.0)


def select_by_signal(chunks: list[dict], k: int,
                     audio_z: list[tuple[float, float]] | None = None) -> list[dict]:
    """No-brief candidate pool: the k chunks where the stream itself reacted hardest.

    The brief-driven flat path ranks by embedding similarity to the brief; with no brief
    there is nothing to embed against, so rank by excitement instead -- each chunk's
    chat-rate z (already on every chunk) plus its peak loudness z. Pure and order-stable.
    """
    def hype(c: dict) -> float:
        return max(0.0, float(c.get("chat_z", 0.0))) \
            + max(0.0, _max_z(audio_z, c["start"], c["end"]))
    return sorted(chunks, key=hype, reverse=True)[:k]
