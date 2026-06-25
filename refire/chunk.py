"""Group transcript words into ~30-90s chunks on natural boundaries."""
from __future__ import annotations

from typing import TypedDict

from .transcribe import Word


class Chunk(TypedDict):
    start: float
    end: float
    text: str
    chat_z: float


def _chat_z_for(span_start: float, span_end: float,
                chat_signal: list[tuple[float, float]]) -> float:
    """Max chat z-score whose bin center falls within the span."""
    zs = [z for (t, z) in chat_signal if span_start <= t < span_end]
    return max(zs) if zs else 0.0


def make_chunks(
    words: list[Word],
    chat_signal: list[tuple[float, float]],
    min_len: float = 30.0,
    max_len: float = 90.0,
    pause_gap: float = 0.8,
) -> list[Chunk]:
    """Window words into chunks, breaking on sentence end or long pause once
    past min_len, and force-breaking at max_len."""
    chunks: list[Chunk] = []
    if not words:
        return chunks

    buf: list[Word] = []
    start = words[0]["start"]

    def flush(end: float) -> None:
        if not buf:
            return
        text = " ".join(w["text"] for w in buf).strip()
        chunks.append({
            "start": start, "end": end, "text": text,
            "chat_z": _chat_z_for(start, end, chat_signal),
        })

    for i, w in enumerate(words):
        buf.append(w)
        dur = w["end"] - start
        nxt_gap = (words[i + 1]["start"] - w["end"]) if i + 1 < len(words) else 0.0
        sentence_end = w["text"][-1:] in ".?!"
        if dur >= max_len or (dur >= min_len and (sentence_end or nxt_gap >= pause_gap)):
            flush(w["end"])
            buf = []
            if i + 1 < len(words):
                start = words[i + 1]["start"]

    flush(words[-1]["end"])
    return chunks
