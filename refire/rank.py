"""Rank chunks by raw stream excitement (chat rate + loudness) when there's no brief."""
from __future__ import annotations


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
