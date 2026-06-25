"""Choose and order which detected segments go into the compilation."""
from __future__ import annotations

from .rank import Segment


def select_segments(
    segments: list[Segment],
    count: int | None = None,
    min_score: float | None = None,
    order: str = "chrono",
) -> list[Segment]:
    """Filter by min_score, keep the top `count` by score, then order output.

    order="chrono" (default) -> ascending by start time (natural recap),
    order="score"            -> descending by score (best first).
    """
    segs = list(segments)
    if min_score is not None:
        segs = [s for s in segs if s["score"] >= min_score]
    segs.sort(key=lambda s: s["score"], reverse=True)
    if count is not None:
        segs = segs[:count]
    if order == "chrono":
        segs.sort(key=lambda s: s["start"])
    return segs
