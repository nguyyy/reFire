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


def parse_duration(text) -> float:
    """'20m' / '90s' / '1.5h' / '20:00' / 'hh:mm:ss' / bare seconds -> seconds."""
    t = str(text).strip().lower()
    if ":" in t:                       # mm:ss or hh:mm:ss
        sec = 0.0
        for part in t.split(":"):
            sec = sec * 60.0 + float(part)
        return sec
    mult = 1.0
    if t.endswith("h"):
        mult, t = 3600.0, t[:-1]
    elif t.endswith("m"):
        mult, t = 60.0, t[:-1]
    elif t.endswith("s"):
        mult, t = 1.0, t[:-1]
    return float(t) * mult


def budget_select(scored, target_s: float, tol: float = 0.25, order: str = "chrono"):
    """Greedily pick highest-scored clips until total duration reaches target_s.

    Items need start/end/score; durations are measured as end-start, so callers
    that snap to sentences should pass POST-snap bounds for an accurate budget.
    Stops once the target is met. If the pool can't reach the lower tolerance band
    it returns what it has plus a warning string (never pads with filler).
    Returns (clips, warning|None); clips ordered chrono (default) or by score.
    """
    lo = target_s * (1.0 - tol)
    picked, total = [], 0.0
    for s in sorted(scored, key=lambda x: x["score"], reverse=True):
        if total >= target_s:
            break
        picked.append(s)
        total += s["end"] - s["start"]
    warning = None
    if total < lo:
        warning = (f"only {total:.0f}s of material cleared selection "
                   f"(target {target_s:.0f}s) -- not padding with filler.")
    if order == "chrono":
        picked.sort(key=lambda s: s["start"])
    else:
        picked.sort(key=lambda s: s["score"], reverse=True)
    return picked, warning


def snap_to_sentences(words, start: float, end: float, max_pad: float = 5.0):
    """Nudge a clip's [start, end] onto sentence boundaries so it never cuts mid-sentence.

    Pulls `start` back to the first word of its sentence and extends `end` forward to
    the next word ending in .?!, each capped by `max_pad` seconds so we never drag in
    unrelated audio. Words are {text,start,end} in absolute seconds.
    """
    # start of the sentence that `start` falls in (word after the previous terminator)
    sent_start = None
    prev_terminated = True               # first word always begins a sentence
    for w in words:
        if w["start"] > start:
            break
        if prev_terminated:
            sent_start = w["start"]
        prev_terminated = w["text"][-1:] in ".?!"
    new_start = sent_start if (sent_start is not None and start - sent_start <= max_pad) else start

    # end of the sentence that `end` falls in (first terminator at/after end)
    new_end = end
    for w in words:
        if w["end"] >= end and w["text"][-1:] in ".?!":
            if w["end"] - end <= max_pad:
                new_end = w["end"]
            break

    return max(0.0, new_start), max(new_end, new_start)


def speech_intervals(words, max_gap: float = 0.4) -> list[tuple[float, float]]:
    """Merge words into speaking runs, splitting on silence >= max_gap (seconds).

    A run is one continuous phrase; the gaps between runs are the natural pauses
    where a zoom may release without cutting off mid-sentence. Absolute seconds.
    """
    runs: list[list[float]] = []
    for w in words:
        if runs and w["start"] - runs[-1][1] < max_gap:
            runs[-1][1] = max(runs[-1][1], w["end"])
        else:
            runs.append([w["start"], w["end"]])
    return [(s, e) for s, e in runs]


def _demo() -> None:
    ws = [{"text": t, "start": i * 1.0, "end": i * 1.0 + 0.8}
          for i, t in enumerate("Today we review builds. Alright lets go now.".split())]
    # contiguous 0.2s gaps -> one speech run spanning all words
    si = speech_intervals(ws)
    assert len(si) == 1 and si[0] == (0.0, ws[-1]["end"]), si
    # a 1s pause splits into two runs
    ws2 = ws + [{"text": "later", "start": ws[-1]["end"] + 1.0, "end": ws[-1]["end"] + 1.5}]
    assert len(speech_intervals(ws2)) == 2, speech_intervals(ws2)
    # "builds." ends at index 3 (end 3.8); cut mid-sentence at 2.5..3.0 -> snap out
    s, e = snap_to_sentences(ws, 2.5, 3.0)
    assert s == 0.0, s              # pulled back to sentence start "Today"
    assert e == 3.8, e             # extended to "builds." end
    # cap respected: huge gap -> no snap
    s2, e2 = snap_to_sentences(ws, 2.5, 3.0, max_pad=0.1)
    assert (s2, e2) == (2.5, 3.0), (s2, e2)
    print("snap_to_sentences ok")


if __name__ == "__main__":
    _demo()
