"""Casting: fill each story beat with the single best real clip.

Takes the director's outline (see `director.py`) and, beat by beat in story order,
retrieves and scores candidate chunks against that beat's intent, then casts the best
unused one. Every clip ends up with a reason to exist (the beat it serves), and the
beats become titled sections -> AE section cards. Reuses the local embed/score/snap
pipeline -- no new model work here.
"""
from __future__ import annotations

K_RETRIEVE = 6   # candidates pulled per beat by embedding similarity (fallback path)
N_SCORE = 4      # of those, how many get the (costlier) local LLM relevance score


def _valid_bounds(s0, e0, stream_end: float) -> bool:
    """The director gave usable in/out timestamps for this beat (the primary path)."""
    if s0 is None or e0 is None:
        return False
    if e0 <= s0 or s0 < 0:
        return False
    return stream_end == 0.0 or s0 < stream_end


def cast(outline, chunks: list[dict], words: list[dict], target_s: float,
         model: str, tol: float = 0.25):
    """Outline -> (sections, outline_log, warning).

    Primary path: the director chose explicit in/out timestamps per beat, so we just snap
    them onto clean sentence bounds and score the chosen span (the one component that
    understands the story points at the footage directly). Fallback, when a beat's bounds
    are missing/invalid: retrieve candidates by its query, score the top few against its
    intent, and cast the best unused chunk. One clip per beat. Over budget -> drop the
    lowest-scored beats; under the tolerance band -> return a shortfall warning (never pad
    with filler). `sections` feeds `ae_export.build_manifest` unchanged; `outline_log` is
    the inspectable comprehension artifact written to run/outline.json.
    """
    from .retrieve import embed, retrieve_with_vec
    from .score import score_chunk
    from .select import snap_to_sentences

    stream_end = words[-1]["end"] if words else 0.0

    def span_text(a: float, b: float) -> str:
        return " ".join(w["text"] for w in words if a <= w["start"] < b)

    used: set[float] = set()           # chunk start times cast via the fallback path
    picked: list[dict] = []
    for beat in outline.beats:
        s0, e0 = getattr(beat, "start_s", None), getattr(beat, "end_s", None)
        if _valid_bounds(s0, e0, stream_end):
            # Primary: snap the director's span to clean sentence bounds, then score it
            # (one local call -- budget criterion + outline.json; cheaper than the N_SCORE
            # candidate scoring the fallback does).
            a, b = snap_to_sentences(words, float(s0), float(e0))
            res = score_chunk({"start": a, "end": b, "text": span_text(a, b)},
                              brief=beat.intent, model=model)
            sc, reason = res["llm_score"], res.get("reason", "")
        else:
            # Fallback: no usable bounds -> retrieve footage for this beat by its query.
            pool = [c for c in chunks if c["start"] not in used]
            if not pool:
                break
            qv = embed([beat.query])[0]
            cands = retrieve_with_vec(qv, pool, k=min(K_RETRIEVE, len(pool)))
            scored = []
            for c in cands[:N_SCORE]:
                r = score_chunk(c, brief=beat.intent, model=model)
                scored.append((r["llm_score"], c, r.get("reason", "")))
            if not scored:
                continue
            sc, c, reason = max(scored, key=lambda t: t[0])
            used.add(c["start"])
            a, b = snap_to_sentences(words, c["start"], c["end"])
            s0 = e0 = None             # mark this beat as fallback in the log
        picked.append({"title": beat.title, "intent": beat.intent,
                       "query": beat.query, "dir_start": s0, "dir_end": e0,
                       "start": a, "end": b, "score": sc, "reason": reason})

    # Budget: trim lowest-scored beats if we overran; warn (don't pad) if we fell short.
    def total() -> float:
        return sum(p["end"] - p["start"] for p in picked)

    while len(picked) > 1 and total() > target_s:
        weakest = min(picked, key=lambda p: p["score"])
        picked.remove(weakest)
    warning = None
    if picked and total() < target_s * (1.0 - tol):
        warning = (f"only {total():.0f}s of material cleared casting "
                   f"(target {target_s:.0f}s) -- not padding with filler.")

    # Emit in chronological order so the cut plays forward in time (the beat title
    # travels with its clip as the section card). For a linear stream this is also the
    # natural story order; it guarantees output is never reverse/scrambled.
    picked.sort(key=lambda p: p["start"])
    sections = [{"title": p["title"], "clips": [{"start": p["start"], "end": p["end"]}]}
                for p in picked]
    outline_log = {
        "central_idea": getattr(outline, "central_idea", ""),
        "beats": [
            {"title": p["title"], "intent": p["intent"], "query": p["query"],
             "dir_start": p["dir_start"], "dir_end": p["dir_end"],
             "start": round(p["start"], 2), "end": round(p["end"], 2),
             "score": p["score"], "reason": p["reason"]}
            for p in picked
        ],
    }
    return sections, outline_log, warning
