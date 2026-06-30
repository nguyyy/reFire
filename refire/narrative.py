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
         model: str, tol: float = 0.25, progress=None):
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

    report = progress or (lambda *_a, **_k: None)
    used: set[float] = set()           # chunk start times cast via the fallback path
    picked: list[dict] = []
    n_beats = len(outline.beats)
    for i, beat in enumerate(outline.beats):
        report(i / n_beats, f"casting beat {i + 1}/{n_beats}: {beat.title}")
        s0, e0 = getattr(beat, "start_s", None), getattr(beat, "end_s", None)
        if _valid_bounds(s0, e0, stream_end):
            # Primary: widen the director's span by its editorial anchors, snap to clean
            # bounds, then score it (one local call -- budget criterion + outline.json;
            # cheaper than the N_SCORE candidate scoring the fallback does).
            s_in, e_out = float(s0), float(e0)
            setup = getattr(beat, "setup_start_s", None)
            payoff = getattr(beat, "payoff_start_s", None)
            reaction = getattr(beat, "reaction_end_s", None)
            if setup is not None:
                s_in = min(s_in, float(setup))        # start before the context if needed
            if payoff is not None:
                e_out = max(e_out, float(payoff))     # never cut before the payoff lands
            if reaction is not None:
                e_out = max(e_out, float(reaction))   # keep the reaction tail
            # final clip is the most visible ending -> be generous + fall back to a clean
            # breath when whisper dropped the closing punctuation.
            is_final = i == n_beats - 1
            a, b = snap_to_sentences(words, s_in, e_out,
                                     max_pad=8.0 if is_final else 5.0,
                                     phrase_fallback=is_final)
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
                       "query": getattr(beat, "query", ""),
                       "role": getattr(beat, "role", ""),
                       "viewer_question": getattr(beat, "viewer_question", ""),
                       "turn": getattr(beat, "turn", ""),
                       "transition_in": getattr(beat, "transition_in", ""),
                       "texture": getattr(beat, "texture", ""),
                       "energy": getattr(beat, "energy", 3),
                       "dir_start": s0, "dir_end": e0,
                       "start": a, "end": b, "score": sc, "reason": reason,
                       "text": span_text(a, b)})   # realized transcript -> editor-review

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
    # role/energy travel with the clip so the style pass (ae_export/overlay/render) can
    # let the story drive presentation: hook punches in, button stays out of the laugh.
    sections = [{"title": p["title"], "role": p["role"], "energy": p["energy"],
                 "clips": [{"start": p["start"], "end": p["end"],
                            "role": p["role"], "energy": p["energy"]}]}
                for p in picked]
    outline_log = {
        "central_idea": getattr(outline, "central_idea", ""),
        "story_shape": getattr(outline, "story_shape", ""),
        "viewer_promise": getattr(outline, "viewer_promise", ""),
        "ending_needed": getattr(outline, "ending_needed", ""),
        "beats": [
            {"title": p["title"], "role": p["role"], "intent": p["intent"],
             "viewer_question": p["viewer_question"], "turn": p["turn"],
             "transition_in": p["transition_in"], "texture": p["texture"],
             "energy": p["energy"], "query": p["query"],
             "dir_start": p["dir_start"], "dir_end": p["dir_end"],
             "start": round(p["start"], 2), "end": round(p["end"], 2),
             "score": p["score"], "reason": p["reason"], "text": p["text"]}
            for p in picked
        ],
    }
    return sections, outline_log, warning


def _mmss(s) -> str:
    s = int(s or 0)
    return f"{s // 60:02d}:{s % 60:02d}"


def cut_plan_md(outline_log: dict) -> str:
    """Render the editor's-notebook Markdown cut plan from an `outline_log` (see `cast`).

    Pure -- written beside run/outline.json so taste is debuggable by a human or agent:
    if the cut plan reads boring, the video will too. Tolerates missing fields (flat
    fallback never calls this) and missing critic notes (review-rounds=0 / local).
    """
    out = ["# Cut Plan", ""]
    if outline_log.get("story_shape"):
        out += [f"**Story shape:** {outline_log['story_shape']}", ""]
    out += ["**Central idea:**", outline_log.get("central_idea", ""), ""]
    if outline_log.get("viewer_promise"):
        out += [f"**Viewer promise:** {outline_log['viewer_promise']}", ""]
    if outline_log.get("ending_needed"):
        out += [f"**Ending needed:** {outline_log['ending_needed']}", ""]

    for i, b in enumerate(outline_log.get("beats", []), 1):
        role = b.get("role") or "?"
        out.append(f"## Beat {i} - {b.get('title', '')}  ({role})")
        out.append("")
        out.append(f"- Why it exists: {b.get('intent', '')}")
        if b.get("transition_in"):
            out.append(f"- Follows because: {b['transition_in']}")
        if b.get("viewer_question"):
            out.append(f"- Leaves viewer wondering: {b['viewer_question']}")
        if b.get("texture") or b.get("energy") is not None:
            out.append(f"- Texture/energy: {b.get('texture', '?')} / {b.get('energy', '?')}")
        out.append(f"- Clip: {_mmss(b.get('start'))}-{_mmss(b.get('end'))} "
                   f"(score {b.get('score', '?')})")
        if b.get("reason"):
            out.append(f"- Payoff: {b['reason']}")
        out.append("")

    review = outline_log.get("review") or []
    if review:
        out.append("## Critic notes")
        out.append("")
        out.append(f"Rounds: {outline_log.get('rounds', len(review))}")
        out.append("")
        for r in review:
            verdict = "approved" if r.get("approved") else "revised"
            out.append(f"- Round {r.get('round', '?')} ({verdict}): {r.get('notes', '')}")
        out.append("")
    return "\n".join(out).strip() + "\n"
