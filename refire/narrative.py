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

SEG_MERGE_GAP = 0.75    # segments closer than this after snapping merge (no stutter cuts)
WIDEN_SLOP = 15.0       # max s an editorial anchor may widen past the director's own bounds
WIDEN_SLOP_FINAL = 30.0  # the final beat gets extra room so the ending still breathes

# Budget trimming protects the story instead of shedding whatever the local scorer thinks
# is boring (that would re-introduce the score-driven selection the director exists to
# replace, killing low-scoring but essential setup/connective beats). Lower = more
# protected: the arc spine (open/peak/close) survives, then the connective setup, and the
# most expendable redundant escalations go first.
_DROP_PRIORITY = {"hook": 0, "climax": 0, "button": 0, "setup": 1,
                  "reversal": 2, "payoff": 2, "escalation": 3}


def _drop_rank(p: dict):
    """Sort key for the most-droppable beat: highest role priority, then lowest score."""
    return (_DROP_PRIORITY.get(p.get("role", ""), 2), -p["score"])


def _dur(p: dict) -> float:
    """A beat's KEPT footage length (sum of its segments), not its envelope span."""
    return p.get("dur", p["end"] - p["start"])


def _trim_to_budget(picked: list[dict], target_s: float, tol: float):
    """Drop beats only past the tolerance CEILING (target is a goal, not a hard cap), and
    drop the most expendable first (`_drop_rank`) so the arc survives. Returns
    (kept, dropped) -- the dropped list is surfaced to the critic so it stops re-adding
    budget-trimmed beats at full length every round. tol widens both ways: overshoot up
    to target*(1+tol) ships untrimmed."""
    ceil = target_s * (1.0 + tol)
    kept = list(picked)
    dropped: list[dict] = []
    while len(kept) > 1 and sum(_dur(p) for p in kept) > ceil:
        loser = max(kept, key=_drop_rank)
        kept.remove(loser)
        dropped.append(loser)
    return kept, dropped


COLD_OPEN_MAX_S = 8.0     # a teaser, not a scene
COLD_OPEN_MIN_S = 1.5     # shorter than this reads as a glitch, not a promise
COLD_OPEN_NEAR_S = 30.0   # a teaser this close to the opening beat is just the opening beat


def _cold_open(outline, picked: list[dict], words: list[dict]) -> dict | None:
    """The flash-forward: a few seconds of the peak, played before the story starts.

    This is the ONE place the cut is allowed out of chronological order, and it is the
    lever against a middle that loses people -- a viewer who has seen where this is going
    sits through the setup to get there. Guarded deterministically rather than trusted,
    because a teaser the cut never delivers is worse than no teaser: the span must overlap
    footage a real beat already contains, it is clamped to a teaser's length, and it is
    dropped when the first beat is already right there (a stutter, not a promise).
    """
    from .select import snap_to_sentences

    seg = getattr(outline, "cold_open", None)
    if seg is None or not picked:
        return None
    a, b = float(getattr(seg, "start_s", 0.0)), float(getattr(seg, "end_s", 0.0))
    if b <= a:
        return None
    b = min(b, a + COLD_OPEN_MAX_S)
    if not any(p["start"] <= a < p["end"] for p in picked):
        return None                        # promises footage the cut doesn't deliver
    if abs(a - min(p["start"] for p in picked)) < COLD_OPEN_NEAR_S:
        return None                        # the opening beat already is this moment
    # snap so it can't open or stop mid-word, but with a tight pad -- a sentence-length
    # pull here would turn a teaser back into a scene. The pad can carry it ~2s past
    # COLD_OPEN_MAX_S; landing on a clean word beats holding the ceiling exactly.
    x, y = snap_to_sentences(words, a, b, max_pad=2.0)
    if y - x < COLD_OPEN_MIN_S:
        return None
    return {"start": x, "end": y, "dur": y - x}


def _valid_bounds(s0, e0, stream_end: float) -> bool:
    """The director gave usable in/out timestamps for this beat (the primary path)."""
    if s0 is None or e0 is None:
        return False
    if e0 <= s0 or s0 < 0:
        return False
    return stream_end == 0.0 or s0 < stream_end


def cast(outline, chunks, words: list[dict], target_s: float,
         model: str, tol: float = 0.35, progress=None):
    """Outline -> (sections, outline_log, warning).

    Primary path: the director chose explicit in/out timestamps per beat, so we just snap
    them onto clean sentence bounds and score the chosen span (the one component that
    understands the story points at the footage directly). Fallback, when a beat's bounds
    are missing/invalid: retrieve candidates by its query, score the top few against its
    intent, and cast the best unused chunk. One clip per beat. Over budget -> drop the
    lowest-scored beats; under the tolerance band -> return a shortfall warning (never pad
    with filler). `sections` feeds `ae_export.build_manifest` unchanged; `outline_log` is
    the inspectable comprehension artifact written to run/outline.json.

    `chunks` may be a list, or a zero-arg callable returning one -- only the fallback
    branch below needs chunk embeddings, and computing them costs a full local embedding
    pass over the VOD, so the caller can defer that until a beat actually needs it.
    """
    from .retrieve import embed, retrieve_with_vec
    from .score import score_chunk
    from .select import snap_to_sentences

    resolved: list[list[dict]] = []       # 1-slot memo so we embed at most once per cast

    def get_chunks() -> list[dict]:
        if not resolved:
            resolved.append(chunks() if callable(chunks) else chunks)
        return resolved[0]

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
            # Primary: the director's EDIT. Its `segments` (the exact kept lines, jump
            # cuts between them) are authoritative; a beat without usable segments is one
            # span, exactly the old behavior. Each kept span snaps to clean sentence
            # bounds, then the joined kept text gets one local score (budget criterion +
            # outline.json; cheaper than the N_SCORE candidate scoring the fallback does).
            segs = sorted(
                (float(s.start_s), float(s.end_s))
                for s in (getattr(beat, "segments", None) or [])
                if _valid_bounds(getattr(s, "start_s", None),
                                 getattr(s, "end_s", None), stream_end))
            if not segs:
                segs = [(float(s0), float(e0))]
            # Editorial anchors widen the EDGES only -- context before the first kept
            # line, payoff/reaction after the last -- clamped so a stray anchor can't
            # silently re-inflate a cut the director (or critic) deliberately tightened.
            is_final = i == n_beats - 1
            slop = WIDEN_SLOP_FINAL if is_final else WIDEN_SLOP
            setup = getattr(beat, "setup_start_s", None)
            payoff = getattr(beat, "payoff_start_s", None)
            reaction = getattr(beat, "reaction_end_s", None)
            fa, fb = segs[0]
            if setup is not None:                     # start before the context if needed
                segs[0] = (max(min(fa, float(setup)), fa - slop), fb)
            la, lb = segs[-1]
            out = lb
            if payoff is not None:
                out = max(out, float(payoff))         # never cut before the payoff lands
            if reaction is not None:
                out = max(out, float(reaction))       # keep the reaction tail
            segs[-1] = (la, min(out, lb + slop))
            # snap each kept span; the final beat's LAST span is the most visible ending
            # -> be generous + fall back to a clean breath when whisper dropped the
            # closing punctuation.
            snapped = []
            for j, (sa, sb) in enumerate(segs):
                closing = is_final and j == len(segs) - 1
                x, y = snap_to_sentences(words, sa, sb,
                                         max_pad=8.0 if closing else 5.0,
                                         phrase_fallback=closing)
                snapped.append([x, y])
            spans = [snapped[0]]
            for x, y in snapped[1:]:                  # merge overlaps/near-touches
                if x <= spans[-1][1] + SEG_MERGE_GAP:
                    spans[-1][1] = max(spans[-1][1], y)
                else:
                    spans.append([x, y])
            spans = [(x, y) for x, y in spans]
            a, b = spans[0][0], spans[-1][1]          # envelope (ordering + display)
            text = " ".join(t for x, y in spans if (t := span_text(x, y)))
            res = score_chunk({"start": a, "end": b, "text": text},
                              brief=beat.intent, model=model)
            sc, reason = res["llm_score"], res.get("reason", "")
        else:
            # Fallback: no usable bounds -> retrieve footage for this beat by its query.
            pool = [c for c in get_chunks() if c["start"] not in used]
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
            spans = [(a, b)]
            text = span_text(a, b)
            s0 = e0 = None             # mark this beat as fallback in the log
        picked.append({"title": beat.title, "intent": beat.intent,
                       "query": getattr(beat, "query", ""),
                       "role": getattr(beat, "role", ""),
                       "viewer_question": getattr(beat, "viewer_question", ""),
                       "turn": getattr(beat, "turn", ""),
                       "transition_in": getattr(beat, "transition_in", ""),
                       "texture": getattr(beat, "texture", ""),
                       "energy": getattr(beat, "energy", 3),
                       # editorial anchors travel into the log too -- the pacing audit
                       # measures setup lead (how long before the point lands) from them
                       "setup_start_s": getattr(beat, "setup_start_s", None),
                       "payoff_start_s": getattr(beat, "payoff_start_s", None),
                       "dir_start": s0, "dir_end": e0,
                       "start": a, "end": b,
                       "segments": spans,                       # the kept spans (the edit)
                       "dur": sum(y - x for x, y in spans),     # kept footage, not envelope
                       "score": sc, "reason": reason,
                       "text": text})   # realized KEPT transcript -> editor-review

    # Budget: only shed beats that overrun the tolerance ceiling, and shed the most
    # expendable first so a low-scoring setup/connective beat isn't cut for being "boring"
    # (see _trim_to_budget). Warn (don't pad) if we fell short of the lower band.
    picked, dropped = _trim_to_budget(picked, target_s, tol)

    def total() -> float:
        return sum(_dur(p) for p in picked)

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
    # One clip PER KEPT SPAN -- multiple clips in a section render as jump cuts within
    # the beat (build_manifest/render_clips already handle multi-clip sections).
    sections = [{"title": p["title"], "role": p["role"], "energy": p["energy"],
                 "clips": [{"start": a, "end": b,
                            "role": p["role"], "energy": p["energy"]}
                           for a, b in p["segments"]]}
                for p in picked]
    # The teaser goes in AFTER the chronological sort (it is the one clip allowed to play
    # out of stream order) and after the budget trim, so a promise can never be the thing
    # that gets shed. Roled `hook` so the style pass gives it no section card.
    # ponytail: its ~5s aren't subtracted from the budget -- under 1% of a 15-minute
    # target. Count it if teasers ever get long enough to matter.
    cold = _cold_open(outline, picked, words)
    if cold:
        sections.insert(0, {"title": "Cold Open", "role": "hook", "energy": 5,
                            "clips": [{"start": cold["start"], "end": cold["end"],
                                       "role": "hook", "energy": 5}]})
    outline_log = {
        "central_idea": getattr(outline, "central_idea", ""),
        "story_shape": getattr(outline, "story_shape", ""),
        "viewer_promise": getattr(outline, "viewer_promise", ""),
        "ending_needed": getattr(outline, "ending_needed", ""),
        "cold_open": cold,
        "beats": [
            {"title": p["title"], "role": p["role"], "intent": p["intent"],
             "viewer_question": p["viewer_question"], "turn": p["turn"],
             "transition_in": p["transition_in"], "texture": p["texture"],
             "energy": p["energy"], "query": p["query"],
             "setup_start_s": p["setup_start_s"], "payoff_start_s": p["payoff_start_s"],
             "dir_start": p["dir_start"], "dir_end": p["dir_end"],
             "start": round(p["start"], 2), "end": round(p["end"], 2),
             "segments": [[round(a, 2), round(b, 2)] for a, b in p["segments"]],
             "dur": round(p["dur"], 2),
             "score": p["score"], "reason": p["reason"], "text": p["text"]}
            for p in picked
        ],
        # what the budget trim shed -- the critic reads this via _realized_script so it
        # tightens other beats instead of re-adding these at full length every round.
        "dropped": [{"title": p["title"], "role": p["role"],
                     "dur": round(_dur(p), 2)} for p in dropped],
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
    if outline_log.get("cold_open"):
        co = outline_log["cold_open"]
        out += [f"**Cold open:** {_mmss(co['start'])}-{_mmss(co['end'])} "
                f"({co['dur']:.1f}s flash-forward)", ""]

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
        segs = b.get("segments") or []
        if len(segs) > 1:
            out.append(f"- Cut: {len(segs)} segments, {int(b.get('dur') or 0)}s kept")
        if b.get("reason"):
            out.append(f"- Payoff: {b['reason']}")
        out.append("")

    # the same numbers the critic was judged against -- so a human can see the sag the
    # review round was reacting to without scrubbing the cut
    from .pacing import audit_note
    out.append("## Pacing audit")
    out.append("")
    out.append(audit_note(outline_log))
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
