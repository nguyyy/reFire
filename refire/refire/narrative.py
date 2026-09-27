"""Casting: fill each story beat with the single best real clip.

Takes the director's outline (see `director.py`) and, beat by beat in story order,
retrieves and scores candidate chunks against that beat's intent, then casts the best
unused one. Every clip ends up with a reason to exist (the beat it serves), and the
beats become titled sections -> AE section cards. Reuses the local embed/score/snap
pipeline -- no new model work here.
"""
from __future__ import annotations

from .select import SILENCE_PAD, compress_silence   # pure; imports nothing back

K_RETRIEVE = 6   # candidates pulled per beat by embedding similarity (fallback path)
N_SCORE = 4      # of those, how many get the (costlier) local LLM relevance score

# Segments closer than this after snapping merge into one span (no stutter cuts).
#
# Was 0.75, which is shorter than any conversational turn: an NPC finishes a line, the
# streamer answers 2s later, and the director's two segments stayed two hard jump cuts
# with the reply deleted between them. 2.5s swallows a reply-sized hole and still leaves
# a deliberate skip (the 30s+ gaps that ARE the edit) untouched -- measured median
# intra-beat gap across existing runs is 34.7s, nowhere near this.
# ponytail: one flat threshold, no turn detection. A speaker-aware merge needs
# diarization, or at minimum an un-downmixed source (audio.py forces -ac 1 mono today).
SEG_MERGE_GAP = 2.5
# A beat is ONE moment. Segments further apart than this are two different moments the
# director stapled together -- measured across the existing runs as one "hook" splicing
# 349s to 2130s, and 13 beats (8% of them) doing something like it.
#
# 300 and not lower BECAUSE of the build-up machinery, which is the opposite failure and
# is deliberately sanctioned: `pacing.BUILD_GAP_S` fires at 60s and the MISSING BUILD-UP
# rubric asks the director to keep a representative RUN of a repetition -- a wordle spiral
# sampled across four minutes is one moment and must survive. Measured: a 180s cap would
# have hit 14% of beats and thrown 33s per run, much of it legitimate spirals; 300s hits
# 8% and ~15s. Above the build-up range, below anything defensible as one moment.
SEG_SPLIT_GAP = 300.0
WIDEN_SLOP = 15.0       # max s an editorial anchor may widen past the director's own bounds
WIDEN_SLOP_FINAL = 30.0  # the final beat gets extra room so the ending still breathes


def _one_moment(spans, payoff=None, title="") -> list:
    """Keep only the cluster of spans that is actually ONE moment.

    A beat is a slot in the story, not a folder: when the director returns spans minutes
    apart it has stapled two unrelated moments into one beat, and the cut plays them back
    to back as if they belonged together. Cluster on `SEG_SPLIT_GAP` and keep the cluster
    holding the payoff (the anchor that says where the point lands); with no payoff given,
    keep the one with the most footage. The rest are dropped, loudly -- silently shedding
    footage the director asked for is how you get an unexplained short cut.
    """
    groups = [[spans[0]]]
    for x, y in spans[1:]:
        if x - groups[-1][-1][1] > SEG_SPLIT_GAP:
            groups.append([])
        groups[-1].append((x, y))
    if len(groups) == 1:
        return spans
    if payoff is not None:
        best = max(groups, key=lambda g: (g[0][0] <= float(payoff) <= g[-1][1],
                                          sum(y - x for x, y in g)))
    else:
        best = max(groups, key=lambda g: sum(y - x for x, y in g))
    lost = sum(y - x for g in groups if g is not best for x, y in g)
    print(f"[cast] '{title}': dropped {len(groups) - 1} stray span group(s) "
          f"({int(lost)}s) more than {int(SEG_SPLIT_GAP)}s from the beat's own moment")
    return best


def _seam_log(spans, span_text) -> list[dict]:
    """What the edit DELETED between kept spans, so the critic can see its own jump cuts.

    The realized transcript handed to the critic is the kept spans joined with a space.
    Read alone that is a run-on paragraph in which a character's line runs straight into
    whatever came next -- the reply that was cut out leaves no trace, so the critic cannot
    flag a severed exchange even in principle. One entry per hole: when, how long, and what
    was said in it.
    """
    out = []
    for (_, end), (nxt, _) in zip(spans, spans[1:]):
        if nxt - end > 0:
            out.append({"at": end, "dur": nxt - end, "text": span_text(end, nxt)})
    return out

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


def _realized_shrink(picked: list[dict]) -> float | None:
    """Finished seconds per second of SELECTED span, over the whole cast.

    This is the correction the director needs and cannot compute: it picks spans off the
    stream map, and the dead air inside them never reaches the viewer. `None` when there
    is nothing to measure, so callers fall back to the constant prior rather than to a
    fabricated ratio.
    """
    raw = sum(y - x for p in picked for x, y in p["segments"])
    return round(sum(_dur(p) for p in picked) / raw, 4) if raw > 0 else None


def _kept_dur(words: list[dict], spans, deadspace: bool, pad: float, voiced=None) -> float:
    """The FINISHED length of `spans`: what the viewer sees, not what the spans cover.

    With `deadspace` on, the renderer strips each span's internal silence, so the honest
    length is what `compress_silence` reports -- the very function `ae_export`/`assemble`
    call later, reused here so the two can't disagree. With it off nothing is compressed
    downstream either, and the raw sum already IS the finished length.
    """
    if not deadspace:
        return sum(y - x for x, y in spans)
    # ponytail: compress_silence re-derives speech_intervals(words) per span -- ~50 calls
    # per cast, invisible beside the model passes. Hoist it if it ever shows in a profile.
    return sum(compress_silence(words, x, y, pad=pad, voiced=voiced)[2] for x, y in spans)


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

# A stack moment is a different object from a teaser: long enough to register WHAT KIND of
# moment it is, never long enough to explain it. Past ~2.5s it stops being a flash and
# starts being a scene, and the stack's density -- the whole point -- collapses.
STACK_MAX_S = 2.5
STACK_MIN_S = 1.0


def _cold_open_one(seg, picked: list[dict], words: list[dict], first_start: float,
                   stack: bool) -> dict | None:
    """One cold-open segment -> a clip, or None if it fails a guard.

    Guarded deterministically rather than trusted, because a promise the cut never
    delivers is worse than no promise: the span must overlap footage a real beat already
    contains, it is clamped to its own length ceiling, and it is dropped when the first
    beat is already right there (a stutter, not a promise).
    """
    from .select import snap_to_phrase, snap_to_sentences

    a, b = float(getattr(seg, "start_s", 0.0)), float(getattr(seg, "end_s", 0.0))
    if b <= a:
        return None
    hi, lo = (STACK_MAX_S, STACK_MIN_S) if stack else (COLD_OPEN_MAX_S, COLD_OPEN_MIN_S)
    b = min(b, a + hi)
    if not any(p["start"] <= a < p["end"] for p in picked):
        return None                        # promises footage the cut doesn't deliver
    if not stack and abs(a - first_start) < COLD_OPEN_NEAR_S:
        # A single teaser 20s ahead of itself is a stutter, not a promise. A STACK has no
        # such problem: it deliberately spoils moments from across the whole video, it is
        # ordered by energy rather than source time, and the thing that follows it is the
        # rest of the stack -- so this guard would only delete its opening moments.
        return None
    # Snap so it can't open or stop mid-word, but with a tight pad -- a sentence-length
    # pull would turn a flash back into a scene. The pad can carry it ~2s past the
    # ceiling; landing on a clean word beats holding the ceiling exactly. A stack moment
    # is shorter than most sentences, so it snaps to a PHRASE (a natural pause) instead --
    # a sentence snap there would inflate every moment past the ceiling by design.
    x, y = (snap_to_phrase(words, a, b, max_pad=0.5) if stack
            else snap_to_sentences(words, a, b, max_pad=2.0))
    if y - x < lo:
        return None
    return {"start": x, "end": y, "dur": y - x}


def _mismatch(a: dict, b: dict) -> int:
    """How wrong `b` feels straight after `a`. Higher = a bigger tonal jolt.

    Texture dominates: chaos into calm is the joke, and screaming into more screaming is
    not. Energy distance breaks ties, so the run alternates loud and quiet instead of
    merely alternating labels.
    """
    return (2 * (a.get("texture") != b.get("texture"))
            + abs(int(a.get("energy") or 3) - int(b.get("energy") or 3)))


def _whiplash(picked: list[dict]) -> list[dict]:
    """Order beats to MAXIMIZE the tonal jolt between neighbours, not to follow the clock.

    Chronological order is loyalty to the stream, and a dense clip reel gets its comedy
    from juxtaposition instead: the second clip doesn't have to be funnier, it has to be
    tonally wrong. Greedy -- open on the highest-energy beat, then repeatedly take
    whichever unused beat feels most wrong next. `pacing.sameness` already flags adjacent
    same role/texture, so the audit and this ordering want the same thing.
    """
    rest = list(picked)
    out = [rest.pop(rest.index(max(rest, key=lambda p: int(p.get("energy") or 3))))]
    while rest:
        nxt = max(rest, key=lambda p: _mismatch(out[-1], p))
        out.append(rest.pop(rest.index(nxt)))
    return out


def _snap(words, a: float, b: float, mode: str, truncate: float, closing: bool):
    """One kept span -> its real in/out, by cut-point mode.

    `sentence` (default) is the old behavior verbatim: a cut that opens mid-sentence is
    incomprehensible, so spans widen to whole sentences. That widening is also the real
    floor on shot length -- no budget scaling produces a 2s shot out of a 6s sentence --
    which is why the other two modes exist. `phrase` lands on natural pauses (still never
    mid-word); `transient` lands on the audio peak and leaves before it resolves.

    The FINAL span of the final beat keeps its sentence snap in every mode: the ending is
    the most visible cut in the video, and truncating it is an abrupt stop, not a style.
    """
    from .select import snap_to_phrase, snap_to_sentences, snap_to_transient

    if closing or mode == "sentence":
        return snap_to_sentences(words, a, b, max_pad=8.0 if closing else 5.0,
                                 phrase_fallback=closing)
    if mode == "transient":
        return snap_to_transient(words, a, b, truncate=truncate)
    return snap_to_phrase(words, a, b)


def _beat_energy(picked: list[dict], t: float) -> int:
    """Energy of the beat this moment was lifted from -- the stack's escalation key."""
    for p in picked:
        if p["start"] <= t < p["end"]:
            return int(p.get("energy") or 3)
    return 3


def _cold_opens(outline, picked: list[dict], words: list[dict],
                stack: int = 0) -> list[dict]:
    """What plays before beat 1: one flash-forward teaser, or a montage stack of `stack`.

    This is the ONE place the cut is allowed out of chronological order. As a teaser it is
    the lever against a middle that loses people -- a viewer who has seen where this is
    going sits through the setup to get there. As a STACK it is the opposite bet: several
    unexplained moments in a row, each too short to process, so the 5-15s window where
    clip videos hemorrhage viewers becomes the densest part of the video instead of the
    setup for it. Stack moments are ordered weakest-first so the run escalates and the
    single best moment lands last.
    """
    segs = getattr(outline, "cold_open", None) or []
    if isinstance(segs, dict) or not isinstance(segs, (list, tuple)):
        segs = [segs]                      # a bare Segment (older outline / model reply)
    if not segs or not picked:
        return []
    first_start = min(p["start"] for p in picked)
    if stack <= 0:
        one = _cold_open_one(segs[0], picked, words, first_start, stack=False)
        return [one] if one else []
    out = []
    for seg in segs[:stack]:
        clip = _cold_open_one(seg, picked, words, first_start, stack=True)
        if clip:
            out.append(clip)
    # escalate: weakest moment first, the best one LAST. Ties keep the director's order,
    # which is the order it was told to escalate in anyway. The energy travels into the
    # log so `pacing._stack_flag` can measure whether the stack actually escalated.
    for c in out:
        c["energy"] = _beat_energy(picked, c["start"])
    out.sort(key=lambda c: c["energy"])
    return out


def _valid_bounds(s0, e0, stream_end: float) -> bool:
    """The director gave usable in/out timestamps for this beat (the primary path)."""
    if s0 is None or e0 is None:
        return False
    if e0 <= s0 or s0 < 0:
        return False
    return stream_end == 0.0 or s0 < stream_end


def cast(outline, chunks, words: list[dict], target_s: float,
         model: str, tol: float = 0.35, progress=None,
         deadspace: bool = True, silence_pad: float = SILENCE_PAD,
         voiced=None,   # transcribe.speech_regions -- the renderer must get the same list
         # the style's machinery half (see refire/styles.py); all identity-defaulted, so
         # an unstyled cast is byte-identical to the one this module has always produced
         order: str = "chrono", snap: str = "sentence", stack: int = 0,
         truncate: float = 0.3):
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
            if reaction is not None and snap != "transient":
                # The reaction tail is the opposite of a premature cut: transient mode
                # exists to leave DURING the moment, so honouring it there would undo the
                # only thing that mode does.
                out = max(out, float(reaction))       # keep the reaction tail
            segs[-1] = (la, min(out, lb + slop))
            # snap each kept span; the final beat's LAST span is the most visible ending
            # -> be generous + fall back to a clean breath when whisper dropped the
            # closing punctuation.
            snapped = []
            for j, (sa, sb) in enumerate(segs):
                closing = is_final and j == len(segs) - 1
                x, y = _snap(words, sa, sb, snap, truncate, closing)
                snapped.append([x, y])
            spans = [snapped[0]]
            for x, y in snapped[1:]:                  # merge overlaps/near-touches
                if x <= spans[-1][1] + SEG_MERGE_GAP:
                    spans[-1][1] = max(spans[-1][1], y)
                else:
                    spans.append([x, y])
            spans = _one_moment(spans, getattr(beat, "payoff_start_s", None), beat.title)
            a, b = spans[0][0], spans[-1][1]          # envelope (ordering + display)
            parts = [span_text(x, y) for x, y in spans]
            text = " ".join(t for t in parts if t)
            seams = _seam_log(spans, span_text)
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
            parts, seams = [text], []  # one span, so nothing was cut out of the middle
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
                       # FINISHED length: the same silence compression `ae_export`/
                       # `assemble` apply later, so the pacing audit, the critic and
                       # `_trim_to_budget` all describe the video that actually renders.
                       # Measuring the raw spans here made every beat read up to 46%
                       # longer than it ships (a 16:33 cut audited as 21:28), so the
                       # audit flagged beats that were never long and the trimmer shed
                       # beats it had room for.
                       "dur": _kept_dur(words, spans, deadspace, silence_pad, voiced),
                       "score": sc, "reason": reason,
                       "text": text,    # realized KEPT transcript -> editor-review
                       # ...and what the edit threw away between those spans. `text` alone
                       # reads as one paragraph, so a severed exchange is invisible in it.
                       "parts": parts, "seams": seams})

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

    # Play order. `chrono` (default) emits in stream order so the cut plays forward in
    # time -- for a linear stream that is also the natural story order, and it guarantees
    # output is never reverse/scrambled. `director` trusts the outline's own sequencing;
    # `whiplash` throws the clock away entirely and orders for tonal jolt.
    if order == "chrono":
        picked.sort(key=lambda p: p["start"])
    elif order == "whiplash":
        picked = _whiplash(picked)
    # role/energy travel with the clip so the style pass (ae_export/overlay/render) can
    # let the story drive presentation: hook punches in, button stays out of the laugh.
    # One clip PER KEPT SPAN -- multiple clips in a section render as jump cuts within
    # the beat (build_manifest/render_clips already handle multi-clip sections).
    sections = [{"title": p["title"], "role": p["role"], "energy": p["energy"],
                 "clips": [{"start": a, "end": b,
                            "role": p["role"], "energy": p["energy"]}
                           for a, b in p["segments"]]}
                for p in picked]
    # The cold open goes in AFTER the ordering pass (it is the one thing allowed to play
    # out of stream order) and after the budget trim, so a promise can never be the thing
    # that gets shed. Roled `hook` so the style pass gives it no section card. A stack is
    # several clips in the SAME section -- build_manifest already renders a multi-clip
    # section as jump cuts, which is exactly what a montage stack is.
    # ponytail: its seconds aren't subtracted from the budget -- under 1% of a 15-minute
    # target even as an 8-moment stack. Count it if stacks ever get long enough to matter.
    cold = _cold_opens(outline, picked, words, stack)
    if cold:
        sections.insert(0, {"title": "Cold Open", "role": "hook", "energy": 5,
                            "clips": [{"start": c["start"], "end": c["end"],
                                       "role": "hook", "energy": 5} for c in cold]})
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
        # finished seconds per second of selected span, measured on THIS stream. The
        # director budgets in spans and can't see the compression, so it is told to
        # select target/shrink -- with the constant prior on round 0 and this number,
        # which is the truth for this footage, on every review round after.
        "shrink": _realized_shrink(picked),
    }
    return sections, outline_log, warning


def _mmss(s) -> str:
    s = int(s or 0)
    return f"{s // 60:02d}:{s % 60:02d}"


def cut_plan_md(outline_log: dict, pace: float = 1.0,
                keep_build: bool = True, target_s: float = 0.0) -> str:
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
    cold = outline_log.get("cold_open") or []
    if isinstance(cold, dict):                     # an outline from before it was a list
        cold = [cold]
    if len(cold) == 1:
        co = cold[0]
        out += [f"**Cold open:** {_mmss(co['start'])}-{_mmss(co['end'])} "
                f"({co['dur']:.1f}s flash-forward)", ""]
    elif cold:
        spans = ", ".join(f"{_mmss(c['start'])} ({c['dur']:.1f}s)" for c in cold)
        out += [f"**Cold open:** {len(cold)}-moment stack, "
                f"{sum(c['dur'] for c in cold):.1f}s total -- {spans}", ""]

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
    out.append(audit_note(outline_log, pace, keep_build, target_s))
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
