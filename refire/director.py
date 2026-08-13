"""The director pass: read the whole stream, write a story outline (Claude).

One Claude call turns a time-stamped map of everything said in the stream into a
central idea plus an ordered list of BEATS -- story slots with a purpose ("the cocky
setup", "first crack", "the kill", "the reaction"). `narrative.cast` then fills each
beat with the single best real clip. This is the comprehension step that the local
embed/score pipeline can't do: it judges the stream as a story, not clips in isolation.

Runs on Claude via ANTHROPIC_API_KEY; raises if the key/SDK is missing so the
caller (`pipeline.make`) can fall back to the flat retrieval path. Embeddings and
per-clip scoring stay local -- only this whole-stream reasoning call goes to the cloud.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, model_validator

CLAUDE_MODEL = "claude-sonnet-5"
MAX_TOKENS = 32000   # adaptive thinking + the largest outline JSON both fit comfortably


class _JsonModel(BaseModel):
    """Base for the director's JSON payloads. In JSON mode Claude/Ollama emit `null` for
    optional fields; drop them so pydantic applies our defaults instead of rejecting None on
    a str-typed field (the strict output_format endpoint used to coerce these for us)."""

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data):
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data


class Segment(_JsonModel):
    """One kept sub-span of a beat: an exact line (or run of lines) the final cut contains.
    Multiple segments per beat = jump cuts within the moment -- the human-editor move that
    drops rambling between the lines that matter."""
    start_s: float
    end_s: float


class Beat(_JsonModel):
    """A story slot, tagged like an editor's notebook entry (not "an interesting clip").

    Everything past `title`/`start_s`/`end_s` is defaulted so older outlines and the test
    stubs still parse, and so the local director can omit fields it can't fill. `role` casts
    the beat into the arc; setup/payoff/reaction give casting the tight-cut anchors.
    """
    title: str                      # short on-screen section-card title
    role: str = ""                  # hook|setup|escalation|reversal|climax|payoff|button
    intent: str = ""                # this beat's job in the story (also the scorer's mini-brief)
    viewer_question: str = ""       # what the viewer is wondering after this beat
    turn: str = ""                  # what changes in this beat
    transition_in: str = ""         # why this beat follows the previous one
    texture: str = ""               # funny|tense|awkward|triumphant|chaotic|sincere
    energy: int = 3                 # 1-5 pacing value (lets the critic flag monotony)
    setup_start_s: float | None = None   # where needed context begins, if before start_s
    payoff_start_s: float | None = None  # where the joke/reveal/win/fail lands (never cut before)
    reaction_end_s: float | None = None  # tail after the payoff, so the cut doesn't feel abrupt
    start_s: float                  # clip in-point (seconds) -- where this beat's moment begins
    end_s: float                    # clip out-point (seconds) -- right after its payoff lands
    segments: list[Segment] = []    # the EDIT: exact kept lines within the beat (jump cuts)
    query: str = ""                 # fallback retrieval text if start_s/end_s come out unusable


class Outline(_JsonModel):
    central_idea: str
    story_shape: str = ""           # the ONE shape this VOD supports (e.g. "confidence -> chaos")
    viewer_promise: str = ""        # what the viewer is promised they're watching
    ending_needed: str = ""         # what kind of ending the story has to land
    beats: list[Beat]


class Notable(_JsonModel):
    """One moment in a chapter worth an editor's attention, with its map timestamp."""
    t_s: float
    why: str = ""


class Chapter(_JsonModel):
    """A scout-pass chapter: what one ~20-minute stretch of the stream is about.

    The story pass reads the chapter guide instead of (or alongside) the raw map, which is
    how a 7-hour VOD fits the director's context: comprehension is built coarse-to-fine.
    """
    title: str
    start_s: float
    end_s: float
    summary: str = ""
    notable_moments: list[Notable] = []


class _ChapterOut(_JsonModel):
    """The scout model's per-window reply (bounds come from the window, not the model)."""
    title: str = ""
    summary: str = ""
    notable_moments: list[Notable] = []


class Review(_JsonModel):
    approved: bool   # true = the rough cut already tells a cohesive story, ship it
    notes: str       # what's wrong (or why it's approved) -- logged to outline.json
    outline: Outline # the revised plan (full re-plan: reorder/drop/merge/add/rebound)


# The editorial Outline/Review schema is too deep for Claude's strict `output_format`
# structured-output endpoint (it 400s "Schema is too complex"), so we prompt for JSON and
# validate it ourselves -- same approach the local Ollama director already uses. One shared
# shape string keeps the Claude + local directors in lockstep.
_OUTLINE_JSON = (
    '{"central_idea": "<str>", "story_shape": "<str>", "viewer_promise": "<str>", '
    '"ending_needed": "<str>", "beats": [{"title": "<str>", "role": '
    '"<hook|setup|escalation|reversal|climax|payoff|button>", "intent": "<str>", '
    '"viewer_question": "<str>", "turn": "<str>", "transition_in": "<str>", '
    '"texture": "<str>", "energy": <1-5>, "setup_start_s": <number|null>, '
    '"payoff_start_s": <number|null>, "reaction_end_s": <number|null>, '
    '"start_s": <number>, "end_s": <number>, '
    '"segments": [{"start_s": <number>, "end_s": <number>}, ...], '
    '"query": "<str>"}]}'
)
_OUTLINE_INSTR = (
    " Reply ONLY with a single JSON object (no markdown, no prose) of this shape: "
    + _OUTLINE_JSON)
_REVIEW_INSTR = (
    ' Reply ONLY with a single JSON object (no markdown, no prose) of this shape: '
    '{"approved": <true|false>, "notes": "<str>", "outline": ' + _OUTLINE_JSON + "}")


_SYSTEM = (
    "You are a video editor cutting a Twitch gaming stream into one focused, cohesive "
    "short. You are given the stream's title, the editor's brief (the subject and vibe "
    "to capture), and a time-stamped map of everything said. Each map line begins with "
    "its start time IN SECONDS, like `[546s]` (that is 546 seconds, i.e. 9m06s, into the "
    "stream). Streams run for hours, so these values get large. Do NOT list interesting "
    "moments -- find the STORY this VOD can actually support.\n"
    "First decide three things about the whole cut:\n"
    "- story_shape: the single shape this footage supports, e.g. 'confidence collapses "
    "into chaos', 'a bad idea slowly becomes genius', 'chat doubts him, then he proves "
    "it', 'one tiny mistake becomes the whole episode', 'a running joke evolves into the "
    "payoff'.\n"
    "- viewer_promise: what the viewer is promised they're watching (the through-line).\n"
    "- ending_needed: the kind of moment the story has to end on to feel complete.\n"
    "Then break it into BEATS in story order -- which for a mostly-linear gaming stream "
    "is chronological. A beat is a slot in the story with a job, not an interesting clip. "
    "Each beat needs:\n"
    "- title: short, shown on-screen as a section card.\n"
    "- role: ONE of hook, setup, escalation, reversal, climax, payoff, button -- its "
    "place in the arc. Every selected beat must earn one of these roles.\n"
    "- intent: its job in the story (e.g. 'establish he's overconfident before it goes "
    "wrong'). This is also the local scorer's mini-brief.\n"
    "- viewer_question: what the viewer is wondering after this beat (sets up the next).\n"
    "- turn: what changes in this beat.\n"
    "- transition_in: why this beat follows the previous one (the connective tissue).\n"
    "- texture: funny, tense, awkward, triumphant, chaotic, or sincere.\n"
    "- energy: 1-5 pacing value; vary it across beats so the cut isn't monotone.\n"
    "- start_s and end_s: the exact in/out timestamps IN SECONDS, copied from the `[Ns]` "
    "labels in the stream map (e.g. a moment on the `[546s]` line has start_s near 546, "
    "NOT 9.06). Set start_s where the moment begins and end_s RIGHT AFTER its payoff or "
    "punchline lands, on a natural pause.\n"
    "- segments: the EDIT. A beat is edited footage, not a raw span: within start_s..end_s, "
    "list ONLY the lines the final video should contain, as tight sub-spans copied from the "
    "map's `[Ns]` labels. Jump-cut the rambling, tangents, and off-topic filler between "
    "them -- jump cuts within a moment are natural and expected. Each segment should be at "
    "least ~3 seconds; a beat's kept footage should usually total 15-60 seconds and NEVER "
    "exceed ~90 -- if the moment sprawls, keep only its best lines as separate segments. "
    "A single segment equal to the whole span means 'keep everything'.\n"
    "- setup_start_s: if the context the joke needs starts earlier than start_s, put it "
    "here (else omit).\n"
    "- payoff_start_s: where the joke/reveal/win/fail actually lands. NEVER cut before "
    "this.\n"
    "- reaction_end_s: a short tail after the payoff (the laugh/reaction) so the cut "
    "doesn't feel abrupt (else omit).\n"
    "- query: a few words to find the footage as a fallback if the timestamps fail.\n"
    "Shape the beats into an ARC -- open on a hook, set up, escalate, hit a climax, and "
    "land the ending_needed as a closing button -- NOT a flat list of equally-strong "
    "moments (that is a highlight reel, not a story). Give the ENDING as much deliberate "
    "construction as the hook: the closing button must BREATHE, not stop dead on the "
    "payoff. Set its end_s AFTER the reaction/aftermath (the laugh, the sigh, the 'what "
    "just happened' beat), and always give the final beat a reaction_end_s so the cut "
    "winds down instead of cutting off mid-breath. A strong open and a soft, abrupt close "
    "wastes the whole story. Make each beat follow from the one "
    "before via its transition_in, and never repeat the same kind of moment twice. "
    "Only include beats the footage can actually support. "
    "TARGET RUNTIME IS A ROUGH GUIDE, NOT A HARD LIMIT: aim for about {n} beats whose "
    "in/out spans TOTAL somewhere in the neighborhood of the target, but let the story's "
    "own shape decide the real length. Never stretch a beat with slow, padded, or repeated "
    "footage just to fill time, and never gut a beat's setup or cut off its payoff just to "
    "shave seconds -- a complete, well-paced story that lands moderately over or under "
    "target beats a mangled one that hits the number exactly."
)


_MAP_LEGEND = (
    "\nThe map may carry extra perception signals -- use them as an editor uses their "
    "eyes and ears, as evidence of where the real moments are: `(loud)`/`(LOUD)` = the "
    "streamer's audio spiked (raised voice / outright yelling), and standalone "
    "`[scene: ...]` lines are APPROXIMATE machine descriptions of what was on screen at "
    "that moment -- treat them as weak hints of the visual context, never as exact facts."
)


_SYSTEM_STREAM_INTRO = (
    "You are a video editor cutting a Twitch gaming stream into one focused, cohesive "
    "short. NO editorial brief was given: your job is to find the ONE story this stream "
    "actually tells and cut that. Read the whole time-stamped map -- what was said, where "
    "chat exploded, where the streamer got loud, what was on screen -- and mine the "
    "stream's own best arc: its running joke, its slow-burn disaster, its comeback, "
    "whatever the footage genuinely supports. Do not average across everything fun; "
    "commit to the strongest single through-line. "
)


def pick_system(brief: str | None) -> str:
    """The outline system prompt for this mode: brief-driven (default) or stream-driven
    (no brief -> mine the stream's own best story). Pure, so mode selection is testable."""
    if brief:
        return _SYSTEM + _MAP_LEGEND
    # same beat schema + arc rules; only the opening framing changes
    intro_end = _SYSTEM.index("Do NOT list interesting")
    return _SYSTEM_STREAM_INTRO + _SYSTEM[intro_end:] + _MAP_LEGEND


_REVIEW_SYSTEM = (
    "You are a senior video editor reviewing a ROUGH CUT of a Twitch gaming short for "
    "STORY, not individual moments. You are given the cut's central idea, the editor's "
    "brief, the realized cut (what each beat ACTUALLY says, in order, with its intended "
    "role/intent/viewer_question), and the full stream map. Judge it as the VIEWER will "
    "experience it:\n"
    "- HOOK: do the first ~10 seconds grab attention?\n"
    "- ARC: does it build (hook -> setup -> escalation -> climax -> resolution/button), "
    "or is it a flat list of equally-good moments (a highlight reel)?\n"
    "- SETUP CLARITY: does any clip start without enough context for the viewer to "
    "understand the moment? Pull its start earlier (setup_start_s) if so.\n"
    "- EXPECTATION: after each beat, the viewer expects something. Does the NEXT beat "
    "answer, twist, or escalate that expectation, or does it ignore it (a jarring jump)?\n"
    "- CONNECTIVE TISSUE: does each beat follow from the one before (transition_in)?\n"
    "- REDUNDANCY: do two beats make the same point or land the same kind of joke? Cut one.\n"
    "- DEAD WEIGHT: read each beat's realized transcript. If it contains rambling, "
    "tangents, or off-topic filler (a low score/reason often says so), re-cut the beat "
    "with `segments` keeping ONLY the lines that serve its intent -- jump cuts within a "
    "moment are natural. Flag any beat whose kept footage exceeds ~90 seconds.\n"
    "- PAYOFF COMPLETION: does any clip end BEFORE its payoff or reaction lands "
    "(cut off mid-joke)? Extend its end / payoff_start_s / reaction_end_s if so.\n"
    "- ENDING: give the last beats the SAME scrutiny as the hook. Does the central idea "
    "land on a strong closing button, and does that button BREATHE -- holding on the "
    "reaction/aftermath -- rather than cutting off the instant the payoff lands (an abrupt "
    "ending)? If it stops dead, extend the final beat's end_s / reaction_end_s or add a "
    "short wind-down button beat.\n"
    "- PACING: does the energy vary, or is every beat the same intensity (monotony)?\n"
    "If the cut already tells a cohesive story, set approved=true and briefly say why in "
    "notes. Otherwise set approved=false, name the main problems in notes, and return a "
    "REVISED outline: reorder, drop, merge, or ADD connective beats, re-tag roles, and "
    "retighten each beat's in/out (start_s/end_s from the stream map, ending right after "
    "the payoff + reaction lands). Only include beats the footage supports. The target "
    "runtime (about {n} beats) is a rough guide, not a hard limit -- only add/drop/trim "
    "beats for the STORY reasons above; don't chase the exact number by padding a thin cut "
    "or gutting a beat's setup/payoff to shave seconds."
)


def stream_map(words: list[dict]) -> str:
    """Render the transcript as a compact `[<seconds>s] sentence` map for the director.

    Sentence-level (break on .?!) so the director can choose precise in/out timestamps;
    each line's timestamp is that sentence's first word, in SECONDS. Seconds (not mm:ss) so
    the director copies them straight into start_s/end_s -- a `[09:06]` colon form gets
    misread as 9.06s and collapses a multi-hour stream's cut into its first minute.
    Enriched-map runs go through `perception.moment_map` instead (same splitter, plus
    chat/loudness marks and scene lines); this stays as the bare-transcript form.
    """
    from .perception import _sentences
    return "\n".join(f"[{a}s] {text}" for a, _b, text in _sentences(words))


_SCOUT_SYSTEM = (
    "You are logging a Twitch gaming stream for a video editor. You are given one window "
    "of a time-stamped map of the stream (each line starts with `[Ns]` = seconds into the "
    "stream; `(loud)`/`(LOUD)` marks and `[scene: ...]` lines are perception signals). "
    "Reply ONLY with a single JSON object (no markdown, no prose) of this shape: "
    '{"title": "<3-6 word chapter title>", "summary": "<2-4 sentences: what happens, '
    'what the streamer is trying to do, how it goes>", "notable_moments": '
    '[{"t_s": <number, copied from a [Ns] label>, "why": "<one line: why an editor '
    'would care>"}, ...]}. List at most 5 notable moments; strong reactions, fails, '
    "wins, running jokes, and chat/loudness spikes are what count."
)


def _parse_stamp(line: str) -> float | None:
    """The leading `[Ns]` timestamp of a map line, or None."""
    if line.startswith("[") and "s]" in line[:12]:
        try:
            return float(line[1:line.index("s]")])
        except ValueError:
            return None
    return None


def _windows(map_text: str, window_s: float = 1200.0) -> list[tuple[float, float, str]]:
    """Split a stream map into (t0, t1, text) windows of ~window_s by line timestamps.

    Pure. Lines without a parseable stamp ride with the current window. Empty map -> [].
    """
    out: list[tuple[float, float, str]] = []
    cur: list[str] = []
    w0 = 0.0
    last = 0.0
    for line in map_text.splitlines():
        t = _parse_stamp(line)
        if t is not None:
            if cur and t >= w0 + window_s:
                out.append((w0, t, "\n".join(cur)))
                cur = []
                w0 = t
            last = t
        cur.append(line)
    if cur:
        out.append((w0, max(last, w0), "\n".join(cur)))
    return out


def chapterize(map_text: str, window_s: float = 1200.0, model: str = "llama3.1:8b",
               cache_path=None, progress=None) -> list[Chapter]:
    """Scout pass: one local-Ollama call per ~20-min window -> chapter guide.

    Free (no Claude), so it always runs before the story pass. A failed window degrades to
    a stub chapter built from its first line (never crashes the run). `cache_path` =
    run/<vod>/chapters.json, skip-if-exists (keyed to nothing else -- delete to re-scout).
    """
    from pathlib import Path
    if cache_path and Path(cache_path).exists():
        try:
            data = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            return [Chapter.model_validate(c) for c in data]
        except (ValueError, TypeError):
            pass   # corrupt cache -> re-scout
    import ollama
    wins = _windows(map_text, window_s)
    chapters: list[Chapter] = []
    for i, (a, b, text) in enumerate(wins):
        if progress:
            progress((i + 1) / max(1, len(wins)), f"scouting chapter {i + 1}/{len(wins)}")
        try:
            resp = ollama.chat(
                model=model, format="json",
                messages=[{"role": "system", "content": _SCOUT_SYSTEM},
                          {"role": "user", "content": text}],
                options={"num_ctx": 16384})
            got = _ChapterOut.model_validate(json.loads(resp["message"]["content"]))
        except Exception as e:   # any window failure -> stub, keep scouting
            print(f"[scout] window {i + 1} failed ({e}); using stub chapter")
            got = _ChapterOut()
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        chapters.append(Chapter(
            title=got.title or first[:60] or f"chapter {i + 1}",
            start_s=a, end_s=b, summary=got.summary,
            # clamp hallucinated stamps into the window
            notable_moments=[n for n in got.notable_moments if a <= n.t_s <= b][:5]))
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(
            json.dumps([c.model_dump() for c in chapters], indent=2), encoding="utf-8")
    return chapters


def _realized_script(outline_log: dict) -> str:
    """Render the CAST cut as the script the critic reads: central idea + story shape +
    each beat in order with its editorial intent (role/intent/viewer_question/transition)
    AND the ACTUAL transcript of the chosen span.

    This is the closed-loop signal -- the critic judges what the video really says against
    what each beat was meant to do, not the director's original plan. Pure (no model) so
    it's unit-testable.
    """
    lines = [f"CENTRAL IDEA: {outline_log.get('central_idea', '')}"]
    if outline_log.get("story_shape"):
        lines.append(f"STORY SHAPE: {outline_log['story_shape']}")
    if outline_log.get("viewer_promise"):
        lines.append(f"VIEWER PROMISE: {outline_log['viewer_promise']}")
    lines.append("")
    for i, b in enumerate(outline_log.get("beats", []), 1):
        a, z = b.get("start"), b.get("end")
        segs = b.get("segments") or []
        cut = (f", {len(segs)} cuts, {int(b.get('dur') or 0)}s kept" if len(segs) > 1 else "")
        ts = (f"[{int(a) // 60:02d}:{int(a) % 60:02d}-"
              f"{int(z) // 60:02d}:{int(z) % 60:02d}{cut}]"
              if a is not None and z is not None else "[--]")
        role = b.get("role") or "?"
        lines.append(f"{i}. {b.get('title', '')} [{role}] "
                     f"(energy {b.get('energy', '?')}, score {b.get('score', '?')})")
        lines.append(f"   intent: {b.get('intent', '')}")
        if b.get("transition_in"):
            lines.append(f"   transition_in: {b['transition_in']}")
        if b.get("viewer_question"):
            lines.append(f"   leaves viewer wondering: {b['viewer_question']}")
        lines.append(f"   {ts} {str(b.get('text', '')).strip()}")
        lines.append("")
    dropped = outline_log.get("dropped") or []
    if dropped:
        # tell the critic WHY beats vanished, or it re-adds them at full length every
        # round and the budget trimmer deletes them again (an invisible tug-of-war).
        lines.append("NOTE: these beats were dropped to fit the runtime budget: "
                     + ", ".join(str(d.get("title", "?")) for d in dropped)
                     + ". Tighten other beats with segments instead of re-adding these "
                       "at full length.")
    return "\n".join(lines).strip()


def _n_beats(target_s: float) -> int:
    return max(2, round(target_s / 45.0))   # ~45s of finished footage per beat


def _extract_json(txt: str) -> str:
    """Slice the JSON object out of a model reply (tolerates prose / ``` fences)."""
    a, b = txt.find("{"), txt.rfind("}")
    if a == -1 or b <= a:
        raise ValueError(f"no JSON object in reply: {txt[:200]!r}")
    return txt[a:b + 1]


def _trace_write(trace, name: str, text: str) -> None:
    """Full-fidelity request/response dumps -- the console shows the live stream, these
    files hold EVERYTHING (the whole stream map is too big to scroll a terminal)."""
    from pathlib import Path
    d = Path(trace)
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text, encoding="utf-8")


def _complete(client, model, system, user_content, model_cls, tag="claude", trace=None):
    """One JSON-mode Claude call (streamed live to the console) with a retry guard.

    Shared by the director and the review pass. The system prompt already asks for a single
    JSON object (the strict structured-output endpoint 400s on this schema's depth -- even
    more so now that beats carry `segments`), so we take the text block and validate it into
    `model_cls`. Adaptive thinking self-budgets; `MAX_TOKENS` leaves the answer plenty of
    room. Streaming because the SDK refuses non-streaming calls with max_tokens this large
    -- and it lets the terminal watch the model think and write the outline in real time
    (`display: "summarized"`; the raw chain of thought is never returned). `trace` = a dir
    that receives the full request/response text per call. If thinking somehow ate the
    whole budget so no text block was emitted, retry once with thinking off; still empty ->
    raise so the caller falls back instead of silently degrading.
    """
    import time

    user_text = "\n\n".join(b.get("text", "") for b in user_content)
    stamp = time.strftime("%H%M%S")
    print(f"[{tag}] -> {model}: system {len(system):,} chars, "
          f"user {len(user_text):,} chars (~{len(user_text) // 4000}k tok)")
    if trace:
        _trace_write(trace, f"{stamp}-{tag}-request.txt",
                     f"MODEL: {model}\n\n=== SYSTEM ===\n{system}\n\n"
                     f"=== USER ===\n{user_text}")
        print(f"[{tag}] full prompt -> {trace}\\{stamp}-{tag}-request.txt")

    def _call(thinking):
        kw = dict(model=model, system=system, max_tokens=MAX_TOKENS,
                  thinking=({"type": "adaptive", "display": "summarized"} if thinking
                            else {"type": "disabled"}),
                  messages=[{"role": "user", "content": user_content}])
        phase = ""
        thought: list[str] = []
        with client.messages.stream(**kw) as s:
            for ev in s:
                if getattr(ev, "type", "") != "content_block_delta":
                    continue
                d = ev.delta
                kind = getattr(d, "type", "")
                if kind == "thinking_delta" and getattr(d, "thinking", ""):
                    chunk = d.thinking
                    thought.append(chunk)
                elif kind == "text_delta" and getattr(d, "text", ""):
                    chunk = d.text
                else:
                    continue
                label = "thinking" if kind == "thinking_delta" else "writing"
                if phase != label:
                    print(f"\n[{tag}] --- {label} ---")
                    phase = label
                print(chunk, end="", flush=True)
            resp = s.get_final_message()
        if phase:
            print(flush=True)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return text, resp.stop_reason, "".join(thought)

    text, stop, thought = _call(thinking=True)
    if not text.strip():
        print(f"[{tag}] empty answer (stop_reason={stop}); retrying without thinking")
        text, stop, thought = _call(thinking=False)
    print(f"[{tag}] <- stop_reason={stop}, {len(text):,} chars")
    if trace:
        _trace_write(trace, f"{stamp}-{tag}-response.txt",
                     f"STOP_REASON: {stop}\n\n=== THINKING (summarized) ===\n{thought}\n\n"
                     f"=== ANSWER ===\n{text}")
    if not text.strip():
        raise RuntimeError(f"no answer text (stop_reason={stop})")
    return model_cls.model_validate(json.loads(_extract_json(text)))


def _complete_cli(model, system, user_content, model_cls, tag="claude", trace=None,
                  tools: str = "", cwd=None, timeout_s: float = 1200.0):
    """Same contract as `_complete`, but through Claude Code headless mode (`claude -p`)
    instead of the SDK -- the call bills the user's Claude SUBSCRIPTION, not the pay-as-you-
    go API key (~$1.2/video on Opus otherwise). Raises RuntimeError when the CLI is missing
    or fails, so the caller falls through to the API path.

    Wire notes (verified live): user text goes via stdin (the ~180KB stream map blows past
    Windows' arg limit), `--tools "" --setting-sources ""` make it a pure completion (no
    tools, and no user-level hooks injecting their context into every call), cwd is a temp
    dir so Claude Code doesn't walk up into a repo and load its CLAUDE.md persona, and
    ANTHROPIC_API_KEY is stripped from the env or the CLI would bill the key -- defeating
    the point. stream-json deltas mirror the SDK's, so the live console view is identical.
    ponytail: subscription 5h-window rate limits are the ceiling (3 Opus calls x ~45k tok
    per video); --review-rounds 1 or --director-backend api are the pressure valves.
    """
    import os
    import shutil
    import subprocess
    import tempfile
    import time

    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("claude CLI not on PATH")
    user_text = "\n\n".join(b.get("text", "") for b in user_content)
    stamp = time.strftime("%H%M%S")
    print(f"[{tag}] -> {model} via claude CLI (subscription): system {len(system):,} chars, "
          f"user {len(user_text):,} chars (~{len(user_text) // 4000}k tok)")
    if trace:
        _trace_write(trace, f"{stamp}-{tag}-request.txt",
                     f"MODEL: {model} (claude CLI)\n\n=== SYSTEM ===\n{system}\n\n"
                     f"=== USER ===\n{user_text}")
        print(f"[{tag}] full prompt -> {trace}\\{stamp}-{tag}-request.txt")

    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    # The system prompt goes through a temp FILE (--system-prompt-file), never --system-prompt:
    # a multi-line value in argv gets split on the newline by Windows command-line parsing,
    # which silently corrupts every flag after it -- stream-json is dropped and the CLI
    # answers in plain text, so our JSON parser sees nothing (0 chars, stop_reason=None).
    # ponytail: temp file leaks only if the call crashes before the unlink below; the OS
    # temp dir mops that up.
    sp = tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False)
    sp.write(system)
    sp.close()
    # tools="" = pure completion (default). tools="Read" = drill-down mode: the model may
    # Read per-chapter map files, relative to cwd (the run dir); --allowedTools pre-permits
    # them so headless mode never hangs on a prompt
    # (verified live: -p --tools Read --allowedTools Read reads relative paths fine).
    cmd = [exe, "-p", "--model", model, "--tools", tools, "--setting-sources", "",
           "--system-prompt-file", sp.name, "--output-format", "stream-json",
           "--include-partial-messages", "--verbose"]
    if tools:
        cmd += ["--allowedTools", tools]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # non-JSON stderr lines are skipped by the parser
        text=True, encoding="utf-8", errors="replace",
        env=env, cwd=str(cwd) if cwd else tempfile.gettempdir())
    # belt-and-braces: a wedged CLI (auth prompt, network stall) must not hang the run
    import threading
    killer = threading.Timer(timeout_s, lambda: proc.kill())
    killer.start()
    proc.stdin.write(user_text)
    proc.stdin.close()

    phase = ""
    thought: list[str] = []
    text, stop, is_err = "", None, False
    for line in proc.stdout:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "stream_event":
            d = (ev.get("event") or {}).get("delta") or {}
            kind = d.get("type", "")
            if kind == "thinking_delta" and d.get("thinking"):
                chunk = d["thinking"]
                thought.append(chunk)
            elif kind == "text_delta" and d.get("text"):
                chunk = d["text"]
            else:
                continue
            label = "thinking" if kind == "thinking_delta" else "writing"
            if phase != label:
                print(f"\n[{tag}] --- {label} ---")
                phase = label
            print(chunk, end="", flush=True)
        elif ev.get("type") == "result":
            text = ev.get("result") or ""
            stop = ev.get("stop_reason") or ev.get("subtype")
            is_err = bool(ev.get("is_error"))
    proc.wait()
    killer.cancel()
    os.unlink(sp.name)
    if phase:
        print(flush=True)
    print(f"[{tag}] <- stop_reason={stop}, {len(text):,} chars")
    if trace:
        _trace_write(trace, f"{stamp}-{tag}-response.txt",
                     f"STOP_REASON: {stop}\n\n=== THINKING (summarized) ===\n"
                     f"{''.join(thought)}\n\n=== ANSWER ===\n{text}")
    if proc.returncode != 0 or is_err:
        raise RuntimeError(f"claude CLI failed (exit {proc.returncode}, "
                           f"stop_reason={stop}): {text[:200]}")
    if not text.strip():
        raise RuntimeError(f"claude CLI gave no answer text (stop_reason={stop})")
    return model_cls.model_validate(json.loads(_extract_json(text)))


MAX_READS = 3   # each Read turn reprocesses the whole context -- the hidden cost cap


def _read_note() -> str:
    """The CLI-backend drill-down paragraph: the model may Read full-res chapter files."""
    return (
        "\n\nAlso available via the Read tool: map/chapter_NN.txt hold the "
        "FULL-resolution transcript per chapter. Read one before anchoring beats inside "
        f"it if the excerpts above are too coarse. Do not exceed {MAX_READS} Reads total."
    )


def _header(brief: str | None, title: str, target_s: float) -> str:
    what = brief or "(none -- find the stream's own best story)"
    return (
        f"STREAM TITLE: {title or '(none given)'}\n"
        f"EDITOR'S BRIEF: {what}\n"
        f"TARGET RUNTIME: ~{int(target_s)}s (rough guide, not a hard limit)\n\n"
    )


def outline(stream_map_text: str, brief: str, title: str, target_s: float,
            model: str = CLAUDE_MODEL, examples=None, trace=None,
            backend: str = "cli", run_dir=None) -> Outline:
    """One Claude call: stream map + brief -> story outline (central idea + beats).

    `examples` is the deferred reference-video few-shot hook (prior edits' beat
    breakdowns); unused in v1. `backend="cli"` (default) runs on the Claude Code CLI
    (subscription, ~$0) and falls through to the API key if the CLI is missing or fails;
    `"api"` goes straight to the SDK. Raises on no-backend-available / API error so the
    caller falls back to the flat pipeline. The map block is cache-marked so the API
    path's retry-without-thinking guard (same prefix) reads it warm.
    """
    system = pick_system(brief).format(n=_n_beats(target_s)) + _OUTLINE_INSTR
    user_content = [{
        "type": "text",
        "text": (_header(brief, title, target_s)
                 + f"STREAM MAP (timestamp -> what was said):\n{stream_map_text}"),
        "cache_control": {"type": "ephemeral"},
    }]
    if backend == "cli":
        try:
            cli_content = user_content
            if run_dir:
                cli_content = user_content + [{"type": "text", "text": _read_note()}]
            return _complete_cli(model, system, cli_content, Outline,
                                 tag="director", trace=trace,
                                 tools="Read" if run_dir else "",
                                 cwd=run_dir)
        except (RuntimeError, OSError) as e:
            print(f"[director] claude CLI unavailable ({e}); using API key")
    import anthropic  # optional heavy dep; missing key raises -> caller falls back

    client = anthropic.Anthropic()             # reads ANTHROPIC_API_KEY
    return _complete(client, model, system, user_content, Outline,
                     tag="director", trace=trace)


def review(stream_map_text: str, brief: str, title: str, outline_log: dict,
           target_s: float, model: str = CLAUDE_MODEL, trace=None,
           backend: str = "cli", run_dir=None) -> Review:
    """Critic pass: read the realized rough cut + stream map, approve or return a revised
    Outline. One Claude call; same `backend` semantics as `outline()` (CLI-first, API
    fallthrough). Raises on no-backend-available / API error so the caller keeps the
    current cut. The closed-loop signal is `_realized_script` -- the critic judges what the
    video ACTUALLY says, in order, not the original plan.

    The stream map comes FIRST (cache-marked, stable across rounds); the realized cut --
    the part that changes every round -- comes after, so an API-path review round 2 reads
    round 1's cached map at ~0.1x input price instead of re-paying a multi-hour stream.
    """
    system = _REVIEW_SYSTEM.format(n=_n_beats(target_s)) + _MAP_LEGEND
    if not brief:
        system += ("\nNo editorial brief was given: judge the cut against its own "
                   "central idea and the stream's strongest through-line.")
    system += _REVIEW_INSTR
    user_content = [
        {
            "type": "text",
            "text": (_header(brief, title, target_s)
                     + "FULL STREAM MAP (anchor any new or retimed beats to these "
                       f"timestamps):\n{stream_map_text}"),
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": ("CURRENT ROUGH CUT (what the video says, in order):\n"
                     f"{_realized_script(outline_log)}"),
        },
    ]
    if backend == "cli":
        try:
            cli_content = user_content
            if run_dir:
                cli_content = user_content + [{"type": "text", "text": _read_note()}]
            return _complete_cli(model, system, cli_content, Review,
                                 tag="review", trace=trace,
                                 tools="Read" if run_dir else "",
                                 cwd=run_dir)
        except (RuntimeError, OSError) as e:
            print(f"[review] claude CLI unavailable ({e}); using API key")
    import anthropic  # optional heavy dep; missing key raises -> caller keeps current cut

    client = anthropic.Anthropic()
    return _complete(client, model, system, user_content, Review,
                     tag="review", trace=trace)


def outline_local(stream_map_text: str, brief: str, title: str, target_s: float,
                  model: str = "llama3.1:8b") -> Outline:
    """Free local-Ollama director (no API spend). Coarser outlines than Claude; best on
    shorter streams since the whole map must fit the model's context. Same `Outline`.
    """
    import ollama  # already a project dep
    system = pick_system(brief).format(n=_n_beats(target_s)) + _OUTLINE_INSTR
    user = (_header(brief, title, target_s)
            + f"STREAM MAP (timestamp -> what was said):\n{stream_map_text}")
    resp = ollama.chat(
        model=model, format="json",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        options={"num_ctx": 8192},   # ponytail: bump for long streams if it truncates
    )
    return Outline.model_validate(json.loads(resp["message"]["content"]))
