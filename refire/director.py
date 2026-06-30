"""The director pass: read the whole stream, write a story outline (Claude).

One Claude call turns a time-stamped map of everything said in the stream into a
central idea plus an ordered list of BEATS -- story slots with a purpose ("the cocky
setup", "first crack", "the kill", "the reaction"). `narrative.cast` then fills each
beat with the single best real clip. This is the comprehension step that the local
embed/score pipeline can't do: it judges the stream as a story, not clips in isolation.

Runs on Claude Sonnet via ANTHROPIC_API_KEY; raises if the key/SDK is missing so the
caller (`pipeline.make`) can fall back to the flat retrieval path. Embeddings and
per-clip scoring stay local -- only this whole-stream reasoning call goes to the cloud.
"""
from __future__ import annotations

from pydantic import BaseModel

CLAUDE_MODEL = "claude-sonnet-4-6"


class Beat(BaseModel):
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
    query: str = ""                 # fallback retrieval text if start_s/end_s come out unusable


class Outline(BaseModel):
    central_idea: str
    story_shape: str = ""           # the ONE shape this VOD supports (e.g. "confidence -> chaos")
    viewer_promise: str = ""        # what the viewer is promised they're watching
    ending_needed: str = ""         # what kind of ending the story has to land
    beats: list[Beat]


class Review(BaseModel):
    approved: bool   # true = the rough cut already tells a cohesive story, ship it
    notes: str       # what's wrong (or why it's approved) -- logged to outline.json
    outline: Outline # the revised plan (full re-plan: reorder/drop/merge/add/rebound)


_SYSTEM = (
    "You are a video editor cutting a Twitch gaming stream into one focused, cohesive "
    "short. You are given the stream's title, the editor's brief (the subject and vibe "
    "to capture), and a time-stamped map of everything said. Do NOT list interesting "
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
    "- start_s and end_s: the exact in/out timestamps (seconds, from the stream map). "
    "Set start_s where the moment begins and end_s RIGHT AFTER its payoff or punchline "
    "lands, on a natural pause. Keep clips tight -- trim dead air.\n"
    "- setup_start_s: if the context the joke needs starts earlier than start_s, put it "
    "here (else omit).\n"
    "- payoff_start_s: where the joke/reveal/win/fail actually lands. NEVER cut before "
    "this.\n"
    "- reaction_end_s: a short tail after the payoff (the laugh/reaction) so the cut "
    "doesn't feel abrupt (else omit).\n"
    "- query: a few words to find the footage as a fallback if the timestamps fail.\n"
    "Shape the beats into an ARC -- open on a hook, set up, escalate, hit a climax, and "
    "land the ending_needed as a closing button -- NOT a flat list of equally-strong "
    "moments (that is a highlight reel, not a story). Make each beat follow from the one "
    "before via its transition_in, and never repeat the same kind of moment twice. "
    "Only include beats the footage can actually support. "
    "Aim for about {n} beats so the cut lands near the target runtime."
)


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
    "- PAYOFF COMPLETION: does any clip end BEFORE its payoff or reaction lands "
    "(cut off mid-joke)? Extend its end / payoff_start_s / reaction_end_s if so.\n"
    "- ENDING: does the central idea land by the end on a strong closing button?\n"
    "- PACING: does the energy vary, or is every beat the same intensity (monotony)?\n"
    "If the cut already tells a cohesive story, set approved=true and briefly say why in "
    "notes. Otherwise set approved=false, name the main problems in notes, and return a "
    "REVISED outline: reorder, drop, merge, or ADD connective beats, re-tag roles, and "
    "retighten each beat's in/out (start_s/end_s from the stream map, ending right after "
    "the payoff + reaction lands). Only include beats the footage supports. Keep it near "
    "the target runtime (about {n} beats)."
)


def stream_map(words: list[dict]) -> str:
    """Render the transcript as a compact `[mm:ss] sentence` map for the director.

    Sentence-level (break on .?!) so the director can choose precise in/out timestamps;
    each line's timestamp is that sentence's first word. ponytail: the whole-stream map
    fits Sonnet's context for normal VODs; for multi-hour streams this can blow the token
    budget -- window the stream or fall back to a chunk-level map if it ever truncates.
    """
    lines: list[str] = []
    cur: list[str] = []
    sent_start = 0
    for w in words:
        if not cur:
            sent_start = int(w["start"])
        cur.append(w["text"])
        if w["text"][-1:] in ".?!":
            lines.append(f"[{sent_start // 60:02d}:{sent_start % 60:02d}] {' '.join(cur)}")
            cur = []
    if cur:   # trailing words with no terminal punctuation
        lines.append(f"[{sent_start // 60:02d}:{sent_start % 60:02d}] {' '.join(cur)}")
    return "\n".join(lines)


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
        ts = (f"[{int(a) // 60:02d}:{int(a) % 60:02d}-{int(z) // 60:02d}:{int(z) % 60:02d}]"
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
    return "\n".join(lines).strip()


def _n_beats(target_s: float) -> int:
    return max(2, round(target_s / 45.0))   # ~45s of finished footage per beat


THINK_BUDGET = 10000   # explicit thinking cap so the JSON answer always has room after it


def _out_room(target_s: float) -> int:
    """Tokens reserved for the outline JSON itself (~N beats of structured output)."""
    return max(4000, _n_beats(target_s) * 300)


def _max_tokens(target_s: float) -> int:
    """Total output budget = thinking budget + reserved room for the answer.

    The first failure was adaptive thinking consuming the whole budget so no answer text
    block was emitted (parsed_output None -> silent flat fallback). Capping thinking
    explicitly and adding `_out_room` on top guarantees the JSON always fits. 24000 stays
    under Sonnet 4.x's output ceiling (no beta header needed).
    """
    return min(24000, THINK_BUDGET + _out_room(target_s))


def _complete(client, model, system, user, output_format, target_s, out_extra=0):
    """One structured Claude call with the thinking-budget + retry-without-thinking dance.

    Shared by the director and the review pass. Explicit thinking budget with answer room
    reserved AFTER it (`out_extra` adds room for the review's notes on top of the outline);
    if thinking ate the whole budget so no answer block was emitted (parsed_output None),
    retry once with thinking off (the JSON always emits then); still None -> raise so the
    caller falls back instead of silently degrading.
    """
    def _call(thinking):
        kw = dict(model=model, system=system,
                  messages=[{"role": "user", "content": user}],
                  output_format=output_format)
        if thinking:
            kw["thinking"] = {"type": "enabled", "budget_tokens": THINK_BUDGET}
            kw["max_tokens"] = min(24000, _max_tokens(target_s) + out_extra)
        else:
            kw["max_tokens"] = _out_room(target_s) + out_extra   # no thinking -> all answer
        return client.messages.parse(**kw)

    resp = _call(thinking=True)
    if resp.parsed_output is None:
        resp = _call(thinking=False)
    if resp.parsed_output is None:
        raise RuntimeError(f"no structured output (stop_reason={resp.stop_reason})")
    return resp.parsed_output


def _user_prompt(stream_map_text: str, brief: str, title: str, target_s: float) -> str:
    return (
        f"STREAM TITLE: {title or '(none given)'}\n"
        f"EDITOR'S BRIEF: {brief}\n"
        f"TARGET RUNTIME: {int(target_s)}s\n\n"
        f"STREAM MAP (timestamp -> what was said):\n{stream_map_text}"
    )


def outline(stream_map_text: str, brief: str, title: str, target_s: float,
            model: str = CLAUDE_MODEL, examples=None) -> Outline:
    """One Claude call: stream map + brief -> story outline (central idea + beats).

    `examples` is the deferred reference-video few-shot hook (prior edits' beat
    breakdowns); unused in v1. Raises on missing key / API error so the caller falls
    back to the flat pipeline.
    """
    import anthropic  # optional heavy dep; missing key raises -> caller falls back

    client = anthropic.Anthropic()             # reads ANTHROPIC_API_KEY
    # ponytail: no prompt cache -- v1 makes exactly one director call per run, so a
    # cached prefix would only pay the write premium. Add cache_control on the stream
    # map when the phase-2 editor pass makes a second call against the same map.
    system = _SYSTEM.format(n=_n_beats(target_s))
    user = _user_prompt(stream_map_text, brief, title, target_s)
    return _complete(client, model, system, user, Outline, target_s)


def review(stream_map_text: str, brief: str, title: str, outline_log: dict,
           target_s: float, model: str = CLAUDE_MODEL) -> Review:
    """Critic pass: read the realized rough cut + stream map, approve or return a revised
    Outline. One Claude call. Raises on missing key / API error so the caller keeps the
    current cut. The closed-loop signal is `_realized_script` -- the critic judges what the
    video ACTUALLY says, in order, not the original plan.

    ponytail: re-sends the stream map each round (no prompt cache yet). The director call
    is cheap and rounds are capped, so a cached-map prefix isn't worth the risk of
    restructuring the verified `outline()` call -- add cache_control on a shared map block
    if multi-hour streams make the re-send bite.
    """
    import anthropic  # optional heavy dep; missing key raises -> caller keeps current cut

    client = anthropic.Anthropic()
    system = _REVIEW_SYSTEM.format(n=_n_beats(target_s))
    user = (
        f"STREAM TITLE: {title or '(none given)'}\n"
        f"EDITOR'S BRIEF: {brief}\n"
        f"TARGET RUNTIME: {int(target_s)}s\n\n"
        f"CURRENT ROUGH CUT (what the video says, in order):\n"
        f"{_realized_script(outline_log)}\n\n"
        f"FULL STREAM MAP (anchor any new or retimed beats to these timestamps):\n"
        f"{stream_map_text}"
    )
    # out_extra reserves answer room for the review's notes on top of the outline JSON.
    return _complete(client, model, system, user, Review, target_s, out_extra=1500)


def outline_local(stream_map_text: str, brief: str, title: str, target_s: float,
                  model: str = "llama3.1:8b") -> Outline:
    """Free local-Ollama director (no API spend). Coarser outlines than Claude; best on
    shorter streams since the whole map must fit the model's context. Same `Outline`.
    """
    import json

    import ollama  # already a project dep
    system = _SYSTEM.format(n=_n_beats(target_s)) + (
        ' Reply ONLY with JSON of this shape: {"central_idea": "<str>", '
        '"story_shape": "<str>", "viewer_promise": "<str>", "ending_needed": "<str>", '
        '"beats": [{"title": "<str>", "role": "<hook|setup|escalation|reversal|climax|'
        'payoff|button>", "intent": "<str>", "viewer_question": "<str>", "turn": "<str>", '
        '"transition_in": "<str>", "texture": "<str>", "energy": <1-5>, '
        '"setup_start_s": <number|null>, "payoff_start_s": <number|null>, '
        '"reaction_end_s": <number|null>, "start_s": <number>, "end_s": <number>, '
        '"query": "<str>"}]}.'
    )
    resp = ollama.chat(
        model=model, format="json",
        messages=[{"role": "system", "content": system},
                  {"role": "user",
                   "content": _user_prompt(stream_map_text, brief, title, target_s)}],
        options={"num_ctx": 8192},   # ponytail: bump for long streams if it truncates
    )
    return Outline.model_validate(json.loads(resp["message"]["content"]))
