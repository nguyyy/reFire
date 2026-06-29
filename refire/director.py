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
    title: str       # short on-screen section-card title
    intent: str      # this beat's job in the story (also the local scorer's mini-brief)
    start_s: float   # clip in-point (seconds) -- where this beat's moment begins
    end_s: float     # clip out-point (seconds) -- right after its payoff lands
    query: str       # fallback retrieval text if start_s/end_s come out unusable


class Outline(BaseModel):
    central_idea: str
    beats: list[Beat]


_SYSTEM = (
    "You are a video editor cutting a Twitch gaming stream into one focused, "
    "cohesive short. You are given the stream's title, the editor's brief (the "
    "subject and vibe to capture), and a time-stamped map of everything said. "
    "Decide the SINGLE STORY this cut should tell, then break it into BEATS in "
    "story order -- which for a mostly-linear gaming stream is chronological. "
    "A beat is a slot in the story with a purpose, not just an interesting moment. "
    "Each beat needs: a short title (shown on-screen as a section card), an intent "
    "(its job in the story, e.g. 'establish he's overconfident before it goes wrong'), "
    "start_s and end_s -- the exact in and out timestamps (in seconds, taken from the "
    "stream map) for this beat's clip, and a query (a few words to find the footage as "
    "a fallback if the timestamps don't work out). Set start_s where the moment begins "
    "and end_s RIGHT AFTER its payoff or punchline lands, on a natural pause -- never "
    "cut a setup off from the line that pays it off, and never end a beat before the "
    "joke resolves. Keep clips tight: trim dead air around the moment. "
    "Only include beats the footage can actually support. "
    "Aim for about {n} beats so the cut lands near the target runtime."
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


def _n_beats(target_s: float) -> int:
    return max(2, round(target_s / 45.0))   # ~45s of finished footage per beat


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
    resp = client.messages.parse(
        model=model,
        max_tokens=8000,
        thinking={"type": "adaptive"},   # narrative reasoning benefits from thinking
        system=_SYSTEM.format(n=_n_beats(target_s)),
        messages=[{"role": "user",
                   "content": _user_prompt(stream_map_text, brief, title, target_s)}],
        output_format=Outline,
    )
    return resp.parsed_output


def outline_local(stream_map_text: str, brief: str, title: str, target_s: float,
                  model: str = "llama3.1:8b") -> Outline:
    """Free local-Ollama director (no API spend). Coarser outlines than Claude; best on
    shorter streams since the whole map must fit the model's context. Same `Outline`.
    """
    import json

    import ollama  # already a project dep
    system = _SYSTEM.format(n=_n_beats(target_s)) + (
        ' Reply ONLY with JSON of this shape: {"central_idea": "<str>", "beats": '
        '[{"title": "<str>", "intent": "<str>", "start_s": <number>, '
        '"end_s": <number>, "query": "<str>"}]}.'
    )
    resp = ollama.chat(
        model=model, format="json",
        messages=[{"role": "system", "content": system},
                  {"role": "user",
                   "content": _user_prompt(stream_map_text, brief, title, target_s)}],
        options={"num_ctx": 8192},   # ponytail: bump for long streams if it truncates
    )
    return Outline.model_validate(json.loads(resp["message"]["content"]))
