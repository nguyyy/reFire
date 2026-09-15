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
import time

from pydantic import BaseModel, field_validator, model_validator

from .pacing import (AVG_SHOT_S, BEAT_INFLATION_CAP,  # thresholds live in pacing;
                     fill_pace, scaled_budget)        # nothing imports back

# Story shape, continuity and "is this plot-critical or quest filler" are judgment calls,
# which is the axis Opus is actually better on -- and the caching fix below cut the token
# volume enough to pay for it. Was sonnet-5.
CLAUDE_MODEL = "claude-opus-5"
MAX_TOKENS = 32000   # adaptive thinking + the largest outline JSON both fit comfortably

# Nothing set an effort level before this, on either backend -- both ran at their own
# default. xhigh is the sweet spot for long-horizon reasoning; sweep it with --effort.
DEFAULT_EFFORT = "xhigh"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


class _JsonModel(BaseModel):
    """Base for the director's JSON payloads. In JSON mode Claude/Ollama emit `null` for
    optional fields; drop them so pydantic applies our defaults instead of rejecting None on
    a str-typed field (the strict output_format endpoint used to coerce these for us)."""

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data):
        # ponytail: models sometimes Title-Case the keys ("T_s"/"Why"); lowercase is enough
        # since every field name here is already lowercase.
        if isinstance(data, dict):
            return {k.lower(): v for k, v in data.items() if v is not None}
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
    # What plays before beat 1, lifted from footage the cut delivers later -- the one
    # deliberate break in chronological order. One segment is the flash-forward teaser
    # (the default); several is a montage STACK, the signature open of a dense clip reel.
    # A bare object still parses: older outlines and a model that ignores the list shape
    # both land here rather than failing the whole call.
    cold_open: list[Segment] = []
    beats: list[Beat]

    @field_validator("cold_open", mode="before")
    @classmethod
    def _as_list(cls, v):
        return [v] if isinstance(v, dict) else v


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
    '"ending_needed": "<str>", '
    '"cold_open": [{"start_s": <number>, "end_s": <number>}, ...], '
    '"beats": [{"title": "<str>", "role": '
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


# Fraction of the footage the director SELECTS that survives silence compression, i.e.
# finished_seconds / span_seconds. The director picks spans off the stream map and cannot
# see that `compress_silence` will strip the dead air inside them, so every number it is
# given -- the runtime target and the per-role budgets both -- is scaled up by this before
# it goes in the prompt, and the results are read back in finished seconds.
#
# 0.84 is the median over the 7 real cuts in run/, re-measured once `compress_silence`
# stopped cutting speech whisper returned no words for (`voiced`, 2026-09-13): 0.73-0.92
# per cut, where the same cuts measured ~0.81 before that fix (the older 0.77 predates
# both). It is only the PRIOR: after the first cast, `pipeline` feeds the shrink this
# stream actually realized into the review rounds. It matters on its own only when
# --review-rounds is 0. Calibrate here if cuts start landing consistently long (raise) or
# short (lower).
CUT_SHRINK = 0.84


def _role_note(shrink: float = 1.0, pace: float = 1.0) -> str:
    """`pacing.ROLE_BUDGET` as prompt prose, in the SPAN seconds the director picks.

    `pace` is the cut-speed knob, applied through `pacing.fill_pace` -- the SAME scaling
    `pacing.audit` applies to its ceiling, so a faster cut shortens what the director is
    TOLD and what the audit ENFORCES by the identical factor, and the two units still
    cannot drift. `fill_pace` and not `pace` because the beat COUNT is capped and the
    budgets have to make up the difference, or the ask cannot reach the target at all
    (see `pacing.fill_pace` and `_n_beats`).

    ROLE_BUDGET is finished video -- what the audit enforces -- so handing it to the
    director unscaled would under-fill every beat by the compression it can't see: it
    would pick a 90s span for a 90s ceiling and ship ~69s. Same table, two units.

    Brace-free by construction: the system prompts go through `.format()`.
    """
    return ", ".join(f"{role} {round(lo / shrink)}-{round(hi / shrink)}s"
                     for role, (lo, hi) in scaled_budget(fill_pace(pace)).items())


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
    "- cold_open: OPTIONAL flash-forward teaser, 3-6 seconds, played before the first "
    "beat. Lift it from the loudest instant of the climax or peak payoff -- the reaction "
    "itself, no setup and no context -- so the viewer opens on a question ('how did we get "
    "here?') and stays to see it answered. It MUST come from footage a later beat "
    "actually contains, or the video breaks the promise it opened with. Set it to null if "
    "the story genuinely has no single peak worth teasing, or if the first beat already "
    "IS that peak (a teaser 20 seconds ahead of itself is just a stutter).\n"
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
    "least ~3 seconds. BUDGET A BEAT'S SPANS BY ITS ROLE (seconds of stream-map time, "
    "already allowing for the dead air that gets cut out of them): {roles}"
    ". That budget is the SUM OF THE BEAT'S SEGMENTS -- the footage that survives "
    "into the video -- NOT the width of start_s..end_s. A beat whose segments add "
    "up to well under its budget is an under-filled beat, and a cut made of them "
    "ships far short of the target runtime. Not every beat deserves the same room. "
    "Spend the budget "
    "ON the moment, not around it: a beat whose comedy IS the repetition -- failed "
    "guesses, repeated attempts, a spiral getting worse -- must keep a representative RUN "
    "of those attempts inside its budget, not just the first line and the payoff. Two "
    "segments separated by minutes of skipped attempts is a jump to the punchline with no "
    "build, and it reads to the viewer as a missing scene. If a moment still sprawls past "
    "its ceiling, THIN the run -- drop the weaker attempts but keep the shape of the "
    "escalation -- rather than deleting the middle wholesale. "
    "A single segment equal to the whole span means 'keep everything'.\n"
    "  CUT BETWEEN EXCHANGES, NEVER INSIDE ONE. The transcript is one flat stream of "
    "everything audible -- the streamer's commentary AND the game's own dialogue, with no "
    "labels telling them apart -- so you have to read the shape of it. When a line is "
    "ANSWERED, the answer is part of that unit: a character speaks and the player reacts, "
    "a question and its reply, a setup and the laugh after it. Keep such a unit WHOLE or "
    "drop it WHOLE. A segment that ends the instant a character stops talking, with the "
    "response left outside it, does not read as a tight edit -- it reads as a non-sequitur, "
    "because the viewer sees the reaction to something the video never showed. A hole of a "
    "few seconds in the middle of a conversation is never the edit working; the holes that "
    "are the edit working are the long ones, BETWEEN moments.\n"
    "  Story and quest dialogue is exactly two things, and your job is to tell them "
    "apart. LOAD-BEARING: it turns the plot, reveals something, or the streamer's reaction "
    "and commentary over it is the entertainment -- keep the whole exchange, it is why the "
    "viewer is here. SLOP: fetch-quest chatter, objectives restated, menu and tutorial "
    "narration, NPCs saying nothing anyone will remember -- cut the whole exchange and "
    "move to the next real moment. Never take half. Half an exchange costs the same "
    "runtime as the whole thing and delivers none of it, which is the worst trade in the "
    "edit.\n"
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


# The structure paragraph is the one part of _SYSTEM a `--style` may overrule. The
# default mandates an arc and explicitly forbids the alternative ("that is a highlight
# reel, not a story"), so an editor asking for a highlight reel would be arguing with the
# prompt. _STYLE_RULE defers to the DIRECTION instead. It is sliced out of _SYSTEM by its
# own text rather than duplicated, so the default cannot drift out of the swap.
_ARC_HEAD = "Shape the beats into an ARC"
_ARC_TAIL = "Only include beats the footage can actually support."

# The cold-open paragraph is the second swappable region, on the same principle: the
# default describes ONE flash-forward teaser, and a montage stack is a different move
# entirely (many moments, none explained, escalating). Sliced out by its own text so the
# default cannot drift out of the swap -- same guarantee as the arc region above.
_STACK_HEAD = "- cold_open: OPTIONAL flash-forward teaser"
_STACK_TAIL = "Then break it into BEATS"

def _stack_rule(n: int) -> str:
    """The montage-stack replacement for the teaser paragraph. Brace-free by construction
    (an f-string, not a `{stack}` field) because the system prompts go through
    `.format(n=, roles=)` and a stray brace there is a KeyError mid-run."""
    return (
    f"- cold_open: a MONTAGE STACK of about {n} moments played before the first beat, "
    "in the order you list them. Each is 1-2.5 seconds -- only long enough to register "
    "WHAT KIND of moment it is, never long enough to explain it. Pick moments that read "
    "in under a second with no context at all: a scream, a sudden silence, a flat "
    "out-of-pocket line, a physical reaction. Give them NO setup and NO explanation; the "
    "viewer being half a beat behind is the point. Order them so they ESCALATE -- the "
    "weakest first, the single best moment LAST -- and lift every one of them from "
    "footage a later beat actually contains, because the stack is a promise the body has "
    "to pay off. Spoiling the best moments here is correct and intended. "
    )

_STYLE_RULE = (
    "Structure the beats the way the DIRECTION above asks for -- that instruction "
    "outranks any default shape. If the direction names no structure, shape them into "
    "an ARC: open on a hook, set up, escalate, hit a climax, and land the ending_needed "
    "as a closing button. WHATEVER the structure, give the ENDING as much deliberate "
    "construction as the opening: the final beat must BREATHE, not stop dead on the "
    "payoff. Set its end_s AFTER the reaction/aftermath (the laugh, the sigh, the 'what "
    "just happened' beat), and always give the final beat a reaction_end_s so the cut "
    "winds down instead of cutting off mid-breath. Make each beat follow from the one "
    "before via its transition_in, and never repeat the same kind of moment twice. "
)


# The third swappable region, and the one that decides whether a styled cut survives its
# own review. `_SYSTEM` and `_REVIEW_SYSTEM` both instruct at length on PROTECTING a
# moment: keep the whole run of attempts, pull the start earlier for context, extend the
# end until the payoff lands. Those are right for a story and are the exact opposite of a
# cut built on skipped setup and premature cuts -- and two of them are literal orders to
# ADD footage back, so leaving them in place means round 2 stuffs the cut back into shape
# and the direction quietly loses. `keep_build=False` (from a style document) swaps them
# for their mirror images. Same switch the pacing audit reads, so the prompt, the
# measurement and the style cannot disagree about what counts as a defect.
_SEG_HEAD = "Each segment should be at least"
_SEG_TAIL = "BUDGET A BEAT'S SPANS BY ITS ROLE"

_BUILD_HEAD = "Not every beat deserves the same room."
_BUILD_TAIL = "A single segment equal to the whole span means"

_DENSE_BUILD = (
    "Not every beat deserves the same room. Spend the budget ON the moment and nothing "
    "else: keep the lines that land and cut everything between them. A beat that keeps "
    "its whole build-up is not thorough, it is slack -- show enough of a repetition for "
    "the viewer to infer the rest, then leave. Never widen a beat to 'complete' it. "
    "This buys you more SKIPPED MOMENTS, not severed ones: leaving a moment early is this "
    "cut's signature, but cutting a reply out of the middle of an exchange still just "
    "breaks it. Skip whole exchanges freely; never take half of one. "
)

# Rubric bullets in _REVIEW_SYSTEM, each located by its own opening and the next bullet.
_RUBRIC_SWAPS = [
    ("- SETUP CLARITY:", "- EXPECTATION:",
     "- NO SETUP NEEDED: this cut opens moments mid-sequence on purpose, and a viewer "
     "who is half a beat behind is the intent rather than a defect. Do NOT add "
     "setup_start_s to give a moment context. Flag a beat only if it is literally "
     "unreadable -- not merely unexplained.\n"),
    # Ends at SEVERED EXCHANGE, not PAYOFF COMPLETION: that bullet now sits between the
    # two, and a swap reaching past it would delete the one continuity check a style is
    # not allowed to switch off. A dense cut skips MOMENTS; it still may not sever one.
    ("- MISSING BUILD-UP:", "- SEVERED EXCHANGE:",
     "- DENSITY: the failure mode for THIS cut is slackness, not missing build-up. A "
     "LONG gap inside a beat's span is the edit working -- but a few seconds of deleted "
     "speech is still a broken exchange, not density; see the next item.\n"),
    ("- PAYOFF COMPLETION:", "- ENDING:",
     "- PREMATURE CUTS ARE INTENDED: leaving during a reaction, a moment before a line "
     "finishes, is this cut's signature -- the viewer completes it after the cut has "
     "landed them somewhere new. Do NOT extend end_s, payoff_start_s or reaction_end_s "
     "to let a moment finish landing. The FINAL beat is the one exception: it still has "
     "to resolve.\n"),
]


def _seg_floor(pace: float) -> str:
    """The minimum segment length, in the same seconds the prompt states everything else.

    Hardcoded at 3s this silently contradicts every fast style: a 1-2.5s stack moment is
    below the floor the same prompt just set. Scales with `pace` like every other number
    the director is told, and renders byte-identically at pace 1.0.
    """
    return f"{max(1.0, 3.0 * pace):g}"


def _segs_per_beat(role: str, shrink: float = 1.0, pace: float = 1.0) -> int:
    """How many segments a mid-budget `role` beat implies at this cut speed.

    The director was given a kept-footage budget and a mean shot length and asked to
    satisfy both, which is one multiplication it consistently did not do: at pace 0.35 it
    returned 1.77 segments per beat against a budget that wanted ~7. Stating the product
    turns two abstract numbers into a count, which is the form the model actually acts on.
    """
    lo, hi = scaled_budget(fill_pace(pace)).get(role, (0.0, 0.0))
    mid = (lo + hi) / 2.0 / max(shrink, 0.05)
    return max(2, round(mid / max(AVG_SHOT_S * pace, 0.5)))


def _swap(text: str, head: str, tail: str, replacement: str) -> str:
    """Replace the region of `text` from `head` up to `tail` with `replacement`.

    Both swappable regions (the arc mandate, the cold open) are located by their own
    wording rather than duplicated as constants, so editing the default paragraph can
    never leave the swap pointing at stale text.
    """
    a = text.index(head)
    return text[:a] + replacement + text[text.index(tail, a):]


def _styled_system(style: str | None = None, stack: int = 0,
                   keep_build: bool = True, pace: float = 1.0) -> str:
    """`_SYSTEM` with its swappable regions replaced. The regions are INDEPENDENT: a
    `style` swaps the arc mandate, a `stack` swaps the teaser paragraph, `keep_build=False`
    swaps the protect-the-moment paragraph. `--stack` is a cold-open shape, not a licence
    to drop the story arc, so asking for one must not quietly rewrite the structure the
    editor never mentioned."""
    out = _SYSTEM
    if style:
        out = _swap(out, _ARC_HEAD, _ARC_TAIL, _STYLE_RULE)
    if stack > 0:
        out = _swap(out, _STACK_HEAD, _STACK_TAIL, _stack_rule(stack))
    if not keep_build:
        out = _swap(out, _BUILD_HEAD, _BUILD_TAIL, _DENSE_BUILD)
    # always: a 3s floor stated next to a paced budget contradicts it (identity at 1.0)
    seg = f"Each segment should be at least ~{_seg_floor(pace)} seconds. "
    if not keep_build:
        # A floor with no ceiling is how a dense style ships slow shots. `pacing.audit`
        # grades mean shot length against AVG_SHOT_S * pace -- but ONLY on the skipped-
        # build styles, and the director was never told the number it is measured by.
        # Unstated, it picked 14.0s-mean segments against a 4.2s ceiling and every ryuk
        # rule downstream was applied to shots 3x too long. Same constant and the same
        # scaling as the audit, stated exactly when the audit checks it, so what the
        # director is TOLD and what the audit ENFORCES still cannot drift.
        avg = AVG_SHOT_S * pace
        seg += (f"Across a beat those segments must AVERAGE about "
                f"{avg:.1f} seconds -- that average IS what this cut is "
                f"graded on, so prefer many short segments over a few long ones, and "
                f"never let one segment carry a whole beat. The two rules multiply: a "
                f"beat keeping S seconds at a {avg:.1f}s average is about S/{avg:.1f} "
                f"segments, so a mid-budget escalation is roughly "
                f"{_segs_per_beat('escalation', shrink=1.0, pace=pace)} separate spans, "
                f"not two or three. Filling the budget with FEWER, LONGER segments fails "
                f"both rules at once. ")
    return _swap(out, _SEG_HEAD, _SEG_TAIL, seg)


def pick_system(brief: str | None, style: str | None = None, stack: int = 0,
                keep_build: bool = True, pace: float = 1.0) -> str:
    """The outline system prompt for this mode: brief-driven (default) or stream-driven
    (no brief -> mine the stream's own best story), with the arc mandate swapped out when
    the editor gave a `style`, the teaser paragraph swapped for a montage stack when one
    was asked for, and the protect-the-moment paragraph swapped when the style is built on
    skipping it. Pure, so mode selection is testable."""
    body = _styled_system(style, stack, keep_build, pace)
    if brief:
        return body + _MAP_LEGEND
    # same beat schema + structure rules; only the opening framing changes
    intro_end = body.index("Do NOT list interesting")
    return _SYSTEM_STREAM_INTRO + body[intro_end:] + _MAP_LEGEND


_REVIEW_SYSTEM = (
    "You are a senior video editor reviewing a ROUGH CUT of a Twitch gaming short for "
    "STORY, not individual moments. You are given the cut's central idea, the editor's "
    "brief, the realized cut (what each beat ACTUALLY says, in order, with its intended "
    "role/intent/viewer_question), and the full stream map. Judge it as the VIEWER will "
    "experience it:\n"
    "- HOOK: do the first ~10 seconds grab attention?\n"
    "- COLD OPEN: if the cut opens with a flash-forward teaser, does the video actually "
    "deliver that moment later, and is it the strongest thing in the cut? A teaser of a "
    "weak moment, or of something the cut never reaches, is worse than no teaser.\n"
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
    "moment are natural. Spans are budgeted BY ROLE (stream-map seconds, already "
    "allowing for the dead air cut out of them): {roles}"
    ". That budget is the SUM OF A BEAT'S SEGMENTS -- the footage that reaches the "
    "viewer -- not the width of start_s..end_s. Flag a beat only against its OWN "
    "ceiling, and a beat sitting far UNDER its ceiling is under-filled, not tight.\n"
    "- MISSING BUILD-UP: the opposite failure, and the easier one to miss. An escalation "
    "or climax made of a short opener and a distant payoff, with the attempts between "
    "them skipped, is NOT a tight beat -- it is a beat with its build-up deleted, and it "
    "plays as a jump straight to the punchline. When a beat's segments leave a gap of "
    "more than ~60s inside its own span and the skipped footage is the moment escalating "
    "(more failed guesses, the spiral getting worse), RESTORE a representative run of it "
    "with extra `segments` up to that role's ceiling. Do not tighten this beat further.\n"
    "- SEVERED EXCHANGE: the same failure at the small end, and the one that survives "
    "every round because it is invisible in a joined transcript. The realized cut below "
    "marks each of its own jump cuts as `-- CUT Ns: \"...\"` with the words that were "
    "deleted there. READ THOSE. A cut of a few seconds that removed SPEECH from the "
    "middle of a beat has almost always taken one side of an exchange -- the reply to the "
    "line you kept, or the line the kept reaction was reacting to -- and the beat now "
    "plays as a non-sequitur. The transcript mixes the streamer's voice and the game's "
    "dialogue with no labels, so judge it by sense: does the kept text answer something "
    "that is no longer there? Fix it by RESTORING the missing side into the beat's "
    "`segments`, or if the whole exchange is filler, by dropping the whole thing. Never "
    "by tightening further. The pacing audit flags these as `severed_exchange`.\n"
    "- PAYOFF COMPLETION: does any clip end BEFORE its payoff or reaction lands "
    "(cut off mid-joke)? Extend its end / payoff_start_s / reaction_end_s if so.\n"
    "- ENDING: give the last beats the SAME scrutiny as the hook. Does the central idea "
    "land on a strong closing button, and does that button BREATHE -- holding on the "
    "reaction/aftermath -- rather than cutting off the instant the payoff lands (an abrupt "
    "ending)? If it stops dead, extend the final beat's end_s / reaction_end_s or add a "
    "short wind-down button beat.\n"
    "- MOMENTUM: the realized cut ends with a PACING AUDIT -- positions and durations "
    "measured from the finished video, not opinions. Treat its flags as facts and fix the "
    "stretch it names. A flat run is the cut losing the viewer: break it by re-cutting "
    "those beats tighter with `segments`, by moving a stronger moment from the stream map "
    "into that stretch, or by dropping whichever of them is redundant -- NOT by deleting "
    "the middle wholesale, which leaves the story with no bridge from setup to climax. A "
    "thin or over-long beat gets `segments`, not deletion. If both peaks land early, the "
    "problem is structural: find a later escalation the footage supports.\n"
    "- PACING: does the energy vary beat to beat, or is every beat the same intensity?\n"
    "- RUNTIME: the audit states the finished runtime. A cut well UNDER target is not a "
    "tight cut, it is an under-filled one -- and it is the defect that is hardest to "
    "see beat by beat, because every individual beat looks reasonable. Fix it by "
    "giving the beats that deserve it MORE `segments` up to their role ceilings "
    "(more kept moments, not longer ones), and by adding beats the stream map "
    "genuinely supports in the stretches the cut skips over. Never fix it by "
    "padding with slow footage.\n"
    "If the cut already tells a cohesive story, set approved=true and briefly say why in "
    "notes. Otherwise set approved=false, name the main problems in notes, and return a "
    "REVISED outline: reorder, drop, merge, or ADD connective beats, re-tag roles, and "
    "retighten each beat's in/out (start_s/end_s from the stream map, ending right after "
    "the payoff + reaction lands). Only include beats the footage supports. The target "
    "runtime (about {n} beats) is a rough guide, not a hard limit -- only add/drop/trim "
    "beats for the STORY reasons above; don't chase the exact number by padding a thin cut "
    "or gutting a beat's setup/payoff to shave seconds."
)


def _review_system(keep_build: bool = True) -> str:
    """`_REVIEW_SYSTEM` with the three protect-the-moment rubric items swapped for their
    mirror images when the style is built on skipping build-up.

    Those three are the ones that undo a styled cut: two of them explicitly instruct the
    critic to ADD footage back (`RESTORE a representative run`, `Extend its end`) and the
    third to pull a start earlier for context. Silencing the matching pacing flag is not
    enough on its own -- the rubric asks for the same thing in prose, and prose is what
    the critic actually re-plans from.
    """
    if keep_build:
        return _REVIEW_SYSTEM
    out = _REVIEW_SYSTEM
    for head, tail, replacement in _RUBRIC_SWAPS:
        out = _swap(out, head, tail, replacement)
    return out


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


# `ollama.chat` has no timeout and blocks forever (same hazard `score.SCORE_TIMEOUT_S`
# guards). Without it the `except -> stub chapter` fallback below is dead code against a
# real wedge: a blocked socket raises nothing, so the run hangs here with no Claude money
# spent and nothing written. Generous because this is a legitimately slow call -- a 20-min
# window on a local 20B measured 1.5-5 min -- and a timeout that trips on a healthy run is
# worse than none. A reasoning model that never stops thinking is what this catches.
SCOUT_TIMEOUT_S = 600.0


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
    client = ollama.Client(timeout=SCOUT_TIMEOUT_S)
    wins = _windows(map_text, window_s)
    n = max(1, len(wins))
    chapters: list[Chapter] = []
    t_start = time.monotonic()
    for i, (a, b, text) in enumerate(wins):
        k = i + 1
        # Report BEFORE the call as well as after: the call is the multi-minute part, so
        # a bar that only moves on completion sits still for exactly as long as the thing
        # the user wants to watch.
        if progress:
            progress(i / n, f"scouting chapter {k}/{n} "
                            f"({a / 3600:.2f}h-{b / 3600:.2f}h, {len(text) // 1000}k chars)")
        t_win = time.monotonic()
        try:
            resp = client.chat(
                model=model, format="json",
                messages=[{"role": "system", "content": _SCOUT_SYSTEM},
                          {"role": "user", "content": text}],
                options={"num_ctx": 16384})
            got = _ChapterOut.model_validate(json.loads(resp["message"]["content"]))
        except Exception as e:   # any window failure (incl. timeout) -> stub, keep scouting
            print(f"[scout] window {k} failed ({e}); using stub chapter", flush=True)
            got = _ChapterOut()
        dt = time.monotonic() - t_win
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        title = got.title or first[:60] or f"chapter {k}"
        chapters.append(Chapter(
            title=title, start_s=a, end_s=b, summary=got.summary,
            # clamp hallucinated stamps into the window
            notable_moments=[n_ for n_ in got.notable_moments if a <= n_.t_s <= b][:5]))
        if progress:
            eta = (time.monotonic() - t_start) / k * (n - k)
            progress(k / n, f"chapter {k}/{n} done {dt:.0f}s - {title[:40]} "
                            f"- ~{eta / 60:.1f}m left")
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(
            json.dumps([c.model_dump() for c in chapters], indent=2), encoding="utf-8")
    return chapters


def _realized_script(outline_log: dict, pace: float = 1.0,
                     keep_build: bool = True, target_s: float = 0.0) -> str:
    """Render the CAST cut as the script the critic reads: central idea + story shape +
    each beat in order with its editorial intent (role/intent/viewer_question/transition)
    AND the ACTUAL transcript of the chosen span.

    This is the closed-loop signal -- the critic judges what the video really says against
    what each beat was meant to do, not the director's original plan. Pure (no model) so
    it's unit-testable.

    Every beat leads with its position in the FINISHED CUT and the source span comes
    second. The source stamp alone (which is all this used to print) tells the critic
    where a moment sat in the stream, which is what it needs to RE-CUT a beat but tells it
    nothing about where the viewer is when they reach it -- so a rubric item like "does
    the energy vary?" had no timeline to vary across. `pacing.audit` closes the same gap
    with measurements the model can't do in its head.
    """
    from .pacing import _mmss, audit_note, cut_timeline

    lines = [f"CENTRAL IDEA: {outline_log.get('central_idea', '')}"]
    if outline_log.get("story_shape"):
        lines.append(f"STORY SHAPE: {outline_log['story_shape']}")
    if outline_log.get("viewer_promise"):
        lines.append(f"VIEWER PROMISE: {outline_log['viewer_promise']}")
    lines.append("")
    timeline = cut_timeline(outline_log)
    for i, b in enumerate(outline_log.get("beats", []), 1):
        a, z = b.get("start"), b.get("end")
        segs = b.get("segments") or []
        cut = (f", {len(segs)} cuts, {int(b.get('dur') or 0)}s kept" if len(segs) > 1 else "")
        ts = (f"source {int(a) // 60:02d}:{int(a) % 60:02d}-"
              f"{int(z) // 60:02d}:{int(z) % 60:02d}{cut}"
              if a is not None and z is not None else "source [--]")
        role = b.get("role") or "?"
        t = timeline[i - 1] if i <= len(timeline) else None
        where = (f"cut {_mmss(t['at'])}-{_mmss(t['end_at'])}" if t else "cut [--]")
        extra = ""
        if t:
            extra = f", {t['wps']:.1f} w/s"
            if t["setup_lead"] is not None:
                extra += f", {int(t['setup_lead'])}s to payoff"
        lines.append(f"{i}. {b.get('title', '')} [{role}] "
                     f"(energy {b.get('energy', '?')}, score {b.get('score', '?')}{extra})")
        lines.append(f"   intent: {b.get('intent', '')}")
        if b.get("transition_in"):
            lines.append(f"   transition_in: {b['transition_in']}")
        if b.get("viewer_question"):
            lines.append(f"   leaves viewer wondering: {b['viewer_question']}")
        lines.append(f"   [{where} | {ts}]")
        lines.extend(_script_body(b))
        lines.append("")
    lines.append(audit_note(outline_log, pace, keep_build, target_s))
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


DROPPED_PREVIEW = 140   # chars of a hole's transcript shown; enough to see what it was


def _script_body(b: dict) -> list[str]:
    """One beat's realized transcript WITH its jump cuts marked.

    This used to be `b["text"]` on a single line -- the kept spans joined by a space. Read
    that way a severed exchange is undetectable: a character's line runs straight into the
    next kept line and the reply deleted between them leaves no trace, so the critic was
    being asked to notice something it could not see. Now every hole is shown: how long,
    and what was said in it. That is the whole closed loop for continuity.
    """
    from .pacing import _mmss

    parts = b.get("parts") or []
    seams = b.get("seams") or []
    if not parts or len(parts) != len(seams) + 1:
        return [f'   "{str(b.get("text", "")).strip()}"']    # older log, or a single span
    out = []
    for i, part in enumerate(parts):
        if part.strip():
            out.append(f'   "{part.strip()}"')
        if i < len(seams):
            s = seams[i]
            gone = " ".join(str(s.get("text", "")).split())[:DROPPED_PREVIEW]
            out.append(f'   -- CUT {s["dur"]:.1f}s at {_mmss(s["at"])}: '
                       + (f'"{gone}"' if gone else "(no speech in the gap)"))
    return out


# Average finished footage per beat -- sets the beat COUNT only (`_n_beats`); what any
# single beat may actually spend is `pacing.ROLE_BUDGET`, weighted by role.
#
# This sat at 35.0 as "the blunt instrument against a middle that drags", and that was the
# wrong diagnosis: the drag was weak-score screen time, not too-few story turns, and 35s
# of allowance meant an escalation built on repetition (a wordle spiral, a word-ladder
# grind) could not hold its own attempts -- the director kept an opener plus the payoff
# and skipped the minutes between. 48 is the role-weighted average of ROLE_BUDGET, so the
# count and the per-beat budgets agree instead of fighting. Lower = more turns per minute;
# walk it back if cuts start feeling choppy, but raise ROLE_BUDGET, not this, if beats
# feel starved.
SEC_PER_BEAT = 48.0


# `BEAT_INFLATION_CAP` -- how far above the pace-1.0 count `pace` may push the beat ask --
# moved to `pacing`, where it sits next to the ROLE_BUDGET table it has to stay reconciled
# with (`pacing.fill_pace`). It is a ratio and not a flat ceiling because a genuinely long
# target legitimately wants more beats (60 min at pace 1.0 = 75) and must not be clamped
# to a short cut's count. The reason it exists: the director returns 15-24 beats whatever
# we ask for, so an inflated number buys no extra story turns -- only a bigger thinking
# bill. `--pace 0.35` once asked for 57 beats on a 16-min target; the model spent 57,408
# of its 64,000 output tokens deciding, overran the answer by 773, and returned 20 beats
# anyway -- and the truncated outline took the whole run down with it.


def _n_beats(target_s: float, pace: float = 1.0) -> int:
    """Beat COUNT for this runtime. `pace` scales it alongside the per-beat budgets it was
    derived from: shrinking the budgets alone would just ship a SHORTER video instead of a
    faster-cut one -- the exact fight the SEC_PER_BEAT note above records. Capped at
    `BEAT_INFLATION_CAP` x the pace-1.0 count; past that the ask is pure token waste.

    The cap re-opens that same fight from the other side -- a capped count with uncapped
    shrinking budgets cannot fill the target -- which is why the budgets are read at
    `pacing.fill_pace(pace)` rather than `pace`. Count and budget must multiply back out
    to `target_s`; that invariant is what `test_beat_count_and_budget_fill_the_target`
    pins."""
    n = round(target_s / (SEC_PER_BEAT * max(pace, 0.05)))
    ceiling = round(target_s / SEC_PER_BEAT * BEAT_INFLATION_CAP)
    return max(2, min(n, ceiling))


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


def _complete(client, model, system, user_content, model_cls, tag="claude", trace=None,
              effort: str = DEFAULT_EFFORT):
    """One JSON-mode Claude call (streamed live to the console) with a retry guard.

    Shared by the director and the review pass. The system prompt already asks for a single
    JSON object (the strict structured-output endpoint 400s on this schema's depth -- even
    more so now that beats carry `segments`), so we take the text block and validate it into
    `model_cls`. Adaptive thinking self-budgets; `MAX_TOKENS` leaves the answer plenty of
    room. Streaming because the SDK refuses non-streaming calls with max_tokens this large
    -- and it lets the terminal watch the model think and write the outline in real time
    (`display: "summarized"`; the raw chain of thought is never returned). `trace` = a dir
    that receives the full request/response text per call. If thinking somehow ate the
    whole budget so no text block was emitted, retry once at effort=low; still empty ->
    raise so the caller falls back instead of silently degrading.
    """
    user_text = "\n\n".join(b.get("text", "") for b in user_content)
    stamp = time.strftime("%H%M%S")
    print(f"[{tag}] -> {model}: system {len(system):,} chars, "
          f"user {len(user_text):,} chars (~{len(user_text) // 4000}k tok)")
    if trace:
        _trace_write(trace, f"{stamp}-{tag}-request.txt",
                     f"MODEL: {model}\n\n=== SYSTEM ===\n{system}\n\n"
                     f"=== USER ===\n{user_text}")
        print(f"[{tag}] full prompt -> {trace}\\{stamp}-{tag}-request.txt")

    def _call(level):
        # System as a BLOCK, not a bare string, so it can carry a breakpoint: it renders
        # ahead of the messages, so without one the map block's own marker can never be
        # reached on a second call. 1h TTL, not the 5-minute default -- review rounds were
        # measured 6m22s apart start-to-start, so a 5-minute entry expires between them.
        kw = dict(model=model, max_tokens=MAX_TOKENS,
                  system=[{"type": "text", "text": system,
                           "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                  thinking={"type": "adaptive", "display": "summarized"},
                  output_config={"effort": level},
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
        return text, resp.stop_reason, "".join(thought), getattr(resp, "usage", None)

    text, stop, thought, usage = _call(effort)
    # `max_tokens` with text present = thinking ate most of the budget and the answer got
    # cut off mid-JSON. Unlike the CLI path there is no auto-continuation to stitch back
    # together here, so take the same escape hatch as the empty case: starve the thinking
    # and hand the budget to the answer (a 20-beat outline is ~7.4k tokens).
    #
    # Effort "low", NOT thinking {"type": "disabled"}, which is what this used to do:
    # Opus 5 rejects disabled thinking with a 400 at effort xhigh or max, so the old guard
    # would turn a recoverable truncation into a hard failure on the model we now default
    # to. Low effort reaches the same place -- barely any thinking -- and is always legal.
    if not text.strip() or stop == "max_tokens":
        print(f"[{tag}] {'empty' if not text.strip() else 'truncated'} answer "
              f"(stop_reason={stop}); retrying at effort=low")
        text, stop, thought, usage = _call("low")
    cache = ""
    if usage is not None:
        # Nothing read these before, which is why a breakpoint that never fired went
        # unnoticed for so long. A warm round shows a big `read` and a near-zero `in`.
        cache = (f"CACHE: read={getattr(usage, 'cache_read_input_tokens', 0):,} "
                 f"write={getattr(usage, 'cache_creation_input_tokens', 0):,} "
                 f"in={getattr(usage, 'input_tokens', 0):,} "
                 f"out={getattr(usage, 'output_tokens', 0):,}\n")
        print(f"[{tag}] {cache.strip()}")
    print(f"[{tag}] <- stop_reason={stop}, {len(text):,} chars")
    if trace:
        _trace_write(trace, f"{stamp}-{tag}-response.txt",
                     f"STOP_REASON: {stop}\n{cache}\n=== THINKING (summarized) ===\n{thought}\n\n"
                     f"=== ANSWER ===\n{text}")
    if not text.strip():
        raise RuntimeError(f"no answer text (stop_reason={stop})")
    return model_cls.model_validate(json.loads(_extract_json(text)))


_SPIN = "|/-\\"


def _think_tick(tag: str, secs: float, beats: int) -> str:
    """One live line for a REDACTED thinking delta.

    The subscription CLI streams `thinking_delta` on a ~1.6s heartbeat but blanks the
    text, so the only honest thing to show is that it is still going, and for how long.
    Carriage return, not newline: a 7-minute director call ticks ~260 times. ASCII only
    -- the console is cp1252, where a spinner glyph outside it kills the run mid-call.
    """
    return f"\r[{tag}] thinking... {int(secs) // 60}m{int(secs) % 60:02d}s {_SPIN[beats % 4]} "


def _complete_cli(model, system, user_content, model_cls, tag="claude", trace=None,
                  tools: str = "", cwd=None, timeout_s: float = 1200.0,
                  effort: str = DEFAULT_EFFORT,
                  session_id: str | None = None, resume: bool = False):
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

    `session_id` + `resume` are how a run stops paying for the stream map three times. The
    director call opens a named session (`--session-id`); each review round continues it
    (`--resume`) and sends ONLY the new material, because the ~240KB map is already in the
    conversation. Blocks are flattened to one stdin string here, so `cache_control` never
    survives this path -- session continuation is the CLI's equivalent, and it is better:
    the map is not re-sent at all rather than re-sent and read warm.

    One constraint that shapes the caller: the system prompt is fixed when a session is
    created and cannot be swapped on a resume, so a resumed call prepends `system` to the
    user turn instead. Callers must be ready for a resume to fail (stale session, older
    CLI) and retry as a normal standalone call -- see `review()`.
    """
    import os
    import shutil
    import subprocess
    import tempfile

    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("claude CLI not on PATH")
    user_text = "\n\n".join(b.get("text", "") for b in user_content)
    warm = bool(resume and session_id)
    if warm:
        # The session already holds the map and the director's own answer; --system-prompt-file
        # is ignored on a resume, so the rubric has to travel as user text.
        user_text = f"{system}\n\n{user_text}"
    stamp = time.strftime("%H%M%S")
    how = f"resume {session_id[:8]}" if warm else "new session"
    print(f"[{tag}] -> {model} via claude CLI (subscription, {how}, effort={effort}): "
          f"system {len(system):,} chars, "
          f"user {len(user_text):,} chars (~{len(user_text) // 4000}k tok)")
    if trace:
        _trace_write(trace, f"{stamp}-{tag}-request.txt",
                     f"MODEL: {model} (claude CLI, {how}, effort={effort})\n\n"
                     f"=== SYSTEM ===\n{'(in user turn -- resumed session)' if warm else system}"
                     f"\n\n=== USER ===\n{user_text}")
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
           "--effort", effort, "--output-format", "stream-json",
           "--include-partial-messages", "--verbose"]
    if warm:
        cmd += ["--resume", session_id]      # system prompt is the session's already
    else:
        cmd += ["--system-prompt-file", sp.name]
        if session_id:
            cmd += ["--session-id", session_id]   # so the review rounds can resume it
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
    t_call, beats = time.monotonic(), 0
    # Every assistant TURN's text, not just the last. When thinking eats the output budget
    # the message stops on `max_tokens` and Claude Code silently continues in a second
    # message -- but the `result` event below carries only that FINAL turn, so a JSON
    # answer split across turns arrives as a mid-object fragment. (Cost us a whole video:
    # 57,408 of 64,000 output tokens went to thinking, the outline overran by 773, and the
    # 16,487-char head was dropped on the floor.) Whole content blocks, not `text_delta`
    # reassembly -- they don't depend on --include-partial-messages delta ordering.
    turns: list[str] = []
    stops: list[str] = []
    text, stop, is_err = "", None, False
    for line in proc.stdout:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "assistant":
            msg = ev.get("message") or {}
            for b in msg.get("content") or []:
                if b.get("type") == "text" and b.get("text"):
                    turns.append(b["text"])
            if msg.get("stop_reason"):
                stops.append(msg["stop_reason"])
        elif ev.get("type") == "stream_event":
            d = (ev.get("event") or {}).get("delta") or {}
            kind = d.get("type", "")
            if kind == "thinking_delta":
                # The subscription CLI sends these BLANK. Keep them anyway: an empty
                # chunk is still a heartbeat (~1.6s), and dropping it -- the
                # `and d.get("thinking")` guard this replaces -- is what left the
                # terminal dead for the whole think, ~6 of the 7 minutes on a
                # director call.
                chunk = d.get("thinking") or ""
                if chunk:
                    thought.append(chunk)
            elif kind == "text_delta" and d.get("text"):
                chunk = d["text"]
            else:
                continue
            label = "thinking" if kind == "thinking_delta" else "writing"
            if phase != label:
                print(f"\n[{tag}] --- {label} ---", flush=True)
                phase = label
            if chunk:
                print(chunk, end="", flush=True)
            else:
                beats += 1
                print(_think_tick(tag, time.monotonic() - t_call, beats),
                      end="", flush=True)
        elif ev.get("type") == "result":
            text = ev.get("result") or ""
            stop = ev.get("stop_reason") or ev.get("subtype")
            is_err = bool(ev.get("is_error"))
    proc.wait()
    killer.cancel()
    os.unlink(sp.name)
    if phase:
        print(flush=True)
    joined = "".join(turns)
    print(f"[{tag}] <- stop_reason={stop}, {len(text):,} chars "
          f"({len(turns)} turn{'s' * (len(turns) != 1)}, {len(joined):,} chars joined)")
    if trace:
        # turn count + per-turn stop_reasons, because a fragment that LOOKS like a whole
        # answer is exactly what made this failure take hours to spot in the last dump.
        _trace_write(trace, f"{stamp}-{tag}-response.txt",
                     f"STOP_REASON: {stop}\nTURNS: {len(turns)} "
                     f"({', '.join(stops) or 'n/a'})\n\n=== THINKING (summarized) ===\n"
                     f"{''.join(thought)}\n\n=== ANSWER (result event) ===\n{text}"
                     f"\n\n=== ANSWER (all turns joined) ===\n{joined}")
    if proc.returncode != 0 or is_err:
        raise RuntimeError(f"claude CLI failed (exit {proc.returncode}, "
                           f"stop_reason={stop}): {text[:200]}")
    # `result` first: on a healthy single-turn call it IS the answer, and it keeps any
    # drill-down narration (tools="Read") out of the parse. The join is the recovery path.
    for i, candidate in enumerate((text, joined)):
        if not candidate.strip():
            continue
        try:
            parsed = model_cls.model_validate(json.loads(_extract_json(candidate)))
        except ValueError:            # JSONDecodeError / _extract_json / pydantic all subclass it
            continue
        if i:
            print(f"[{tag}] result was a {len(text):,}-char continuation fragment; "
                  f"recovered {len(joined):,} chars from {len(turns)} turns")
        return parsed
    raise RuntimeError(
        f"claude CLI gave no parseable answer (stop_reason={stop}, turns={len(stops)}"
        f"{': ' + ', '.join(stops) if stops else ''}): "
        f"result {len(text):,} chars, joined {len(joined):,} chars")


MAX_READS = 3   # each Read turn reprocesses the whole context -- the hidden cost cap


def _read_note() -> str:
    """The CLI-backend drill-down paragraph: the model may Read full-res chapter files."""
    return (
        "\n\nAlso available via the Read tool: map/chapter_NN.txt hold the "
        "FULL-resolution transcript per chapter. Read one before anchoring beats inside "
        f"it if the excerpts above are too coarse. Do not exceed {MAX_READS} Reads total."
    )


def _header(brief: str | None, title: str, target_s: float,
            style: str | None = None) -> str:
    """Brief + title + target runtime: the part of the preamble that is the SAME every
    round, so it can sit in front of the cached stream map without invalidating it.

    Everything derived from the measured `shrink` lives in `_budget_note` instead and is
    appended AFTER the map. It used to be concatenated here, inside the cache-marked
    block -- and because `pipeline` re-measures `shrink` every review round, one changed
    digit ("~1190s" -> "~1154s") moved the prefix and every one of the ~240KB after it
    missed. Measured on a real 3-call run: 1,695 bytes of shared prefix out of 250,000.
    """
    what = brief or "(none -- find the stream's own best story)"
    # The brief says WHAT to cut; the direction says HOW. Every director call routes
    # through this helper, so this one line reaches outline, review and the local path.
    how = f"DIRECTION (how to cut it): {style}\n" if style else ""
    runtime = (f"TARGET RUNTIME: ~{int(target_s)}s of FINISHED video "
               f"(rough guide, not a hard limit)\n")
    return (f"STREAM TITLE: {title or '(none given)'}\n"
            f"EDITOR'S BRIEF: {what}\n" + how + runtime + "\n")


def _budget_note(target_s: float, shrink: float = CUT_SHRINK, pace: float = 1.0) -> str:
    """The volatile half of the preamble: everything the measured `shrink` moves.

    `target_s` is FINISHED video; the spans the director picks are longer than what they
    ship, because the dead air inside them is cut out. Stating only the target made it aim
    its spans at the target and deliver `shrink` x target -- a 14-minute ask shipping under
    11. So it gets told the span budget too, and why.

    Small and last on purpose. The system prompt quotes the per-role budgets at the
    `CUT_SHRINK` prior so it stays byte-identical across rounds; once a cast has MEASURED
    this stream's real compression, the corrected table is restated here, after the map,
    where rewriting it costs one cache miss on a few hundred bytes instead of on all of it.
    """
    if shrink >= 1.0:
        return ""
    out = (f"\n\nSPAN BUDGET: pick spans totalling ~{int(target_s / shrink)}s. The silence "
           f"inside every span you choose is removed automatically -- on this stream "
           f"that is about {round((1 - shrink) * 100)}% of it -- so spans adding up to "
           f"the target itself would ship well SHORT of it.")
    # Only when the measurement actually disagrees with the prior the system prompt used;
    # restating an identical table would just be noise the model has to reconcile.
    if abs(shrink - CUT_SHRINK) > 0.01:
        out += ("\nMEASURED ROLE BUDGETS (these supersede the per-role seconds in the "
                f"instructions above, which used an estimate): {_role_note(shrink, pace)}.")
    return out + "\n"


def outline(stream_map_text: str, brief: str, title: str, target_s: float,
            model: str = CLAUDE_MODEL, examples=None, trace=None,
            backend: str = "cli", run_dir=None,
            shrink: float = CUT_SHRINK, style: str | None = None,
            pace: float = 1.0, stack: int = 0,
            keep_build: bool = True, effort: str = DEFAULT_EFFORT,
            session_id: str | None = None) -> Outline:
    """One Claude call: stream map + brief -> story outline (central idea + beats).

    `examples` is the deferred reference-video few-shot hook (prior edits' beat
    breakdowns); unused in v1. `backend="cli"` (default) runs on the Claude Code CLI
    (subscription, ~$0) and falls through to the API key if the CLI is missing or fails;
    `"api"` goes straight to the SDK. Raises on no-backend-available / API error so the
    caller falls back to the flat pipeline. The map block is cache-marked so the API
    path's effort=low retry guard (same prefix) reads it warm.
    """
    # CUT_SHRINK, not the caller's measured `shrink`: the system prompt has to render the
    # same bytes on every call of a run or it moves the prefix ahead of the whole map.
    system = (pick_system(brief, style, stack, keep_build, pace)
              .format(n=_n_beats(target_s, pace), roles=_role_note(CUT_SHRINK, pace))
              + _OUTLINE_INSTR)
    user_content = [
        {
            "type": "text",
            "text": (_header(brief, title, target_s, style)
                     + f"STREAM MAP (timestamp -> what was said):\n{stream_map_text}"),
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
        # After the breakpoint: the only part that moves between calls.
        {"type": "text", "text": _budget_note(target_s, shrink, pace)},
    ]
    if backend == "cli":
        try:
            cli_content = user_content
            if run_dir:
                cli_content = user_content + [{"type": "text", "text": _read_note()}]
            return _complete_cli(model, system, cli_content, Outline,
                                 tag="director", trace=trace,
                                 tools="Read" if run_dir else "",
                                 cwd=run_dir, effort=effort,
                                 session_id=session_id)
        except (RuntimeError, OSError) as e:
            print(f"[director] claude CLI unavailable ({e}); using API key")
    import anthropic  # optional heavy dep; missing key raises -> caller falls back

    client = anthropic.Anthropic()             # reads ANTHROPIC_API_KEY
    return _complete(client, model, system, user_content, Outline,
                     tag="director", trace=trace, effort=effort)


def review(stream_map_text: str, brief: str, title: str, outline_log: dict,
           target_s: float, model: str = CLAUDE_MODEL, trace=None,
           backend: str = "cli", run_dir=None,
           shrink: float = CUT_SHRINK, style: str | None = None,
           pace: float = 1.0, stack: int = 0,
           keep_build: bool = True, effort: str = DEFAULT_EFFORT,
           session_id: str | None = None) -> Review:
    """Critic pass: read the realized rough cut + stream map, approve or return a revised
    Outline. One Claude call; same `backend` semantics as `outline()` (CLI-first, API
    fallthrough). Raises on no-backend-available / API error so the caller keeps the
    current cut. The closed-loop signal is `_realized_script` -- the critic judges what the
    video ACTUALLY says, in order, not the original plan.

    The stream map comes FIRST (cache-marked, stable across rounds); the realized cut --
    the part that changes every round -- comes after, so an API-path review round 2 reads
    round 1's cached map at ~0.1x input price instead of re-paying a multi-hour stream.
    """
    system = (_review_system(keep_build).format(n=_n_beats(target_s, pace),
                                                roles=_role_note(CUT_SHRINK, pace))
              + _MAP_LEGEND)
    if not brief:
        system += ("\nNo editorial brief was given: judge the cut against its own "
                   "central idea and the stream's strongest through-line.")
    if style:
        # Load-bearing. The ARC bullet in the rubric above is a default, not a law:
        # without this the critic 'fixes' a deliberately styled cut back into a story
        # arc, and round 2 silently undoes the direction the editor set.
        # Load-bearing, and it must outrank the WHOLE rubric rather than one item of it:
        # naming only ARC leaves a model dutifully applying the other twelve, several of
        # which ask it to widen beats and restore skipped footage.
        system += ("\nThe editor gave a DIRECTION for HOW to cut this, not just what "
                   "to cut. The rubric above is the DEFAULT house style, not a law: "
                   "wherever it and the direction disagree, the direction wins -- "
                   "structure, pacing, how much context a moment gets, and how long it "
                   "is allowed to land. Never flag a cut for following its direction, "
                   "and never widen, extend or re-add footage to satisfy a rubric item "
                   "the direction has overruled. If the cut obeys the direction and the "
                   "rubric still complains, the rubric is what is wrong.")
    if stack > 0:
        # The COLD OPEN rubric item above judges a single teaser; a stack is a different
        # object with its own failure modes, and without this the critic collapses it
        # back to one clip on the first round.
        system += ("\nThe cold open is a MONTAGE STACK, not a single teaser: about "
                   f"{stack} moments of 1-2.5s each, played back to back with no setup "
                   "and no explanation. Judge it as a stack -- every moment must be "
                   "delivered again later in the cut, the strongest one must come LAST "
                   "so it escalates, and none of them should be long enough to explain "
                   "itself. Do not collapse it to one clip, and do not ask for context "
                   "inside it; the viewer being half a beat behind is the intent.")
    system += _REVIEW_INSTR
    # The only part that changes between rounds. Kept separate because a resumed CLI
    # session sends THIS ALONE -- the map is already in the conversation.
    tail = {
        "type": "text",
        "text": (_budget_note(target_s, shrink, pace)
                 + "\nCURRENT ROUGH CUT (what the video says, in order):\n"
                 f"{_realized_script(outline_log, pace, keep_build, target_s)}"),
    }
    user_content = [
        {
            "type": "text",
            "text": (_header(brief, title, target_s, style)
                     + "FULL STREAM MAP (anchor any new or retimed beats to these "
                       f"timestamps):\n{stream_map_text}"),
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
        tail,
    ]

    if backend == "cli":
        read = [{"type": "text", "text": _read_note()}] if run_dir else []
        tools = "Read" if run_dir else ""
        if session_id:
            try:
                # Warm path: the director opened this session and the ~240KB map is already
                # in it. Send the new cut ONLY -- three copies of the map per run was most
                # of the ~15 minutes.
                return _complete_cli(model, system, [tail] + read, Review,
                                     tag="review", trace=trace, tools=tools,
                                     cwd=run_dir, effort=effort,
                                     session_id=session_id, resume=True)
            except (RuntimeError, OSError) as e:
                # Stale session, older CLI, whatever: never let a cache optimisation cost
                # a run that has already spent a director call. Re-send the map and go on.
                print(f"[review] session resume failed ({e}); re-sending the full map")
        try:
            return _complete_cli(model, system, user_content + read, Review,
                                 tag="review", trace=trace, tools=tools,
                                 cwd=run_dir, effort=effort)
        except (RuntimeError, OSError) as e:
            print(f"[review] claude CLI unavailable ({e}); using API key")
    import anthropic  # optional heavy dep; missing key raises -> caller keeps current cut

    client = anthropic.Anthropic()
    return _complete(client, model, system, user_content, Review,
                     tag="review", trace=trace, effort=effort)


def outline_local(stream_map_text: str, brief: str, title: str, target_s: float,
                  model: str = "llama3.1:8b",
                  shrink: float = CUT_SHRINK, style: str | None = None,
                  pace: float = 1.0, stack: int = 0,
                  keep_build: bool = True) -> Outline:
    """Free local-Ollama director (no API spend). Coarser outlines than Claude; best on
    shorter streams since the whole map must fit the model's context. Same `Outline`.
    """
    import ollama  # already a project dep
    # CUT_SHRINK for the same reason as outline(): a measured shrink is restated by
    # `_budget_note`, whose "supersede the estimate above" wording assumes this prior.
    system = (pick_system(brief, style, stack, keep_build, pace)
              .format(n=_n_beats(target_s, pace), roles=_role_note(CUT_SHRINK, pace))
              + _OUTLINE_INSTR)
    user = (_header(brief, title, target_s, style)
            + f"STREAM MAP (timestamp -> what was said):\n{stream_map_text}"
            + _budget_note(target_s, shrink, pace))
    resp = ollama.chat(
        model=model, format="json",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        options={"num_ctx": 8192},   # ponytail: bump for long streams if it truncates
    )
    return Outline.model_validate(json.loads(resp["message"]["content"]))
