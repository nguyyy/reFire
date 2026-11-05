"""Structured critics: four job-specific reviewers of the REALIZED cut, JSON edits only.

Persona critics converge on vague notes and cannot feel pacing from text. These each
have one job and answer with edits on span ids, never prose:
  continuity -- does each clip open with established context? (extend_start)
  payoff     -- does each clip end after the payoff and the reaction land? (extend_end)
  filler     -- spans that fail the removal test (cut)
  dropoff    -- where a viewer would click away, and why (flag / cut)

They read the cut as a viewer hears it: utterances in play order with ids, `[CUT 3.2s]`
where the edit removed footage, `== CLIP n ==` at each clip head -- never the text that
was cut away. Continuity and payoff see only the clip edges they judge (a fraction of
the tokens); filler and dropoff see the whole cut. The four run in parallel.

The arbiter is deterministic: edits under APPLY_CONF are ignored, a span that one critic
wants cut and another wants kept or extended is a CONFLICT -- left alone and turned into
a review marker -- and everything else is applied: extensions to the nearest
`boundary.candidates` point within the learned cap, cuts through `trim.cut_region` (so
never mid-word). At most MAX_ROUNDS rounds, stopping early when nothing changes.
Few-shot examples come from clip_lessons (the caller passes leave-one-out lessons).

Claude, not ollama: this is whole-cut judgment, one call per critic per round, at the
run's own model and a low effort (`--critic-effort`). Off by default (`--critics`):
measured 2026-11 (LOO, 4 finished edits) it RAISED fix_cost 863.2 -> 914.1s -- 46 edits
applied, mostly continuity start-extensions and drop-off cuts the editor never made,
zero conflicts -- for 32 calls / ~265k processed input tokens ($0.64 notional per VOD).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

APPLY_CONF = 0.7
MAX_ROUNDS = 2
EDGE_SPANS = 3            # continuity/payoff see this many spans at each clip edge
MAX_CUT_FRAC = 0.05       # same ceiling as trim.py: the editor's measured rate
DEFAULT_EFFORT = "low"


class Edit(BaseModel):
    span_id: str
    action: str           # extend_start | extend_end | cut | keep | flag
    reason: str = ""
    confidence: float = 0.0


class Edits(BaseModel):
    edits: list[Edit] = []


_RULES = (
    "You review a rough cut of a Twitch stream as the viewer will hear it. Each line is "
    "one utterance: `S<n> text`. `== CLIP n ==` starts a new clip (a jump to another "
    "moment of the stream); `[CUT 3.2s]` marks footage the edit removed. Reply ONLY with "
    'JSON: {"edits": [{"span_id": "S12", "action": "<action>", "reason": "<short>", '
    '"confidence": <0-1>}]}. Return an EMPTY list when nothing is wrong -- most cuts an '
    "editor already chose are mostly right, and every edit you return costs them time "
    "to check. Only flag what you are sure of.\n")

JOBS = {
    "continuity": (
        "Your only job: CONTINUITY. For each clip, does its first line make sense to a "
        "viewer who did not see what came before it? A clip that opens mid-thought, on "
        "an unresolved 'he/that/this/again', or reacting to something not shown needs "
        "an earlier start: action extend_start on that clip's FIRST span. If a later "
        "span only makes sense because of a span another critic might cut, return keep "
        "on the span it depends on.", ("start_late",), "edges_start"),
    "payoff": (
        "Your only job: PAYOFF. For each clip, does its last line come after the joke, "
        "reveal, win or fail has landed AND the reaction has played out? If the clip "
        "stops on the setup, mid-reaction or mid-sentence, return extend_end on that "
        "clip's LAST span.", ("end_early",), "edges_end"),
    "filler": (
        "Your only job: FILLER. A span is filler when no later span depends on it for "
        "meaning AND it carries no energy: stalling, loading/setup talk, verbal tics, "
        "repeated lines, reading chat. Return cut for those. Never cut a setup a later "
        "punchline needs, or a reaction.", ("interior_cut",), "all"),
    "dropoff": (
        "Your only job: DROP-OFF. Read as an impatient viewer. Mark the spans where you "
        "would click away, with why (dead stretch, repetition, nothing at stake). Use "
        "flag for a stretch that drags, cut only for a span that is pure dead weight.",
        ("dropped", "interior_cut"), "all"),
}


def realized(clips, spans, side: str = "all") -> str:
    """The cut as a viewer hears it. `side` = all | edges_start | edges_end."""
    by_clip: dict = {}
    for s in spans:
        by_clip.setdefault(s["clip"], []).append(s)
    lines = []
    for n, c in enumerate(clips, 1):
        sp = by_clip.get(c["i"], [])
        if side == "edges_start":
            sp = sp[:EDGE_SPANS]
        elif side == "edges_end":
            sp = sp[-EDGE_SPANS:]
        lines.append(f"== CLIP {n} ==")
        keep = c["keep"]
        prev = None
        for s in sp:
            if prev is not None:
                gone = sum(max(0.0, min(b, s["a"]) - max(a, prev["b"]))
                           for a, b in _gaps(keep))
                if gone >= 0.5:
                    lines.append(f"[CUT {gone:.1f}s]")
            lines.append(f"{s['id']} {s['text']}")
            prev = s
    return "\n".join(lines)


def _gaps(keep):
    return [(b, a2) for (_a, b), (a2, _b) in zip(keep, keep[1:])]


def _examples(lessons, labels, n: int = 3) -> str:
    from .learn.lessons import few_shot
    out = few_shot(lessons, labels, n)
    if not out:
        return ""
    return ("\nWhat this editor actually changed on past reFire cuts (calibrate to how "
            "rarely they act):\n" + "\n".join(out[:n * 2]))


def cli_critic(model: str, effort: str = DEFAULT_EFFORT, trace=None):
    """`ask(job, system, user) -> Edits` on Claude: CLI (subscription) first, API key next."""
    from . import director

    def ask(job, system, user):
        content = [{"type": "text", "text": user}]
        try:
            return director._complete_cli(model, system, content, Edits, tag=f"critic {job}",
                                          trace=trace, effort=effort, live=False)
        except (RuntimeError, OSError) as e:
            print(f"[critic {job}] claude CLI unavailable ({e}); using API key")
        import anthropic
        return director._complete(anthropic.Anthropic(), model, system, content, Edits,
                                  tag=f"critic {job}", trace=trace, effort=effort)
    return ask


def run_critics(clips, spans, ask, lessons=None) -> dict[str, list[Edit]]:
    """All four critics, in parallel. A failed critic returns no edits."""
    def one(job):
        text, labels, side = JOBS[job]
        try:
            return job, ask(job, _RULES + text + _examples(lessons, labels),
                            realized(clips, spans, side)).edits
        except Exception as e:                 # never lose the cut over a critic
            print(f"[critic {job}] failed ({e}); no edits")
            return job, []
    with ThreadPoolExecutor(max_workers=len(JOBS)) as ex:
        return dict(ex.map(one, JOBS))


def arbitrate(edits: dict[str, list[Edit]], spans) -> tuple[list[dict], list[dict]]:
    """(applied edits, conflicts). Conflict = cut vs keep/extend on one span."""
    ids = {s["id"] for s in spans}
    by: dict[str, list[tuple[str, Edit]]] = {}
    for job, es in edits.items():
        for e in es:
            if e.span_id in ids and e.confidence >= APPLY_CONF:
                by.setdefault(e.span_id, []).append((job, e))
    applied, conflicts = [], []
    for sid, es in by.items():
        acts = {e.action for _j, e in es}
        if "cut" in acts and acts & {"keep", "extend_start", "extend_end"}:
            conflicts.append({"span_id": sid, "critics": {j: f"{e.action}: {e.reason}"
                                                          for j, e in es}})
            continue
        for job, e in es:
            if e.action in ("cut", "extend_start", "extend_end", "flag"):
                applied.append({"span_id": sid, "action": e.action, "critic": job,
                                "reason": e.reason, "confidence": e.confidence})
    return applied, conflicts


def apply_edits(clips, spans, applied, conflicts, ctx, lessons=None, words=None,
                voiced=None):
    """Edits -> new clips (+ critic markers). Returns (clips, changed clip ids)."""
    from .boundary import DEFAULT_BACK_S, DEFAULT_TAIL_S, _cap, candidates
    from .learn.diff import subtract
    from .trim import cut_region
    back = _cap(lessons, "start_late", DEFAULT_BACK_S)
    tail = _cap(lessons, "end_early", DEFAULT_TAIL_S)
    sid = {s["id"]: s for s in spans}
    by_clip: dict = {}
    for s in spans:
        by_clip.setdefault(s["clip"], []).append(s)
    out = {c["i"]: dict(c) for c in clips}
    changed = set()
    cuts = sorted((e for e in applied if e["action"] == "cut"),
                  key=lambda e: -e["confidence"])[:int(MAX_CUT_FRAC * len(spans))]
    for e in applied:
        s = sid[e["span_id"]]
        c = out[s["clip"]]
        mk = None
        if e["action"] == "extend_start" and by_clip[s["clip"]][0] is s:
            cand = candidates(ctx, c["keep"][0][0], "start", back)
            if cand:
                c["start"] = min(c["start"], cand[-1])
                changed.add(c["i"])
                mk = "extended start"
        elif e["action"] == "extend_end" and by_clip[s["clip"]][-1] is s:
            cand = candidates(ctx, c["keep"][-1][1], "end", tail)
            if cand:
                c["end"] = max(c["end"], cand[0])
                changed.add(c["i"])
                mk = "extended end"
        elif e["action"] == "cut" and e in cuts:
            a, b = cut_region(s, by_clip, ctx.words)
            c["keep"] = [k for k in subtract(c["keep"], [[a, b]]) if k[1] - k[0] > 0.05]
            c["cut"] = (c.get("cut") or []) + [[a, b]]
            mk = "cut"
        elif e["action"] == "flag":
            mk = "viewer may click away"
        if mk:
            c["markers"] = (c.get("markers") or []) + [{
                "side": "critic", "kind": e["critic"], "vod": round(s["a"], 2),
                "risk": round(e["confidence"], 2), "name": f"reFire {e['critic']}: {mk}",
                "comment": e["reason"][:160]}]
    for cf in conflicts:
        s = sid[cf["span_id"]]
        c = out[s["clip"]]
        c["markers"] = (c.get("markers") or []) + [{
            "side": "critic", "kind": "conflict", "vod": round(s["a"], 2), "risk": 0.9,
            "name": "reFire: critics disagree",
            "comment": " | ".join(f"{j} {r}" for j, r in cf["critics"].items())[:200]}]
    res = [out[c["i"]] for c in clips]
    if changed and words is not None:
        from .learn.evaluate import rekeep
        res = rekeep(res, words, voiced, changed)
        for c in res:                          # a re-derived keep must still honour cuts
            if c["i"] in changed and c.get("cut"):
                c["keep"] = subtract(c["keep"], c["cut"]) or c["keep"]
    return res, changed


def review_clips(clips, ctx, lessons=None, ask=None, words=None, voiced=None,
                 rounds: int = MAX_ROUNDS):
    """The critic loop over clips [{i, start, end, keep}]. Returns (clips, log)."""
    from .trim import utterances
    log = {"rounds": []}
    for rnd in range(max(0, min(rounds, MAX_ROUNDS))):
        spans = utterances(clips, ctx.words, ctx=ctx)
        edits = run_critics(clips, spans, ask, lessons)
        applied, conflicts = arbitrate(edits, spans)
        clips, changed = apply_edits(clips, spans, applied, conflicts, ctx, lessons,
                                     words if words is not None else ctx.words, voiced)
        log["rounds"].append({"edits": {j: len(v) for j, v in edits.items()},
                              "applied": len(applied), "conflicts": len(conflicts)})
        if not any(e["action"] != "flag" for e in applied):
            break
    return clips, log


def replay_critics(prop, ctx, lessons, cache_dir, model=None, effort=DEFAULT_EFFORT):
    """`refire eval --critics`: the loop over a recorded cut (Claude calls, costs quota)."""
    from . import director
    from .learn.lessons import voiced_of
    ask = cli_critic(model or director.CLAUDE_MODEL, effort)
    return review_clips(prop, ctx, lessons, ask, voiced=voiced_of(cache_dir))


def review_sections(sections, words, ctx, lessons, voiced=None, pad: float = 0.3,
                    model=None, effort=DEFAULT_EFFORT, trace=None) -> dict:
    """`make --critics`: the loop over cast sections, written back in place."""
    from . import director
    from .select import compress_silence
    flat, refs = [], []
    for sec in sections:
        for c in sec["clips"]:
            k, _r, _d = compress_silence(words, c["start"], c["end"], pad=pad,
                                         voiced=voiced, exclude=c.get("cut"))
            flat.append({"i": len(flat), "start": c["start"], "end": c["end"],
                         "keep": [[c["start"] + x, c["start"] + y] for x, y in k],
                         "cut": c.get("cut")})
            refs.append(c)
    ask = cli_critic(model or director.CLAUDE_MODEL, effort, trace)
    new, log = review_clips(flat, ctx, lessons, ask, words=words, voiced=voiced)
    for c, n in zip(refs, new):
        c["start"], c["end"] = n["start"], n["end"]
        if n.get("cut"):
            c["cut"] = n["cut"]
        if n.get("markers"):
            c["markers"] = n["markers"]
    return log
