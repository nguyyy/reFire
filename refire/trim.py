"""Intra-clip trim: cut the filler inside kept clips, never a word and never a setup.

1. split each clip's kept footage into utterances (speech runs, silence-separated)
2. label every span setup|payoff|reaction|context|filler|dead_air|chat_reading|off_topic
   and list which LATER spans depend on it -- a batched local-LLM call, span ids in,
   JSON out (ollama: this is bulk work, never a per-span Claude call)
3. removal test, deterministic: a span goes only if its label is cuttable AND no later
   kept span depends on it AND its energy is low (loudness z < LOW_ENERGY_Z)
4. cut points sit in the silence between words, so a trim can never clip one; the
   pause right before a loud span (the comedic beat before a punchline) is kept
5. aggressiveness is capped at the editor's own rate: in the 2026-11 finished edits they
   removed 36 of 992 utterances (3.6%) inside kept clips. Lessons' interior_cut examples
   are shown to the judge as what this editor actually cuts.

Dead air itself is already handled by select.compress_silence. Off by default
(`--trim` / style `trim: true`) until `refire eval --trim` shows it helps.
"""
from __future__ import annotations

import json

from .select import speech_intervals

LABELS = ("setup", "payoff", "reaction", "context", "filler", "dead_air",
          "chat_reading", "off_topic")
CUTTABLE = {"filler", "dead_air", "chat_reading", "off_topic"}
SPLIT_GAP = 0.6         # silence that separates two utterances
LOW_ENERGY_Z = 1.0      # louder than this is a reaction, never filler
PUNCH_Z = 1.5           # the pause before a span this loud is comedic timing, keep it
MIN_CONF = 0.6
MAX_CUT_FRAC = 0.05     # editor's measured rate (3.6%), with a little room
BATCH = 60
EDGE = 0.05             # cut this far inside the silence next to a word, never into it

_SYSTEM = (
    "You are an assistant editor tightening a rough cut of a Twitch stream. You get the "
    "cut's utterances in play order, one per line: `S<n> [<seconds>s, loud <z>] text`. "
    "Label every span with exactly one of: setup (context the moment needs), payoff (the "
    "joke/reveal/win/fail lands), reaction (the streamer or friends reacting), context "
    "(explains what is going on), filler (stalling, rambling, verbal tics, repeated "
    "lines), dead_air (nothing meaningful said), chat_reading (reading chat or donations "
    "aloud), off_topic (setup/loading/tech talk, tangents unrelated to the moment). Then "
    "list `needed_by`: the ids of LATER spans that would stop making sense if this span "
    "were cut (an answer needs its question; a reaction needs what it reacts to). Be "
    "strict: most spans in a cut an editor already chose are needed. Reply ONLY with JSON: "
    '{"spans": [{"id": "S1", "label": "<label>", "needed_by": ["S3"], '
    '"confidence": <0-1>}, ...]} with one entry per span.')


def utterances(clips, words, gap: float = SPLIT_GAP, ctx=None) -> list[dict]:
    """Kept footage of every clip -> [{id, clip, a, b, text, loud}] in play order.
    `loud` (max loudness z over the span) needs `ctx`, else 0."""
    from bisect import bisect_left
    starts = [w["start"] for w in words]
    out = []
    for c in clips:
        for ka, kb in c["keep"]:
            ws = words[bisect_left(starts, ka):bisect_left(starts, kb)]
            for a, b in speech_intervals(ws, max_gap=gap):
                text = " ".join(w["text"] for w in ws if a <= w["start"] < b)
                loud = ((ctx._stat(ctx.loud, a, b) or {}).get("max") or 0.0) if ctx else 0.0
                out.append({"id": f"S{len(out) + 1}", "clip": c["i"], "a": a, "b": b,
                            "text": text, "loud": loud})
    return out


def _examples(lessons: dict, n: int = 4) -> str:
    from .learn.lessons import few_shot
    lines = few_shot(lessons, ("interior_cut",), n)
    return ("\nSpans this editor actually cut in past videos (calibrate to these, they "
            "remove only a few % of lines):\n" + "\n".join(lines)) if lines else ""


def ollama_judge(model: str):
    """`judge(system, user) -> dict` on the local scorer model, JSON mode."""
    def judge(system, user):
        import ollama
        resp = ollama.chat(model=model, format="json",
                           messages=[{"role": "system", "content": system},
                                     {"role": "user", "content": user}],
                           options={"num_ctx": 16384, "temperature": 0})
        return json.loads(resp["message"]["content"])
    return judge


def label(spans, judge, lessons=None, batch: int = BATCH) -> dict:
    """{span id: {label, needed_by, confidence}}. A failed batch labels nothing (= keep)."""
    system = _SYSTEM + _examples(lessons)
    out = {}
    for k in range(0, len(spans), batch):
        chunk = spans[k:k + batch]
        user = "\n".join(f"{s['id']} [{s['b'] - s['a']:.1f}s, loud {s['loud']:.1f}] "
                         f"{s['text']}" for s in chunk)
        try:
            got = judge(system, user)
        except Exception as e:                  # never lose a cut over a trim hiccup
            print(f"[trim] judge failed on spans {k + 1}-{k + len(chunk)} ({e}); keeping them")
            continue
        for row in (got or {}).get("spans") or []:
            if isinstance(row, dict) and row.get("id"):
                out[str(row["id"])] = {
                    "label": str(row.get("label", "")).lower(),
                    "needed_by": [str(x) for x in row.get("needed_by") or []],
                    "confidence": float(row.get("confidence") or 0.0)}
    return out


def removable(spans, labels, max_frac: float = MAX_CUT_FRAC) -> list[dict]:
    """The removal test, then the editor's rate cap (highest confidence first)."""
    later_needs = {}
    for s in spans:
        for dep in labels.get(s["id"], {}).get("needed_by") or []:
            later_needs.setdefault(s["id"], []).append(dep)
    cands = []
    for s in spans:
        lab = labels.get(s["id"])
        if not lab or lab["label"] not in CUTTABLE or lab["confidence"] < MIN_CONF:
            continue
        if later_needs.get(s["id"]) or s["loud"] >= LOW_ENERGY_Z:
            continue
        cands.append((lab["confidence"], s))
    cands.sort(key=lambda t: -t[0])
    return [s for _c, s in cands[:int(max_frac * len(spans))]]


def cut_region(span, spans_by_clip, words) -> list[float]:
    """[a, b] to remove for one span: from just after the previous word to just before
    the next, so no word is clipped -- except the pause before a loud next span stays."""
    sib = spans_by_clip[span["clip"]]
    k = sib.index(span)
    prev_end = sib[k - 1]["b"] if k > 0 else None
    nxt = sib[k + 1] if k + 1 < len(sib) else None
    a = max(span["a"] - EDGE, prev_end + EDGE) if prev_end is not None else span["a"] - EDGE
    if nxt is not None and nxt["loud"] < PUNCH_Z:
        b = max(span["b"] + EDGE, nxt["a"] - EDGE)
    else:
        b = span["b"] + EDGE
    return [a, b]


def trim_clips(clips, ctx, lessons=None, voiced=None, judge=None, model=None,
               max_frac: float = MAX_CUT_FRAC):
    """clips [{i, start, end, keep}] -> (clips with filler subtracted from keep, log)."""
    from .learn.diff import subtract
    if judge is None:
        from .pipeline import _resolve_ollama_model
        from .score import DEFAULT_MODEL
        judge = ollama_judge(_resolve_ollama_model(model or DEFAULT_MODEL))
    spans = utterances(clips, ctx.words, ctx=ctx)
    labels = label(spans, judge, lessons)
    gone = removable(spans, labels, max_frac)
    by_clip = {}
    for s in spans:
        by_clip.setdefault(s["clip"], []).append(s)
    cuts = {}
    for s in gone:
        cuts.setdefault(s["clip"], []).append(cut_region(s, by_clip, ctx.words))
    out = []
    for c in clips:
        if c["i"] in cuts:
            keep = [k for k in subtract(c["keep"], cuts[c["i"]]) if k[1] - k[0] > 0.05]
            c = {**c, "keep": keep or c["keep"], "cut": cuts[c["i"]],
                 "markers": (c.get("markers") or []) + [
                     {"side": "trim", "vod": round(a, 2), "risk": 0.5,
                      "name": "reFire: trimmed",
                      "comment": f"trimmed {b - a:.1f}s of filler -- check the flow"}
                     for a, b in cuts[c["i"]]]}
        out.append(c)
    log = {"spans": len(spans), "labelled": len(labels), "cut": len(gone),
           "cut_s": round(sum(b - a for v in cuts.values() for a, b in v), 1),
           "by_label": {lab: sum(1 for v in labels.values() if v["label"] == lab)
                        for lab in LABELS}}
    return out, log


def trim_sections(sections, words, ctx, lessons, voiced=None, pad: float = 0.3,
                  model=None) -> dict:
    """`trim_clips` over cast sections (the `make` entry point): writes each clip's
    `cut` spans, which select.compress_silence(exclude=) removes in every renderer."""
    from .select import compress_silence
    flat, refs = [], []
    for sec in sections:
        for c in sec["clips"]:
            k, _r, _d = compress_silence(words, c["start"], c["end"], pad=pad, voiced=voiced)
            flat.append({"i": len(flat), "start": c["start"], "end": c["end"],
                         "keep": [[c["start"] + x, c["start"] + y] for x, y in k]})
            refs.append(c)
    new, log = trim_clips(flat, ctx, lessons, voiced=voiced, model=model)
    for c, n in zip(refs, new):
        if n.get("cut"):
            c["cut"] = n["cut"]
    return log
