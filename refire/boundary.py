"""Boundary stage: move clip edges to where the moment actually starts and finishes.

Deterministic first, as candidate cut points from the transcript + signals:
  starts: sentence heads, speech-run heads after a real silence (>= THREAD_GAP_S)
  ends:   sentence ends, speech-run tails, the point loudness falls back near baseline
Rules (all opt-in, `--boundary` / style `boundary: true`):
  back  -- the clip opens mid-thought (speech runs straight into it) or on an unresolved
           referent ("he", "that", "wait"): pull the start back to the head of the
           thread, i.e. the nearest sentence head that follows a real silence
  tail  -- the reaction is still going at the out-point (loud AND speech continues, or
           chat still elevated): push the end to the end of that speech run / sentence,
           stopping where loudness decays
  merge -- VOD-adjacent clips closer than the merge gap become one continuous span
Every move is capped by the matching lesson from past edits (`clip_lessons_<game>.json`:
p75 of start_late / end_early), and the merge gap is the largest gap the human ever
merged (0 with no `merged` lessons -- so it never fires until an editor has done it).

Measured 2026-11 (4 finished edits, LOO): every blanket extension rule RAISED fix_cost,
because ~80% of edges are already where the editor left them, and the edges they did
move cluster in a few stretches rather than following any local signal. So the rules
fire only when two signals agree. Off by default; see MAKEOVER_REPORT.md.
"""
from __future__ import annotations

import re
from bisect import bisect_left

REF = re.compile(r"^(he|she|they|it|that|this|those|these|him|her|them|again|wait|"
                 r"i told you|like i said|because|cause|which)\b", re.I)
THREAD_GAP_S = 0.7        # a silence this long separates two thoughts
DECAY_Z = 0.5             # loudness z under this = back near baseline
LOUD_Z = 1.0
CHAT_Z = 1.5
DEFAULT_BACK_S = 6.0      # caps when no lessons exist (about the p75 measured 2026-11)
DEFAULT_TAIL_S = 8.0


def _cap(lessons: dict, label: str, default: float) -> float:
    v = ((lessons or {}).get("labels") or {}).get(label) or {}
    return float(v.get("p75_abs") or default) if v.get("count") else default


def merge_gap(lessons: dict) -> float:
    """Largest gap the editor ever bridged between two reFire clips (0 = never merges)."""
    ex = (((lessons or {}).get("labels") or {}).get("merged") or {})
    return float(ex.get("p75_abs") or 0.0) if ex.get("count") else 0.0


def sentence_heads(words) -> list[float]:
    out, prev_term, prev_end = [], True, None
    for w in words:
        if prev_term:
            out.append((w["start"], w["start"] - prev_end if prev_end is not None else 99.0))
        prev_term = w["text"][-1:] in ".?!"
        prev_end = w["end"]
    return out


def candidates(ctx, t: float, side: str, reach: float) -> list[float]:
    """Deterministic cut points within `reach` of t (before t for starts, after for ends)."""
    W, S = ctx.words, ctx.starts
    if side == "start":
        lo = bisect_left(S, t - reach)
        hi = bisect_left(S, t)
        out = []
        for k in range(lo, hi):
            w = W[k]
            gap = w["start"] - W[k - 1]["end"] if k > 0 else 99.0
            sent = k == 0 or W[k - 1]["text"][-1:] in ".?!"
            if gap >= THREAD_GAP_S or (sent and gap >= 0.3):
                out.append(w["start"])
        return out
    lo = bisect_left(S, t)
    hi = bisect_left(S, t + reach)
    out = []
    for k in range(max(0, lo - 1), min(hi, len(W))):
        w = W[k]
        if w["end"] <= t:
            continue
        nxt_gap = W[k + 1]["start"] - w["end"] if k + 1 < len(W) else 99.0
        if w["text"][-1:] in ".?!" or nxt_gap >= THREAD_GAP_S:
            out.append(w["end"])
    return out


def _z(ctx, sig, a, b):
    return (ctx._stat(sig, a, b) or {}).get("max") or 0.0


def refine(clips: list[dict], ctx, lessons: dict | None = None, snap: str = "sentence",
           keep_build: bool = True) -> tuple[list[dict], set, dict]:
    """clips [{i, start, end, keep, role?, section?}] -> (clips, changed i's, log).

    `snap="transient"` skips the tail rule (that style leaves before the moment
    resolves) and `keep_build=False` skips the back rule (it skips setup on purpose):
    the boundary stage must not undo the cut-point modes it sits next to."""
    lessons = lessons or {}
    back_cap = _cap(lessons, "start_late", DEFAULT_BACK_S)
    tail_cap = _cap(lessons, "end_early", DEFAULT_TAIL_S)
    gap_cap = merge_gap(lessons)
    W, S = ctx.words, ctx.starts
    out = [dict(c) for c in clips]
    order = sorted(range(len(out)), key=lambda k: out[k]["start"])
    log = {"back": 0, "tail": 0, "merge": 0, "moved_s": 0.0}
    changed = set()
    for pos, k in enumerate(order):
        c = out[k]
        if c.get("section") == "Cold Open":
            continue                       # a flash-forward is short on purpose
        prev_end = out[order[pos - 1]]["end"] if pos else 0.0
        next_start = out[order[pos + 1]]["start"] if pos + 1 < len(order) else 1e12
        ks, ke = c["keep"][0][0], c["keep"][-1][1]
        j = bisect_left(S, ks)
        prev = W[j - 1] if j > 0 else None
        first = " ".join(w["text"] for w in W[j:j + 3]).strip(" ,.!?\"'")
        mid_thought = prev is not None and ks - prev["end"] < 0.3
        if keep_build and (mid_thought and REF.match(first)):
            cands = [t for t in candidates(ctx, ks, "start", back_cap) if t > prev_end]
            if cands:
                t = cands[-1]              # nearest thread head before the cut
                log["back"] += 1
                log["moved_s"] += ks - t
                c["start"] = min(c["start"], t)
                changed.add(c["i"])
        j = bisect_left(S, ke)
        nxt = W[j] if j < len(W) else None
        speech_on = nxt is not None and nxt["start"] - ke < 0.3
        reacting = _z(ctx, ctx.loud, ke - 1, ke + 2) >= LOUD_Z and (
            speech_on or _z(ctx, ctx.chat, ke, ke + 8) >= CHAT_Z)
        if snap != "transient" and reacting:
            cands = [t for t in candidates(ctx, ke, "end", tail_cap) if t < next_start]
            # first candidate where the room has calmed down, else the last in reach
            calm = [t for t in cands if _z(ctx, ctx.loud, t - 1, t + 1) < DECAY_Z]
            t = (calm or cands or [None])[0]
            if t is not None and t > ke:
                log["tail"] += 1
                log["moved_s"] += t - ke
                c["end"] = max(c["end"], t)
                changed.add(c["i"])
    if gap_cap > 0:
        for a, b in zip(order, order[1:]):
            gap = out[b]["start"] - out[a]["end"]
            if 0 < gap <= gap_cap:
                out[a]["end"] = out[b]["start"]
                changed.add(out[a]["i"])
                log["merge"] += 1
    log["moved_s"] = round(log["moved_s"], 1)
    return out, changed, log


def refine_sections(sections, words, ctx, lessons, voiced=None, pad: float = 0.3,
                    snap: str = "sentence", keep_build: bool = True) -> dict:
    """`refine` over `narrative.cast` sections, in place (the `make` entry point).
    Visible edges come from the same compress_silence the manifest will apply."""
    from .select import compress_silence
    flat, refs = [], []
    for sec in sections:
        for c in sec["clips"]:
            k, _r, _d = compress_silence(words, c["start"], c["end"], pad=pad, voiced=voiced)
            flat.append({"i": len(flat), "start": c["start"], "end": c["end"],
                         "keep": [[c["start"] + x, c["start"] + y] for x, y in k],
                         "section": sec.get("title", "")})
            refs.append(c)
    new, _changed, log = refine(flat, ctx, lessons, snap=snap, keep_build=keep_build)
    for c, n in zip(refs, new):
        c["start"], c["end"] = n["start"], n["end"]
    return log
