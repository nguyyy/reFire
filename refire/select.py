"""Duration parsing, budget selection, and sentence/silence-aware clip boundaries."""
from __future__ import annotations


def parse_duration(text) -> float:
    """'20m' / '90s' / '1.5h' / '20:00' / 'hh:mm:ss' / bare seconds -> seconds."""
    t = str(text).strip().lower()
    if ":" in t:                       # mm:ss or hh:mm:ss
        sec = 0.0
        for part in t.split(":"):
            sec = sec * 60.0 + float(part)
        return sec
    mult = 1.0
    if t.endswith("h"):
        mult, t = 3600.0, t[:-1]
    elif t.endswith("m"):
        mult, t = 60.0, t[:-1]
    elif t.endswith("s"):
        mult, t = 1.0, t[:-1]
    return float(t) * mult


def budget_select(scored, target_s: float, tol: float = 0.35, order: str = "chrono"):
    """Greedily pick highest-scored clips until total duration reaches target_s.

    Items need start/end/score; durations are measured as end-start, so callers
    that snap to sentences should pass POST-snap bounds for an accurate budget.
    Stops once the target is met. If the pool can't reach the lower tolerance band
    it returns what it has plus a warning string (never pads with filler).
    Returns (clips, warning|None); clips ordered chrono (default) or by score.
    """
    lo = target_s * (1.0 - tol)
    picked, total = [], 0.0
    for s in sorted(scored, key=lambda x: x["score"], reverse=True):
        if total >= target_s:
            break
        picked.append(s)
        total += s["end"] - s["start"]
    warning = None
    if total < lo:
        warning = (f"only {total:.0f}s of material cleared selection "
                   f"(target {target_s:.0f}s) -- not padding with filler.")
    if order == "chrono":
        picked.sort(key=lambda s: s["start"])
    else:
        picked.sort(key=lambda s: s["score"], reverse=True)
    return picked, warning


def snap_to_sentences(words, start: float, end: float, max_pad: float = 5.0,
                      phrase_fallback: bool = False):
    """Nudge a clip's [start, end] onto sentence boundaries so it never cuts mid-sentence.

    Pulls `start` back to the first word of its sentence and extends `end` forward to
    the next word ending in .?!, each capped by `max_pad` seconds so we never drag in
    unrelated audio. Words are {text,start,end} in absolute seconds.

    `phrase_fallback` (used for the final clip, where ending mid-word is most visible):
    when whisper dropped the terminal punctuation so no .?! lands within `max_pad`, end
    on the nearest natural pause instead -- the end of the speech run at/after `end`
    (a `speech_intervals` boundary), still capped by `max_pad`. Off => byte-identical.
    """
    # start of the sentence that `start` falls in (word after the previous terminator)
    sent_start = None
    prev_terminated = True               # first word always begins a sentence
    for w in words:
        if w["start"] > start:
            break
        if prev_terminated:
            sent_start = w["start"]
        prev_terminated = w["text"][-1:] in ".?!"
    new_start = sent_start if (sent_start is not None and start - sent_start <= max_pad) else start

    # end of the sentence that `end` falls in (first terminator at/after end)
    new_end = None
    for w in words:
        if w["end"] >= end and w["text"][-1:] in ".?!":
            if w["end"] - end <= max_pad:
                new_end = w["end"]
            break
    if new_end is None and phrase_fallback:
        # no punctuation in reach -> land on a clean breath (next speech-run boundary)
        for _s, e in speech_intervals(words):
            if e >= end and e - end <= max_pad:
                new_end = e
                break
    if new_end is None:
        new_end = end

    return max(0.0, new_start), max(new_end, new_start)


def speech_intervals(words, max_gap: float = 0.4) -> list[tuple[float, float]]:
    """Merge words into speaking runs, splitting on silence >= max_gap (seconds).

    A run is one continuous phrase; the gaps between runs are the natural pauses
    where a zoom may release without cutting off mid-sentence. Absolute seconds.
    """
    runs: list[list[float]] = []
    for w in words:
        if runs and w["start"] - runs[-1][1] < max_gap:
            runs[-1][1] = max(runs[-1][1], w["end"])
        else:
            runs.append([w["start"], w["end"]])
    return [(s, e) for s, e in runs]


SILENCE_PAD = 0.3   # the "short reasonable breath" left around each phrase


def compress_silence(words, start: float, end: float, pad: float = SILENCE_PAD):
    """Clip [start, end] -> (keep, retimed, dur) with internal dead air removed.

    Speech runs (`speech_intervals`) are each padded by `pad` on both sides, clamped
    to the clip, and merged where the padded spans touch -- so a gap smaller than
    ~2*pad survives untouched (nothing worth cutting) while a real silence collapses
    to a `pad`-sized breath on each side. No speech -> the whole clip is kept (no-op).

    Returns:
      keep    : clip-relative spans [(a, b)...] for the ffmpeg select filter
      retimed : words on the gap-free, 0-based clip timeline (carry `emph`)
      dur     : compressed clip length (sum of kept span lengths)
    """
    runs = [(s, e) for s, e in speech_intervals(words)
            if e > start and s < end]
    if not runs:
        retimed = [{"text": w["text"], "start": w["start"] - start,
                    "end": w["end"] - start, "emph": bool(w.get("emph"))}
                   for w in words if start <= w["start"] < end]
        return [(0.0, end - start)], retimed, end - start

    # pad + clamp to clip, then merge spans that touch after padding
    merged: list[list[float]] = []
    for s, e in runs:
        a, b = max(start, s - pad), min(end, e + pad)
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    keep = [(a - start, b - start) for a, b in merged]   # clip-relative
    # cumulative output offset where each kept span begins on the tight timeline
    offsets, acc = [], 0.0
    for a, b in keep:
        offsets.append(acc)
        acc += b - a
    dur = acc

    retimed = []
    for w in words:
        wc = w["start"] - start                          # clip-relative word start
        for (a, b), off in zip(keep, offsets):
            if a <= wc < b:
                retimed.append({"text": w["text"],
                                "start": off + (wc - a),
                                "end": off + min(b, w["end"] - start) - a,
                                "emph": bool(w.get("emph"))})
                break
    return keep, retimed, dur


def _demo() -> None:
    ws = [{"text": t, "start": i * 1.0, "end": i * 1.0 + 0.8}
          for i, t in enumerate("Today we review builds. Alright lets go now.".split())]
    # contiguous 0.2s gaps -> one speech run spanning all words
    si = speech_intervals(ws)
    assert len(si) == 1 and si[0] == (0.0, ws[-1]["end"]), si
    # a 1s pause splits into two runs
    ws2 = ws + [{"text": "later", "start": ws[-1]["end"] + 1.0, "end": ws[-1]["end"] + 1.5}]
    assert len(speech_intervals(ws2)) == 2, speech_intervals(ws2)
    # "builds." ends at index 3 (end 3.8); cut mid-sentence at 2.5..3.0 -> snap out
    s, e = snap_to_sentences(ws, 2.5, 3.0)
    assert s == 0.0, s              # pulled back to sentence start "Today"
    assert e == 3.8, e             # extended to "builds." end
    # cap respected: huge gap -> no snap
    s2, e2 = snap_to_sentences(ws, 2.5, 3.0, max_pad=0.1)
    assert (s2, e2) == (2.5, 3.0), (s2, e2)
    # phrase_fallback: no terminal punctuation -> end on the next speech-run pause
    npw = [{"text": "uh", "start": 0.0, "end": 0.5},      # no .?! anywhere
           {"text": "wait", "start": 0.5, "end": 1.0},    # run ends here (then a 2s gap)
           {"text": "what", "start": 3.0, "end": 3.5}]
    _s, e3 = snap_to_sentences(npw, 0.8, 0.9, phrase_fallback=True)
    assert e3 == 1.0, e3                                  # snapped out to the pause at 1.0
    _s, e4 = snap_to_sentences(npw, 0.8, 0.9)             # off -> unchanged end
    assert e4 == 0.9, e4
    print("snap_to_sentences ok")

    # compress_silence: two phrases split by a long dead gap -> gap collapses to ~2*pad
    cw = [{"text": "hi", "start": 10.0, "end": 10.5},
          {"text": "there", "start": 10.5, "end": 11.0},
          {"text": "again", "start": 20.0, "end": 20.5}]   # 9s of dead air at 11..20
    keep, retimed, dur = compress_silence(cw, 10.0, 21.0, pad=0.3)
    assert len(keep) == 2, keep                            # the dead gap was cut
    assert dur < 11.0 - 9.0 + 1.0, dur                     # ~2.0..2.6s, not the full 11s
    assert abs(dur - sum(b - a for a, b in keep)) < 1e-9   # dur == kept length
    assert [w["start"] for w in retimed] == sorted(w["start"] for w in retimed)  # monotonic
    assert retimed[-1]["start"] < dur, retimed             # last word lands inside the cut
    # no speech -> whole clip kept, no-op
    k2, r2, d2 = compress_silence([], 0.0, 5.0)
    assert k2 == [(0.0, 5.0)] and d2 == 5.0, (k2, d2)
    print("compress_silence ok")


if __name__ == "__main__":
    _demo()
