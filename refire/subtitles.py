"""Build per-clip ASS subtitles with karaoke (word-by-word) highlighting.

Reuses Stage 1 word timestamps. Times are made 0-based relative to the clip so
the ASS can be burned onto a freshly-trimmed segment.
"""
from __future__ import annotations

from .emphasis import style_text
from .transcribe import Word

# 720p canvas, libass scales to the real frame. yellow highlight, white idle
_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Helvetica,48,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,5,1,2,60,60,70,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ts(seconds: float) -> str:
    """Seconds -> ASS time 'H:MM:SS.cs' (centiseconds)."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


PAUSE_GAP = 0.35      # seconds of silence that forces a new caption line
MAX_CARRY = 0.15      # max seconds a word's karaoke highlight may hold past its end
MAX_WORD_S = 0.7      # longest one spoken word is assumed to take, see _gap


def _gap(w: dict, nxt: dict) -> float:
    """Silence between two words, corrected for the ASR's word-end padding.

    faster-whisper does not leave gaps: it stretches each word's `end` to meet the next
    word's `start`, so `nxt["start"] - w["end"]` reads 0.0 across 74% of a real
    transcript -- including every audible pause, which is exactly where a caption line
    has to break. A held pause is billed to the word before it ('me' 400.78-402.14 is
    "me" plus a second of silence), so clamping the word to a plausible spoken length
    recovers the silence hiding inside it.

    Detection only -- displayed caption times keep the real timestamps.

    ponytail: MAX_WORD_S is the calibration knob, not a truth. Raise it if drawn-out
    words ("whaaaat") split lines too eagerly, lower it if pauses are still missed. A
    per-speaker estimate (median word length x xN) is the upgrade if one value can't
    cover both.
    """
    return nxt["start"] - min(w["end"], w["start"] + MAX_WORD_S)


def group_words(
    words: list[Word],
    seg_start: float,
    seg_end: float,
    words_per_line: int = 3,
) -> list[list[dict]]:
    """Words within [seg_start, seg_end) -> clip-relative, zero-based line groups.

    A line breaks on sentence-ending punctuation, on a pause >= PAUSE_GAP before
    the next word (measured by `_gap`, which sees through the ASR's word-end padding),
    or once it hits words_per_line. This keeps each line on a single phrase and stops a
    caption from hanging on screen across a silence -- or, worse, from opening with the
    tail of one phrase and running into the head of the next, which puts words on screen
    a full second before they are spoken.
    """
    clip_len = seg_end - seg_start
    sel = [
        {"text": w["text"],
         "start": min(max(w["start"] - seg_start, 0.0), clip_len),
         "end": min(max(w["end"] - seg_start, 0.0), clip_len),
         "emph": bool(w.get("emph"))}
        for w in words
        if seg_start <= w["start"] < seg_end and w["text"]
    ]
    groups: list[list[dict]] = []
    buf: list[dict] = []
    for i, w in enumerate(sel):
        buf.append(w)
        sentence_end = w["text"][-1:] in ".?!"
        nxt_gap = _gap(w, sel[i + 1]) if i + 1 < len(sel) else 0.0
        if sentence_end or nxt_gap >= PAUSE_GAP or len(buf) >= words_per_line:
            groups.append(buf)
            buf = []
    if buf:
        groups.append(buf)
    return groups


def build_ass(
    words: list[Word],
    seg_start: float,
    seg_end: float,
    words_per_line: int = 3,
) -> str:
    """Return ASS file text for words falling within [seg_start, seg_end)."""
    lines = []
    for group in group_words(words, seg_start, seg_end, words_per_line):
        line_start = group[0]["start"]
        line_end = group[-1]["end"]
        parts = []
        for j, w in enumerate(group):
            # highlight holds until the next word but never more than MAX_CARRY past this word's end,
            # otherwise it hangs over a pause
            nxt = group[j + 1]["start"] if j + 1 < len(group) else w["end"]
            hold_end = min(nxt, w["end"] + MAX_CARRY)
            kcs = max(1, round((hold_end - w["start"]) * 100))
            parts.append(f"{{\\k{kcs}}}{style_text(w['text'], w['emph'])} ")
        text = "".join(parts).strip()
        lines.append(
            f"Dialogue: 0,{_ts(line_start)},{_ts(line_end)},Default,,0,0,0,,{text}"
        )

    return _HEADER + "\n".join(lines) + ("\n" if lines else "")
