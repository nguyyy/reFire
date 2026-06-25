"""Build per-clip ASS subtitles with karaoke (word-by-word) highlighting.

Reuses Stage 1 word timestamps. Times are made 0-based relative to the clip so
the ASS can be burned onto a freshly-trimmed segment.
"""
from __future__ import annotations

from .transcribe import Word

# 720p canvas; libass scales to the actual frame. Yellow highlight, white idle.
_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,52,&H0000FFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,4,1,2,60,60,70,1

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


def build_ass(
    words: list[Word],
    seg_start: float,
    seg_end: float,
    words_per_line: int = 4,
) -> str:
    """Return ASS file text for words falling within [seg_start, seg_end)."""
    clip_len = seg_end - seg_start
    sel = [
        {"text": w["text"],
         "start": min(max(w["start"] - seg_start, 0.0), clip_len),
         "end": min(max(w["end"] - seg_start, 0.0), clip_len)}
        for w in words
        if seg_start <= w["start"] < seg_end and w["text"]
    ]

    lines = []
    for i in range(0, len(sel), words_per_line):
        group = sel[i:i + words_per_line]
        line_start = group[0]["start"]
        line_end = group[-1]["end"]
        parts = []
        for j, w in enumerate(group):
            # highlight holds until the next word begins (continuous karaoke)
            nxt = group[j + 1]["start"] if j + 1 < len(group) else w["end"]
            kcs = max(1, round((nxt - w["start"]) * 100))
            parts.append(f"{{\\k{kcs}}}{w['text']} ")
        text = "".join(parts).strip()
        lines.append(
            f"Dialogue: 0,{_ts(line_start)},{_ts(line_end)},Default,,0,0,0,,{text}"
        )

    return _HEADER + "\n".join(lines) + ("\n" if lines else "")
