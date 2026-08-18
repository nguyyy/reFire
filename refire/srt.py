"""Manifest captions -> one .srt in master-timeline time (the Premiere path).

After Effects gets its captions as real text layers built by `refire/ae/reFire.jsx`.
Premiere has no scriptable text layers, so the Premiere panel (`refire/ppro/`) reads
this .srt instead. Cue times are computed with the SAME layout rule the panel's JSX
uses to lay clips on V1 -- clips concatenated in section order, each contributing
`dur` (dead-air-tightened) or its raw span -- so the two line up by construction.

Straight cuts only: unlike the AE Master there is no crossfade to subtract, because
the Premiere build is clip selection + subtitling and nothing else.
"""
from __future__ import annotations

import json
from pathlib import Path


def _ts(seconds: float) -> str:
    """Seconds -> SRT time 'HH:MM:SS,mmm'."""
    seconds = max(0.0, seconds)
    ms = int(round(seconds * 1000.0))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ordered_clips(manifest):
    """Clips in the order the sequence lays them out. Mirrors reFire.jsx:buildMaster."""
    clips = manifest.get("clips") or []
    sections = manifest.get("sections") or []
    if not sections:
        return list(clips)
    out = []
    for sec in sections:
        for i in sec.get("clip_indices") or []:
            if 0 <= i < len(clips):
                out.append(clips[i])
    return out or list(clips)


def build_srt(manifest, offset: float = 0.0) -> str:
    """Manifest dict -> SRT text. `offset` shifts every cue (the panel's nudge)."""
    cues = []
    playhead = 0.0
    for clip in _ordered_clips(manifest):
        # `dur` is the tightened length when dead air was cut, else the raw span --
        # the same choice reFire.jsx:buildClip makes for its comp duration.
        dur = clip.get("dur") or (clip["end"] - clip["start"])
        for cap in clip.get("captions") or []:
            text = (cap.get("text") or "").strip()
            if not text:
                continue
            # clamp into the clip: a caption whose word ran past the cut would
            # otherwise sit over the NEXT clip on the timeline.
            a = min(max(cap["start"], 0.0), dur)
            b = min(max(cap["end"], a), dur)
            if b <= a:
                continue
            cues.append((playhead + a + offset, playhead + b + offset, text))
        playhead += dur

    return "".join(
        f"{i}\n{_ts(a)} --> {_ts(b)}\n{text}\n\n"
        for i, (a, b, text) in enumerate(cues, 1)
    )


def write_srt(manifest_path, offset: float = 0.0) -> Path:
    """Read manifest.json, write captions.srt beside it. Returns the .srt path."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = manifest_path.parent / "captions.srt"
    out.write_text(build_srt(manifest, offset), encoding="utf-8")
    return out


def _demo() -> None:
    m = {
        "clips": [
            {"start": 100.0, "end": 110.0,
             "captions": [{"text": "first", "start": 0.5, "end": 1.5},
                          {"text": "second", "start": 2.0, "end": 3.0}]},
            # dead air cut: 20s span plays as 6s, so cue times must use `dur`
            {"start": 200.0, "end": 220.0, "dur": 6.0, "keep": [[0, 3], [10, 13]],
             "captions": [{"text": "third", "start": 4.0, "end": 5.5},
                          {"text": "ran past the cut", "start": 5.0, "end": 99.0}]},
        ],
        "sections": [{"title": "b", "clip_indices": [1]},
                     {"title": "a", "clip_indices": [0]}],
    }
    # section order wins over clip order: clip 1 (dur 6) lays down first
    out = build_srt(m)
    assert out.startswith("1\n00:00:04,000 --> 00:00:05,500\nthird\n"), out
    # a caption overrunning the cut is clamped to the clip's tightened length
    assert "00:00:05,000 --> 00:00:06,000\nran past the cut" in out, out
    # clip 0 starts at clip 1's dur (6.0), not at its raw 20s span
    assert "00:00:06,500 --> 00:00:07,500\nfirst" in out, out
    assert "00:00:08,000 --> 00:00:09,000\nsecond" in out, out
    assert out.count(" --> ") == 4

    # no sections -> plain clip order; offset shifts every cue
    flat = build_srt({"clips": m["clips"]}, offset=0.25)
    assert flat.startswith("1\n00:00:00,750 --> 00:00:01,750\nfirst\n"), flat
    # negative offset clamps at zero rather than emitting a negative timestamp
    assert build_srt({"clips": m["clips"]}, offset=-9.0).startswith(
        "1\n00:00:00,000 --> 00:00:00,000\nfirst\n")
    print("srt ok")


if __name__ == "__main__":
    _demo()
