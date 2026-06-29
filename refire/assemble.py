"""Concat rendered clips and mix an optional ducked music bed -> final.mp4.

Also hosts the Stage 2 orchestrator `edit()`: select -> render -> assemble.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .continuity import organize
from .emphasis import annotate_emphasis
from .render import render_clip
from .score import DEFAULT_MODEL
from .select import select_segments, snap_to_sentences, speech_intervals
from .subtitles import build_ass


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}):\n{proc.stderr[-2000:]}")


def assemble(
    clips: list[Path],
    out_path: str | Path,
    music: str | Path | None = None,
    music_db: float = -18.0,
) -> Path:
    """Concat clips (identical params -> stream copy), then mix music under."""
    out_path = Path(out_path)
    work = out_path.parent
    listing = work / "concat.txt"
    listing.write_text(
        "".join(f"file '{c.resolve().as_posix()}'\n" for c in clips),
        encoding="utf-8",
    )

    if not music:
        _run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(listing),
              "-c", "copy", "-y", str(out_path)])
        return out_path

    joined = work / "_joined.mp4"
    _run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(listing),
          "-c", "copy", "-y", str(joined)])
    # loop music, lower it, mix under the joined audio, cut to video length
    _run([
        "ffmpeg", "-i", str(joined), "-stream_loop", "-1", "-i", str(music),
        "-filter_complex",
        f"[1:a]volume={music_db}dB[m];[0:a][m]amix=inputs=2:duration=first[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
        "-y", str(out_path),
    ])
    joined.unlink(missing_ok=True)   # drop the music-less intermediate
    return out_path


def render_clips(
    video: str | Path,
    run_dir: str | Path,
    clips: list[dict],
    words: list[dict],
    music: str | Path | None = None,
    encoder: str = "libx264",
    out_name: str = "rough.mp4",
) -> Path:
    """Render a flat list of {start,end} clips into a watchable rough cut.

    Per-clip trim -> reframe -> burn subs, concat, optional music bed. `words` must
    already be emphasis-annotated. Reuses the ffmpeg path so a `make` run can be
    eyeballed without After Effects; skips AE-only polish (section cards, emote
    overlays, SFX). Returns run_dir/out_name.
    """
    run_dir = Path(run_dir)
    speech = speech_intervals(words)            # phrase runs; hold zoom through these
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for i, seg in enumerate(clips):
        ass = clips_dir / f"clip{i:03d}.ass"
        out = clips_dir / f"clip{i:03d}.mp4"
        ass.write_text(build_ass(words, seg["start"], seg["end"]), encoding="utf-8")
        seg_speech = [(max(s, seg["start"]) - seg["start"], min(e, seg["end"]) - seg["start"])
                      for s, e in speech if e > seg["start"] and s < seg["end"]]
        render_clip(video, seg, ass, out, encoder=encoder, speech=seg_speech)
        rendered.append(out)
    return assemble(rendered, run_dir / out_name, music=music)


def edit(
    video: str | Path,
    run_dir: str | Path = "run",
    music: str | Path | None = None,
    count: int | None = None,
    min_score: float | None = None,
    order: str = "chrono",
    encoder: str = "libx264",
    topic: str = "",
    model: str = DEFAULT_MODEL,
) -> Path:
    """Stage 2 end-to-end. Requires run_dir/segments.json + transcript.json."""
    run_dir = Path(run_dir)
    segments = json.loads((run_dir / "segments.json").read_text(encoding="utf-8"))
    words = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
    annotate_emphasis(words, run_dir / "audio.wav")  # lowercase + caps-on-hype

    chosen = select_segments(segments, count=count, min_score=min_score, order=order)
    if not chosen:
        raise SystemExit("No segments selected (check --count/--min-score).")

    # snap onto sentence boundaries, then let the topic organize narrative order
    snapped = []
    for seg in chosen:
        s, e = snap_to_sentences(words, seg["start"], seg["end"])
        snapped.append({**seg, "start": s, "end": e})
    chosen = [c for sec in organize(snapped, topic, model) for c in sec["clips"]]

    speech = speech_intervals(words)        # phrase runs; hold zoom through these
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    for i, seg in enumerate(chosen):
        ass = clips_dir / f"clip{i:03d}.ass"
        out = clips_dir / f"clip{i:03d}.mp4"
        ass.write_text(build_ass(words, seg["start"], seg["end"]), encoding="utf-8")
        # clip-relative speech so the zoom never releases mid-sentence
        seg_speech = [(max(s, seg["start"]) - seg["start"], min(e, seg["end"]) - seg["start"])
                      for s, e in speech if e > seg["start"] and s < seg["end"]]
        # ponytail: always re-render; a clip cache silently ships stale styling
        # on re-runs. Add a content-hash cache only if re-encode cost ever bites.
        render_clip(video, seg, ass, out, encoder=encoder, speech=seg_speech)
        clips.append(out)

    return assemble(clips, run_dir / "final.mp4", music=music)
