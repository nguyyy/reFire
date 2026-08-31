"""Concat rendered clips and mix an optional ducked music bed -> rough.mp4."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .reframe import ENTER, EXIT
from .render import render_clip
from .select import SILENCE_PAD, compress_silence, speech_intervals
from .style import style_for
from .subtitles import build_ass


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
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
    silence_pad: float = SILENCE_PAD,
    motion_zoom: bool = True,
    deadspace: bool = True,
    loudnorm: bool = False,
) -> Path:
    """Render a flat list of {start,end} clips into a watchable rough cut.

    With `deadspace`, per-clip dead air is removed (`compress_silence`, leaving
    `silence_pad` of breath around each phrase) so the cut plays tight; captions + zoom
    ride the compressed timeline. Then trim -> reframe -> burn subs, concat, optional music bed. `words`
    must already be emphasis-annotated. Reuses the ffmpeg path so a `make` run can be
    eyeballed without After Effects; skips AE-only polish (section cards, emote
    overlays, SFX). Returns run_dir/out_name.
    """
    run_dir = Path(run_dir)
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for i, seg in enumerate(clips):
        ass = clips_dir / f"clip{i:03d}.ass"
        out = clips_dir / f"clip{i:03d}.mp4"
        # Style pass: a beat's role tilts caption pacing + zoom punchiness here too, so the
        # rough cut previews the same story-driven style as the AE build. No role -> defaults.
        role = seg.get("role", "")
        if role:
            st = style_for(role, seg.get("energy", 3))
            wpl = st["words_per_line"]
            c_enter = min(0.95, ENTER / st["zoom_sens"])
            c_exit = min(c_enter * 0.9, EXIT / st["zoom_sens"])
        else:
            wpl, c_enter, c_exit = 3, ENTER, EXIT
        if deadspace:
            keep, retimed, cdur = compress_silence(words, seg["start"], seg["end"],
                                                   pad=silence_pad)
        else:                                   # keep=None -> ffmpeg trims, cuts nothing
            keep, cdur = None, seg["end"] - seg["start"]
            retimed = [{"text": w["text"], "start": w["start"] - seg["start"],
                        "end": w["end"] - seg["start"], "emph": bool(w.get("emph"))}
                       for w in words if seg["start"] <= w["start"] < seg["end"]]
        ass.write_text(build_ass(retimed, 0.0, cdur, wpl), encoding="utf-8")
        seg_speech = speech_intervals(retimed)  # phrase runs on the clip timeline
        render_clip(video, seg, ass, out, encoder=encoder, speech=seg_speech, keep=keep,
                    motion_zoom=motion_zoom, enter=c_enter, exit=c_exit,
                    loudnorm=loudnorm)
        rendered.append(out)
    return assemble(rendered, run_dir / out_name, music=music)
