"""Render one finished clip: trim -> action-aware dynamic reframe -> burn subs.

All clips end at identical params (1280x720, 30fps, yuv420p, aac, no data stream)
so the assemble step can concat them with -c copy.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .reframe import ENTER, EXIT, reframe_clip

W, H, FPS = 1280, 720, 30

# EBU R128 target for --style loudnorm. -16 LUFS is the usual web level and leaves
# headroom for the screams we keep on purpose
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}):\n{proc.stderr[-2000:]}")


def _ass_filter_path(ass_path: Path) -> str:
    """Escape a path for use inside ffmpeg's subtitles= filter (Windows-safe)."""
    p = str(ass_path.resolve()).replace("\\", "/")
    return p.replace(":", r"\:")  # escape drive-letter colon


def render_clip(
    video: str | Path,
    seg: dict,          # needs start/end, role/energy optional for styling
    ass_path: str | Path,
    out_path: str | Path,
    encoder: str = "libx264",
    speech=None,
    keep=None,
    motion_zoom: bool = True,
    enter: float = ENTER,
    exit: float = EXIT,
    loudnorm: bool = False,
) -> Path:
    """Trim [seg.start, seg.end], dynamic reframe, burn subtitles. Raises on fail.

    `speech` is clip-relative speaking runs; passed to the reframe so the zoom
    holds through speech and releases at a pause. `keep`, if given, is a list of
    clip-relative spans [(a, b)...] to retain -- the gaps between them (dead air)
    are dropped from the trim so the clip plays tight. Callers passing `keep` must
    feed captions/`speech` on the matching compressed timeline. `motion_zoom=False`
    skips OpenCV motion analysis and uses a static bottom-left fit/crop.

    `loudnorm` normalizes this clip to a fixed perceived loudness. A cut that jumps
    between moments recorded minutes apart also jumps in level, and once shots are short
    the level jump lands on every cut -- audible as fatigue rather than as energy. Off by
    default: it costs an audio filter pass and changes the mix, so only a style that
    asks for it gets it.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dur = seg["end"] - seg["start"]
    work = out_path.with_name(out_path.stem + "_work.mp4")
    reframed = out_path.with_name(out_path.stem + "_reframed.mp4")

    # A: trim + normalize fps, keep source res (reframe crops from it) and audio.
    # if keep cuts dead air, select+setpts drops the gaps in one encode. one full span skips it
    cut, afilters = [], []
    if keep and not (len(keep) == 1 and keep[0][0] <= 1e-3 and keep[0][1] >= dur - 1e-3):
        sel = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in keep)
        cut = ["-vf", f"select='{sel}',setpts=N/FRAME_RATE/TB"]
        afilters += [f"aselect='{sel}'", "asetpts=N/SR/TB"]
    if loudnorm:
        # single pass, two-pass needs a measuring run per clip and doesn't help at these lengths
        afilters.append(LOUDNORM)
    if afilters:
        cut += ["-af", ",".join(afilters)]
    _run(["ffmpeg", "-ss", str(seg["start"]), "-t", str(dur), "-i", str(video),
          *cut, "-r", str(FPS), "-c:v", encoder, "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-ar", "48000", "-ac", "2", "-dn", "-y", str(work)])

    # B: reframe to the output canvas. motion zoom is optional, it can eat render time on long cuts
    if motion_zoom:
        reframe_clip(work, reframed, out_w=W, out_h=H, speech=speech, enter=enter, exit=exit)
    else:
        _run(["ffmpeg", "-i", str(work), "-vf",
              f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}:0:ih-{H}",
              "-an", "-r", str(FPS), "-c:v", encoder, "-pix_fmt", "yuv420p",
              "-dn", "-y", str(reframed)])

    # C: burn subs on the reframed video, audio from the work clip
    _run(["ffmpeg", "-i", str(reframed), "-i", str(work),
          "-filter_complex", f"[0:v]subtitles='{_ass_filter_path(Path(ass_path))}'[v]",
          "-map", "[v]", "-map", "1:a", "-r", str(FPS),
          "-c:v", encoder, "-pix_fmt", "yuv420p", "-c:a", "aac",
          "-dn", "-shortest", "-y", str(out_path)])

    work.unlink(missing_ok=True)
    reframed.unlink(missing_ok=True)
    return out_path
