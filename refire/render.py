"""Render one finished clip: trim -> action-aware dynamic reframe -> burn subs.

All clips end at identical params (1280x720, 30fps, yuv420p, aac, no data stream)
so the assemble step can concat them with -c copy.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .rank import Segment
from .reframe import reframe_clip

W, H, FPS = 1280, 720, 30


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}):\n{proc.stderr[-2000:]}")


def _ass_filter_path(ass_path: Path) -> str:
    """Escape a path for use inside ffmpeg's subtitles= filter (Windows-safe)."""
    p = str(ass_path.resolve()).replace("\\", "/")
    return p.replace(":", r"\:")  # escape drive-letter colon


def render_clip(
    video: str | Path,
    seg: Segment,
    ass_path: str | Path,
    out_path: str | Path,
    encoder: str = "libx264",
) -> Path:
    """Trim [seg.start, seg.end], dynamic reframe, burn subtitles. Raises on fail."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dur = seg["end"] - seg["start"]
    work = out_path.with_name(out_path.stem + "_work.mp4")
    reframed = out_path.with_name(out_path.stem + "_reframed.mp4")

    # A: trim + normalize fps, keep source resolution (reframe crops from it), keep audio
    _run(["ffmpeg", "-ss", str(seg["start"]), "-t", str(dur), "-i", str(video),
          "-r", str(FPS), "-c:v", encoder, "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-ar", "48000", "-ac", "2", "-dn", "-y", str(work)])

    # B: action-aware dynamic reframe (zoom/pan/motion-blur) -> 720p video, no audio
    reframe_clip(work, reframed, out_w=W, out_h=H)

    # C: burn subtitles on the reframed video, take audio from the work clip
    _run(["ffmpeg", "-i", str(reframed), "-i", str(work),
          "-filter_complex", f"[0:v]subtitles='{_ass_filter_path(Path(ass_path))}'[v]",
          "-map", "[v]", "-map", "1:a", "-r", str(FPS),
          "-c:v", encoder, "-pix_fmt", "yuv420p", "-c:a", "aac",
          "-dn", "-shortest", "-y", str(out_path)])

    work.unlink(missing_ok=True)
    reframed.unlink(missing_ok=True)
    return out_path
