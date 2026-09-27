"""Extract mono 16kHz audio from a video via ffmpeg (system binary)."""
from __future__ import annotations

import subprocess
from pathlib import Path


def extract_audio(video_path: str | Path, out_wav: str | Path) -> Path:
    """Run ffmpeg to produce a mono 16kHz WAV. Raises on ffmpeg failure."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-ac", "1", "-ar", "16000",
        "-y", str(out_wav),
    ]
    # errors="replace": ffmpeg echoes the source's metadata (a Twitch title carries bytes
    # cp1252 has no mapping for), and decoding that in subprocess's reader THREAD raised
    # UnicodeDecodeError -- printing a traceback per call while the run carried on, since a
    # dead reader thread doesn't fail the process. Cosmetic until ffmpeg really fails, at
    # which point the stderr below is the only diagnostic and would come back empty.
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}):\n{proc.stderr}")
    return out_wav
