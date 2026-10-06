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
    # errors="replace" since ffmpeg echoes source metadata (twitch titles) that cp1252 can't decode.
    # without it the reader thread dies quietly and stderr comes back empty when ffmpeg actually fails
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}):\n{proc.stderr}")
    return out_wav
