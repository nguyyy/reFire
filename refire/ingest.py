"""Download a Twitch VOD's video + chat by id, caching to disk.

Shells the repo-root TwitchDownloaderCLI (gitignored). First step of the hands-off
`make` pipeline: turns a bare VOD number into local video + chat-JSON paths.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

CLI = Path(__file__).resolve().parents[1] / "TwitchDownloaderCLI.exe"


def ensure_vod(vod_id, cache_dir: str | Path = "vods") -> tuple[Path, Path]:
    """Return (video_path, chat_path) for `vod_id`, downloading whichever is missing."""
    vid = str(vod_id)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    video = cache / f"{vid}.mp4"
    chat = cache / f"{vid}.chat.json"
    if not CLI.exists():
        raise SystemExit(f"TwitchDownloaderCLI not found at {CLI}")
    if not video.exists():
        subprocess.run([str(CLI), "videodownload", "--id", vid, "-o", str(video)], check=True)
    if not chat.exists():
        subprocess.run([str(CLI), "chatdownload", "--id", vid, "-o", str(chat)], check=True)
    return video, chat
