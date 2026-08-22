"""Download a Twitch VOD's video + chat by id, caching to disk.

Shells the repo-root TwitchDownloaderCLI (gitignored). First step of the hands-off
`make` pipeline: turns a bare VOD number into local video + chat-JSON paths.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

CLI = Path(__file__).resolve().parents[1] / "TwitchDownloaderCLI.exe"


def window_tag(start: float | None = None, end: float | None = None) -> str:
    """Cache-name suffix for a partial download; '' for the whole VOD.

    Keeps a windowed download (and the run/ cache keyed off it) from colliding with the
    full stream's files -- every cache in this pipeline is skip-if-exists.
    """
    if not start and end is None:
        return ""
    return f"@{int(start or 0)}-{'end' if end is None else int(end)}"


def ensure_vod(vod_id, cache_dir: str | Path = "vods",
               start: float | None = None, end: float | None = None) -> tuple[Path, Path]:
    """Return (video_path, chat_path) for `vod_id`, downloading whichever is missing.

    `start`/`end` (seconds into the stream) download only that slice -- the downloader
    crops server-side, so a 1h window off a 6h stream costs 1h of transfer. The cropped
    mp4 starts at t=0, but chat offsets stay absolute (see `chat_signal(offset=)`).
    """
    vid = str(vod_id)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    tag = window_tag(start, end)
    video = cache / f"{vid}{tag}.mp4"
    chat = cache / f"{vid}{tag}.chat.json"
    crop = []                                   # "#s" -- a bare number is ambiguous to the CLI
    if start:
        crop += ["-b", f"{int(start)}s"]
    if end is not None:
        crop += ["-e", f"{int(end)}s"]
    if not CLI.exists():
        raise SystemExit(f"TwitchDownloaderCLI not found at {CLI}")
    if not video.exists():
        subprocess.run([str(CLI), "videodownload", "--id", vid, "-o", str(video)] + crop,
                       check=True)
    if not chat.exists():
        subprocess.run([str(CLI), "chatdownload", "--id", vid, "-o", str(chat)] + crop,
                       check=True)
    return video, chat
