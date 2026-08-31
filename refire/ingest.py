"""Download a Twitch VOD's video + chat by id, caching to disk.

Shells the repo-root TwitchDownloaderCLI (gitignored). First step of the hands-off
`make` pipeline: turns a bare VOD number into local video + chat-JSON paths.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

CLI = Path(__file__).resolve().parents[1] / "TwitchDownloaderCLI.exe"

# Parallel download threads. The CLI's own default is 4, which measured 17 Mbps here while
# a second concurrent download simultaneously pulled 36 -- i.e. 4 threads throttles the
# DOWNLOAD, not the link. This is a calibration knob, not a constant: the right value is
# whatever your connection and Twitch's rate limiting agree on, so it is a plain flag
# (`--threads`). Back it off if downloads start failing partway.
DL_THREADS = 8


def window_tag(start: float | None = None, end: float | None = None) -> str:
    """Cache-name suffix for a partial download; '' for the whole VOD.

    Keeps a windowed download (and the run/ cache keyed off it) from colliding with the
    full stream's files -- every cache in this pipeline is skip-if-exists.
    """
    if not start and end is None:
        return ""
    return f"@{int(start or 0)}-{'end' if end is None else int(end)}"


def _done(p: Path) -> bool:
    """A cached download counts only if it has bytes -- an interrupted run leaves an
    empty file, which `exists()` happily mistakes for a finished download."""
    return p.exists() and p.stat().st_size > 0


def ensure_audio(vod_id, cache_dir: str | Path = "vods",
                 threads: int = DL_THREADS) -> Path:
    """Return the whole VOD as audio only (.m4a), downloading it if missing.

    The downloader picks the download TYPE off the output extension, so this pulls
    Twitch's own audio-only rendition (~163 kbps) rather than a video download with the
    picture thrown away: ~440MB for a 6h stream against ~15GB for the mp4. Enough to
    find the loud moments; only the windows that survive get their video fetched.
    """
    out = Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    out = out / f"{vod_id}.m4a"
    if _done(out):
        return out
    if not CLI.exists():
        raise SystemExit(f"TwitchDownloaderCLI not found at {CLI}")
    out.unlink(missing_ok=True)             # an aborted run leaves a 0-byte stub behind
    subprocess.run([str(CLI), "videodownload", "--id", str(vod_id), "-o", str(out),
                    "-t", str(threads)], check=True)
    if not _done(out):
        raise RuntimeError(f"audio download produced an empty {out}")
    return out


def ensure_vod(vod_id, cache_dir: str | Path = "vods",
               start: float | None = None, end: float | None = None,
               want_chat: bool = True, threads: int = DL_THREADS) -> tuple[Path, Path]:
    """Return (video_path, chat_path) for `vod_id`, downloading whichever is missing.

    `start`/`end` (seconds into the stream) download only that slice -- the downloader
    crops server-side, so a 1h window off a 6h stream costs 1h of transfer. The cropped
    mp4 starts at t=0, but chat offsets stay absolute (see `chat_signal(offset=)`).

    `want_chat=False` skips the chat leg (the path is still returned, just not fetched):
    a caller pulling down dozens of short windows would otherwise spawn a second CLI
    process per window for a chat log nothing downstream reads.
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
    jobs = [("videodownload", video)] + ([("chatdownload", chat)] if want_chat else [])
    for kind, out in jobs:
        if _done(out):
            continue
        out.unlink(missing_ok=True)         # an aborted run leaves a 0-byte stub behind
        subprocess.run([str(CLI), kind, "--id", vid, "-o", str(out),
                        "-t", str(threads)] + crop, check=True)
        if not _done(out):
            raise RuntimeError(f"{kind} produced an empty {out}")
    return video, chat
