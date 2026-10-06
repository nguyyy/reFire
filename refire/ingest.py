"""Download a Twitch VOD's video + chat by id, caching to disk.

Shells the repo-root TwitchDownloaderCLI (gitignored). First step of the hands-off
`make` pipeline: turns a bare VOD number into local video + chat-JSON paths.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

CLI = Path(__file__).resolve().parents[1] / "TwitchDownloaderCLI.exe"

# download threads. the CLI default of 4 was throttling the download itself (17 Mbps vs 36
# with a second download running). tune with --threads, lower it if downloads fail partway
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


# the downloader separates progress with bare \r and only ends a line when a phase finishes,
# so piped output looks stuck for the whole download. parse it and send it to progress instead
_STATUS = re.compile(r"\[STATUS\] - (.+?)(?: (\d+)%)? \[(\d+)/(\d+)\]")


def _status(line: str) -> tuple[float, str] | None:
    """`(fraction, phase text)` for one downloader status line, else None.

    The percent is rounded down to 5s: the caller prints on every CHANGE of message, and
    per-percent lines would bury the rest of the run's log in a few hundred of them.
    """
    m = _STATUS.search(line)
    if not m:
        return None
    phase, pct, step, steps = m[1], m[2], int(m[3]), int(m[4])
    pct = None if pct is None else int(pct)
    frac = (step - 1 + (pct or 0) / 100.0) / max(steps, 1)
    # ascii only, the console is cp1252 and one bad glyph kills the run after the download
    phase = phase.encode("ascii", "ignore").decode().lower()
    return min(1.0, max(0.0, frac)), phase if pct is None else f"{phase} {pct - pct % 5}%"


def _run(cmd: list[str], progress=None) -> None:
    """Run the downloader, forwarding its status line to `progress(frac, msg)`.

    Without a callback this is plain `subprocess.run`: in a terminal the CLI's own live
    output is already fine, and only a piped caller needs the translation.
    """
    if progress is None:
        subprocess.run(cmd, check=True)
        return
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    buf, last, tail = "", None, []
    while True:
        ch = proc.stdout.read(1)    # read by char, readline() blocks until the phase ends
        if not ch:
            break
        if ch not in "\r\n":
            buf += ch
            continue
        line, buf = buf.strip(), ""
        got = _status(line)
        if got is None:
            # not a status line (banner or error text). keep the last few so we can say why it failed
            tail = (tail + [line])[-3:] if line else tail
        elif got[1] != last:
            last = got[1]
            progress(*got)
    if proc.wait():
        raise RuntimeError(f"{cmd[1]} exited {proc.returncode}: "
                           + (" | ".join(tail) or "no output"))


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
    out.unlink(missing_ok=True)             # aborted runs leave a 0-byte stub
    _run([str(CLI), "videodownload", "--id", str(vod_id), "-o", str(out),
          "-t", str(threads)])
    if not _done(out):
        raise RuntimeError(f"audio download produced an empty {out}")
    return out


def ensure_vod(vod_id, cache_dir: str | Path = "vods",
               start: float | None = None, end: float | None = None,
               want_chat: bool = True, threads: int = DL_THREADS,
               progress=None) -> tuple[Path, Path]:
    """Return (video_path, chat_path) for `vod_id`, downloading whichever is missing.

    `start`/`end` (seconds into the stream) download only that slice -- the downloader
    crops server-side, so a 1h window off a 6h stream costs 1h of transfer. The cropped
    mp4 starts at t=0, but chat offsets stay absolute (see `chat_signal(offset=)`).

    `want_chat=False` skips the chat leg (the path is still returned, just not fetched):
    a caller pulling down dozens of short windows would otherwise spawn a second CLI
    process per window for a chat log nothing downstream reads.

    `progress(frac, msg)` reports the downloader's own phases across all jobs (0..1).
    """
    vid = str(vod_id)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    tag = window_tag(start, end)
    video = cache / f"{vid}{tag}.mp4"
    chat = cache / f"{vid}{tag}.chat.json"
    crop = []                                   # "#s", the CLI misreads a bare number
    if start:
        crop += ["-b", f"{int(start)}s"]
    if end is not None:
        crop += ["-e", f"{int(end)}s"]
    if not CLI.exists():
        raise SystemExit(f"TwitchDownloaderCLI not found at {CLI}")
    jobs = [("videodownload", video)] + ([("chatdownload", chat)] if want_chat else [])
    for i, (kind, out) in enumerate(jobs):
        if _done(out):
            continue
        out.unlink(missing_ok=True)         # aborted runs leave a 0-byte stub
        step = None if progress is None else (
            lambda f, m, i=i: progress((i + f) / len(jobs), m))
        _run([str(CLI), kind, "--id", vid, "-o", str(out),
              "-t", str(threads)] + crop, step)
        if not _done(out):
            raise RuntimeError(f"{kind} produced an empty {out}")
    return video, chat
