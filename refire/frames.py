"""The pipeline's eyes: sample frames from the VOD, tile them into contact sheets.

`sample_frames` gives the local VLM (vlm.py) something to caption and the director's
contact sheets their tiles; `contact_sheet`/`cut_sheet` build the image grids Claude
actually looks at; `render_proxy` is the cheap 480p rough used only so the visual critic
can screenshot the cut. All ffmpeg/OpenCV -- the pure helpers (`_parse_showinfo`,
`_thin_frames`, `_grid`) carry the logic and the tests.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

SCENE_THRESH = 0.30      # ffmpeg select scene score to count as a cut
MIN_INTERVAL_S = 20.0    # never keep two frames closer than this
FLOOR_GAP_S = 180.0      # never leave a stretch longer than this unseen
MAX_FRAMES = 600         # hard cap per VOD (VLM captioning time is the real cost)
FRAME_W = 640            # stored frame width (VLM input; sheets rescale anyway)

_PTS = re.compile(r"pts_time:\s*([0-9.]+)")


def _parse_showinfo(stderr_text: str) -> list[float]:
    """The pts_time stamps out of ffmpeg showinfo stderr chatter (pure)."""
    return [float(m) for m in _PTS.findall(stderr_text)]


def _thin_frames(times: list[float], duration: float,
                 min_interval_s: float = MIN_INTERVAL_S,
                 floor_gap_s: float = FLOOR_GAP_S,
                 max_frames: int = MAX_FRAMES) -> list[float]:
    """Scene-cut stamps -> the sampling plan (pure).

    Enforce a minimum spacing, then backfill any gap longer than `floor_gap_s` with
    uniform stamps (quiet stretches must not be invisible to the director), then cap
    at `max_frames` by even decimation.
    """
    kept: list[float] = []
    for t in sorted(times):
        if not kept or t - kept[-1] >= min_interval_s:
            kept.append(t)
    filled: list[float] = []
    prev = 0.0
    for t in kept + [duration]:
        gap = t - prev
        if gap > floor_gap_s:
            n = int(gap // floor_gap_s)
            filled.extend(prev + gap * (i + 1) / (n + 1) for i in range(n))
        if t < duration:
            filled.append(t)
        prev = t
    if len(filled) > max_frames:
        idx = np.linspace(0, len(filled) - 1, max_frames).round().astype(int)
        filled = [filled[i] for i in sorted(set(idx))]
    return filled


def _video_duration(video: str | Path) -> float:
    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return n / fps if fps > 0 else 0.0
    finally:
        cap.release()


def sample_frames(video: str | Path, out_dir: str | Path,
                  scene_thresh: float = SCENE_THRESH,
                  min_interval_s: float = MIN_INTERVAL_S,
                  max_frames: int = MAX_FRAMES) -> list[dict]:
    """Scene-sampled frames from the VOD -> out_dir/f_<t>.jpg + index.json.

    Pass 1 finds scene cuts decoding KEYFRAMES ONLY (a multi-hour VOD scans in minutes);
    pass 2 seek-extracts each planned stamp. Cached by index.json (delete to resample).
    Returns [{"t": seconds, "path": str}]. Any ffmpeg failure -> [] (director stays
    text-only rather than the run dying for want of pictures).
    """
    out_dir = Path(out_dir)
    index = out_dir / "index.json"
    if index.exists():
        try:
            return json.loads(index.read_text(encoding="utf-8"))
        except ValueError:
            pass
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        duration = _video_duration(video)
        scan = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-skip_frame", "nokey",
             "-i", str(video),
             "-vf", f"select='gt(scene,{scene_thresh})',showinfo",
             "-f", "null", "-"],
            capture_output=True, text=True, errors="replace")
        stamps = _thin_frames(_parse_showinfo(scan.stderr), duration,
                              min_interval_s=min_interval_s, max_frames=max_frames)
        print(f"[frames] {len(stamps)} frames planned over {duration / 60:.0f} min")
        frames: list[dict] = []
        for t in stamps:
            path = out_dir / f"f_{int(t)}.jpg"
            if not path.exists():
                r = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                     "-vf", f"scale={FRAME_W}:-2", "-q:v", "4", str(path)],
                    capture_output=True, text=True)
                if r.returncode != 0 or not path.exists():
                    continue
            frames.append({"t": float(t), "path": str(path)})
        index.write_text(json.dumps(frames), encoding="utf-8")
        return frames
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[frames] sampling failed ({e}); continuing without vision")
        return []


def _grid(n: int, cols: int, rows: int) -> list[tuple[int, int]]:
    """(row, col) for each of n tiles, row-major, capped at the grid size (pure)."""
    return [(i // cols, i % cols) for i in range(min(n, cols * rows))]


def _label(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def contact_sheet(frames: list[dict], out_path: str | Path,
                  cols: int = 3, rows: int = 3, sheet_w: int = 1536) -> Path | None:
    """Tile up to cols*rows frames into one labeled grid image.

    Each tile gets a big timestamp bar (Claude downsamples sheets to ~1568px -- the
    labels must survive, tiny HUD text won't; sheets ground SCENES, not details).
    Returns None when there is nothing to tile or a frame fails to load.
    """
    frames = frames[:cols * rows]
    if not frames:
        return None
    tw = sheet_w // cols
    tiles: list[np.ndarray | None] = []
    for f in frames:
        img = cv2.imread(str(f["path"]))
        if img is None:
            continue
        th = int(tw * img.shape[0] / img.shape[1])
        img = cv2.resize(img, (tw, th))
        bar = np.zeros((28, tw, 3), dtype=np.uint8)
        cv2.putText(bar, _label(f["t"]), (6, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(np.vstack([img, bar]))
    if not tiles:
        return None
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT) for t in tiles]
    grid = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for tile, (r, c) in zip(tiles, _grid(len(tiles), cols, rows)):
        grid[r * th:r * th + tile.shape[0], c * tw:c * tw + tw] = tile
    used_rows = _grid(len(tiles), cols, rows)[-1][0] + 1
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid[:used_rows * th])
    return out_path


def cut_sheet(video: str | Path, out_dir: str | Path,
              n_sheets: int = 4, cols: int = 3, rows: int = 3) -> list[Path]:
    """Uniformly screenshot an already-rendered (short) cut into labeled sheets.

    The visual critic's view of the actual video: labels are CUT time, not stream time.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = _video_duration(video)
    if dur <= 0:
        return []
    n = n_sheets * cols * rows
    cap = cv2.VideoCapture(str(video))
    frames: list[dict] = []
    try:
        for i in range(n):
            t = dur * (i + 0.5) / n
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            if not ok:
                continue
            p = out_dir / f"cutframe_{i:03d}.jpg"
            cv2.imwrite(str(p), img)
            frames.append({"t": t, "path": str(p)})
    finally:
        cap.release()
    per = cols * rows
    sheets = []
    for s in range(n_sheets):
        sheet = contact_sheet(frames[s * per:(s + 1) * per],
                              out_dir / f"cut_{s + 1}.jpg", cols=cols, rows=rows)
        if sheet:
            sheets.append(sheet)
    for f in frames:   # tiles only existed to be sheeted
        Path(f["path"]).unlink(missing_ok=True)
    return sheets


def render_proxy(video: str | Path, out_path: str | Path, clips: list[dict],
                 height: int = 480, encoder: str = "libx264") -> Path:
    """Fast low-res concat of the cast clips -- exists ONLY to be screenshotted by the
    visual critic (no subs, no zoom, no silence compression). Raises on ffmpeg failure.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    for i, c in enumerate(clips):
        p = out_path.with_name(f"_proxy_{i:02d}.mp4")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-ss", f"{c['start']:.2f}", "-to", f"{c['end']:.2f}",
               "-i", str(video), "-vf", f"scale=-2:{height}",
               "-c:v", encoder, "-c:a", "aac"]
        if encoder == "libx264":
            cmd += ["-preset", "ultrafast", "-crf", "30"]
        r = subprocess.run(cmd + [str(p)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"proxy clip failed: {r.stderr[-300:]}")
        parts.append(p)
    lst = out_path.with_name("_proxy_list.txt")
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", str(out_path)], capture_output=True, text=True)
    for p in parts + [lst]:
        p.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"proxy concat failed: {r.stderr[-300:]}")
    return out_path
