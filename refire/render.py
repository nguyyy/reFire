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

# de-pop ramp at every cut: each kept span fades in and out over this long. a jump cut
# between two arbitrary samples is a step in the waveform, i.e. a click. a dip (not an
# overlapping crossfade) so audio stays sample-locked to the video's frame cuts
FADE_S = 0.015


def audio_graph(keep, dur: float, fade: float = FADE_S, tails=None,
                loudnorm: bool = False) -> str:
    """filter_complex chain [0:a] -> [a] for one clip: the kept spans, each faded in/out.

    `tails[k]` > 0 makes span k an L-CUT: its audio runs that many seconds past its
    video out-point, fading out under the head of span k+1 (mixed, not replacing it), so
    a reaction is not chopped mid-breath by the jump cut. With no tails the spans are
    concatenated, which is exact; tails need amix with each span delayed to its slot.
    """
    spans = [tuple(s) for s in keep] if keep else [(0.0, dur)]
    n = len(spans)
    tails = list(tails or [0.0] * n)
    lcut = any(t > 0 for t in tails)
    head = (f"[0:a]asplit={n}" + "".join(f"[s{k}]" for k in range(n))) if n > 1 else ""
    parts = [head] if head else []
    off = 0.0
    for k, ((a, b), tl) in enumerate(zip(spans, tails)):
        src = f"[s{k}]" if n > 1 else "[0:a]"
        length = (b - a) + tl
        f = min(fade, max(0.0, (b - a) / 4))
        fo = tl if tl > 0 else f
        chain = f"{src}atrim=start={a:.4f}:end={b + tl:.4f},asetpts=PTS-STARTPTS"
        if f > 0:
            chain += (f",afade=t=in:st=0:d={f:.4f}"
                      f",afade=t=out:st={max(0.0, length - fo):.4f}:d={fo:.4f}")
        if lcut:
            chain += f",adelay={off * 1000:.1f}:all=1"
        parts.append(chain + f"[x{k}]")
        off += b - a
    ins = "".join(f"[x{k}]" for k in range(n))
    if lcut:
        tail = f"{ins}amix=inputs={n}:normalize=0:dropout_transition=0,atrim=end={off:.4f}"
    elif n > 1:
        tail = f"{ins}concat=n={n}:v=0:a=1"
    else:
        tail = "[x0]anull"
    parts.append(tail + (f",{LOUDNORM}" if loudnorm else "") + "[a]")
    return ";".join(parts)


def video_quality(encoder: str) -> list[str]:
    """Quality flags per encoder. NVENC's default bitrate is far too low for a 1080p cut that
    goes through three encodes (trim, reframe, subs); libx264 keeps its crf 23 default."""
    if encoder.endswith("_nvenc"):
        return ["-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0"]
    return []


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
    fade: float = FADE_S,
    tails=None,
    size: tuple[int, int] = (W, H),
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

    `fade` is the de-pop ramp at every cut (`audio_graph`; 0 = the old hard cuts) and
    `tails` the per-span L-cut lengths. `size` is the output canvas (default 1280x720).
    """
    out_path = Path(out_path)
    w, h = size
    vq = video_quality(encoder)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dur = seg["end"] - seg["start"]
    work = out_path.with_name(out_path.stem + "_work.mp4")
    reframed = out_path.with_name(out_path.stem + "_reframed.mp4")

    # A: trim + normalize fps, keep source res (reframe crops from it) and audio.
    # if keep cuts dead air, select+setpts drops the gaps in one encode. one full span skips it
    if keep and len(keep) == 1 and keep[0][0] <= 1e-3 and keep[0][1] >= dur - 1e-3:
        keep = None
    cut, afilters = [], []
    if fade > 0 or (tails and any(tails)):
        graph = [audio_graph(keep, dur, fade, tails if keep else None, loudnorm)]
        vmap = "0:v"
        if keep:
            sel = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in keep)
            graph.insert(0, f"[0:v]select='{sel}',setpts=N/FRAME_RATE/TB[v]")
            vmap = "[v]"
        cut = ["-filter_complex", ";".join(graph), "-map", vmap, "-map", "[a]"]
    else:                                   # fade 0: the original hard-cut path
        if keep:
            sel = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in keep)
            cut = ["-vf", f"select='{sel}',setpts=N/FRAME_RATE/TB"]
            afilters += [f"aselect='{sel}'", "asetpts=N/SR/TB"]
        if loudnorm:
            # single pass, two-pass needs a measuring run per clip and doesn't help at these lengths
            afilters.append(LOUDNORM)
        if afilters:
            cut += ["-af", ",".join(afilters)]
    _run(["ffmpeg", "-ss", str(seg["start"]), "-t", str(dur), "-i", str(video),
          *cut, "-r", str(FPS), "-c:v", encoder, *vq, "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-ar", "48000", "-ac", "2", "-dn", "-y", str(work)])

    # B: reframe to the output canvas. motion zoom is optional, it can eat render time on long cuts
    if motion_zoom:
        reframe_clip(work, reframed, out_w=w, out_h=h, speech=speech, enter=enter, exit=exit)
    else:
        _run(["ffmpeg", "-i", str(work), "-vf",
              f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}:0:ih-{h}",
              "-an", "-r", str(FPS), "-c:v", encoder, *vq, "-pix_fmt", "yuv420p",
              "-dn", "-y", str(reframed)])

    # C: burn subs on the reframed video, audio from the work clip
    _run(["ffmpeg", "-i", str(reframed), "-i", str(work),
          "-filter_complex", f"[0:v]subtitles='{_ass_filter_path(Path(ass_path))}'[v]",
          "-map", "[v]", "-map", "1:a", "-r", str(FPS),
          "-c:v", encoder, *vq, "-pix_fmt", "yuv420p", "-c:a", "aac",
          "-dn", "-shortest", "-y", str(out_path)])

    work.unlink(missing_ok=True)
    reframed.unlink(missing_ok=True)
    return out_path
