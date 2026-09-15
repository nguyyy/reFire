"""Export an After Effects build manifest: clips + captions + zoom keyframes.

The primary final stage (the ffmpeg burn-in in `assemble.render_clips` is the
eyeball-it-quick alternative). `pipeline.make` writes `manifest.json`; the reFire CEP
panel (`refire/ae/`) reads it inside After Effects to build styled, hand-tunable comps.
This just reshapes the already-computed per-clip data for AE.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .emphasis import style_text
from .reframe import ENTER, EXIT, motion_intervals
from .select import SILENCE_PAD, compress_silence, speech_intervals
from .style import style_for
from .subtitles import group_words

FPS = 30
OUT_W, OUT_H = 1920, 1080


def _max_dev(ts, zs, i, j) -> float:
    """Max vertical distance of points i+1..j-1 from the line (i)->(j)."""
    if j <= i + 1:
        return 0.0
    t0, z0, t1, z1 = ts[i], zs[i], ts[j], zs[j]
    dt = t1 - t0
    m = 0.0
    for k in range(i + 1, j):
        zline = z0 if dt == 0 else z0 + (z1 - z0) * ((ts[k] - t0) / dt)
        d = abs(zs[k] - zline)
        if d > m:
            m = d
    return m


def decimate_zoom(zoom, fps: float = FPS, tol: float = 1.5) -> list[dict]:
    """Per-frame zoom multipliers -> sparse [{t, scale}] keyframes (percent).

    Greedy linear simplification: keep the farthest point we can reach while
    every skipped sample stays within `tol` percent of the straight line. Keeps
    the punch/creep/zoom-out shape as a handful of keyframes the user can re-ease.
    """
    # ponytail: O(n^2) simplify; fine for clip-length arrays (<~a few hundred).
    n = len(zoom)
    if n == 0:
        return []
    ts = [i / fps for i in range(n)]
    zs = [float(z) * 100.0 for z in zoom]
    keep = [0]
    i = 0
    while i < n - 1:
        j, best = i + 1, i + 1
        while j < n:
            if _max_dev(ts, zs, i, j) <= tol:
                best, j = j, j + 1
            else:
                break
        keep.append(best)
        i = best
    return [{"t": round(ts[k], 3), "scale": round(zs[k], 2)} for k in keep]


def clip_intensity(source, start: float, end: float):
    """Inter-frame motion intensity for source seconds [start, end). (cv2; untested.)"""
    import cv2  # heavy/optional dep, import lazily

    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    inten, prev = [], None
    while True:
        ok, frame = cap.read()
        if not ok or cap.get(cv2.CAP_PROP_POS_MSEC) > end * 1000.0:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        inten.append(0.0 if prev is None else float(cv2.absdiff(small, prev).mean()))
        prev = small
    cap.release()
    return inten, fps


# After Effects clamps EVERY time value -- a layer's startTime, a comp's duration, a Time
# Remap value -- to +/-10800s (3h), so a layer simply cannot reach past the 3-hour mark of
# a source. 6h+ VODs are the norm here, so the source gets split into parts AE can address.
# 2.5h leaves headroom: a keyframe-snapped part slightly over 3h would be unreachable at
# its own tail. ponytail: raise toward 10800 only if the part count ever becomes a problem.
SEG_S = 9000.0


def _duration(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, errors="replace", check=True)
    return float(out.stdout.strip())


def _split_points(dur: float, clips, seg_s: float, pad: float = 5.0) -> list[float]:
    """Cut points every ~seg_s, nudged back out of any clip so none straddles a part.

    `pad` outruns the keyframe snap: ffmpeg cuts at the first keyframe AT OR AFTER the
    requested time, and Twitch VODs key every ~2s, so backing off 5s from a clip's head
    still lands the real cut outside it.
    """
    spans = sorted((c["start"] - pad, c["end"] + pad) for c in clips)
    pts, k = [], 1
    while k * seg_s < dur:
        t = k * seg_s
        for a, b in reversed(spans):        # decreasing, so a chain of clips walks back
            if a <= t <= b:
                t = a
        pts.append(max(0.0, t))
        k += 1
    return pts


def split_source(video, parts_dir, clips, seg_s: float = SEG_S):
    """Long VOD -> AE-addressable parts. Returns [(path, offset_s)], offsets exact.

    Stream copy, no re-encode. Cut points land in the gaps BETWEEN clips (`_split_points`)
    so no clip is torn in half; ffmpeg still snaps each cut forward to the next keyframe,
    so the REAL cut times are read back out of its segment list rather than assumed --
    summing part durations instead would drift by a frame or two per part (the container
    pads the tail past the last video frame). Cached: the parts sit next to the VOD and are
    reused across runs. Source shorter than one part -> [(video, 0.0)], unchanged manifest.
    """
    video = Path(video)
    if not video.exists():          # nothing to probe; AE reports the missing source
        return [(video, 0.0)]
    total = _duration(video)
    if total <= seg_s:
        return [(video, 0.0)]

    parts_dir = Path(parts_dir)
    lst = parts_dir / "parts.csv"
    parts = _read_parts(lst)
    if not parts or abs(parts[-1][2] - total) > 5.0:      # absent / stale / half-written
        parts_dir.mkdir(parents=True, exist_ok=True)
        for stale in parts_dir.glob("part_*.mp4"):
            stale.unlink()
        times = ",".join(f"{t:.3f}" for t in _split_points(total, clips, seg_s))
        print(f"[ae] splitting {video.name} ({total / 3600:.1f}h) into "
              f"{times.count(',') + 2} parts -- AE cannot address past 3h of one file")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-c", "copy", "-map", "0", "-dn",
             "-f", "segment", "-segment_times", times, "-reset_timestamps", "1",
             "-segment_list", str(lst), "-segment_list_type", "csv",
             "-y", str(parts_dir / "part_%02d.mp4")], check=True)
        parts = _read_parts(lst)
    return [(p, off) for p, off, _end in parts]


# AE has no smart long-GOP decoding: every preview frame of an OBS/NVENC H.264 VOD is
# rebuilt from its GOP, which is where most of AE's slowness comes from. DNxHR LB is
# all-intra (one frame = one seek) and usually previews 3-10x faster.
PROXY_PAD = 2.0      # head headroom so a cut can still be nudged earlier inside AE
PROXY_TAIL = 10.0    # tail headroom: clip ends land mid-word far more often than starts,
                     # so there's room to extend to the end of the sentence in AE
# Profile is a real quality knob, not a constant to inline: LB measured SSIM 0.954 against
# the source on an 8Mbps Twitch VOD -- it re-quantizes existing blocking into mush. SQ is
# 0.997 (visually transparent) for ~3.2x the disk. Drop to LB only if scratch space hurts.
PROXY_PROFILE = "dnxhr_sq"


def proxy_clips(video, proxy_dir, clips, pad: float = PROXY_PAD,
                tail: float = PROXY_TAIL):
    """Cut each clip to a padded all-intra proxy. Returns [(path, offset)], one per clip.

    `pad`/`tail` are proxy headroom only -- the manifest's in/out points don't move, so
    a bigger tail costs disk and nothing else.

    Only the seconds actually used get transcoded (a few minutes, not a 6h VOD), and
    every proxy is far under AE's 3h time clamp, so `split_source` is not needed here.
    Cached by span: re-running a build reuses proxies it already cut.
    Returns [] if the source is missing, so callers fall back to the raw-VOD path.
    """
    video = Path(video)
    if not video.exists():
        return []
    proxy_dir = Path(proxy_dir)
    out = []
    for c in clips:
        a = max(0.0, c["start"] - pad)
        b = c["end"] + tail
        # profile is in the name: otherwise a quality change silently reuses the old,
        # worse-looking proxies that already exist for the same span.
        p = proxy_dir / f"p_{PROXY_PROFILE}_{a:.3f}-{b:.3f}.mov"
        if not p.exists():
            proxy_dir.mkdir(parents=True, exist_ok=True)
            print(f"[ae] proxy {p.name} ({b - a:.0f}s)")
            subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}",
                 "-i", str(video), "-c:v", "dnxhd", "-profile:v", PROXY_PROFILE,
                 "-pix_fmt", "yuv422p", "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
                 "-dn", "-y", str(p)], check=True)
        out.append((p, a))
    return out


def _read_parts(lst: Path):
    """ffmpeg's segment list -> [(path, start_s, end_s)] in SOURCE time. [] if unusable."""
    if not lst.exists():
        return []
    out = []
    for line in lst.read_text(encoding="utf-8").splitlines():
        name, _, rest = line.partition(",")
        start, _, end = rest.partition(",")
        p = lst.parent / name
        if not (start and end and p.exists()):
            return []
        out.append((p, float(start), float(end)))
    return out


def _assign_parts(clips, parts) -> None:
    """Tag each clip with the part index holding it (`src`), in place. Absolute times stay."""
    if len(parts) < 2:
        return
    bounds = [off for _p, off in parts] + [float("inf")]
    for c in clips:
        i = max(k for k in range(len(parts)) if bounds[k] <= c["start"])
        if c["end"] > bounds[i + 1]:
            print(f"[ae] warning: clip at {c['start']:.0f}s straddles a part boundary "
                  f"-- its tail will be missing; re-split or nudge the cut.")
        c["src"] = i


def _tighten(keep):
    """Map clip-relative SOURCE seconds -> the dead-air-free timeline `keep` describes.

    Anything inside a removed gap snaps forward to the next kept span's head, so a
    caption/zoom/overlay that started in the silence still lands on the cut.
    """
    offsets, acc = [], 0.0
    for a, b in keep:
        offsets.append(acc)
        acc += b - a

    def at(t: float) -> float:
        for (a, b), off in zip(keep, offsets):
            if t < a:
                return off
            if t <= b:
                return off + (t - a)
        return acc

    return at


def build_manifest(video, run_dir, words, sections, words_per_line: int = 3,
                   z_enter: float = ENTER, z_exit: float = EXIT,
                   overlays_by_clip=None, bgm=None, bgm_db: float = -18.0,
                   motion_zoom: bool = True, deadspace: bool = True,
                   silence_pad: float = SILENCE_PAD, cards: bool = True,
                   proxy: bool = True, captions: str = "all", sources=None,
                   voiced=None) -> Path:
    """Write run_dir/ae/manifest.json from already-chosen sections. Returns its path.

    Shared by `export_ae` (legacy detect path) and `pipeline.make` (brief path).
    `words` must already be emphasis-annotated; `sections` is [{title, clips}] where
    each clip is {start, end} in absolute source seconds (sentence-snapped upstream).
    `overlays_by_clip`, if given, is a per-clip list (in flattened build order) of
    overlay punch-ins; `bgm` is a music-bed path mixed under the whole cut.
    `motion_zoom=False` skips the OpenCV frame scan and writes static-fit clips
    (`zoom_episodes: []`) for faster, calmer builds.
    `deadspace=True` drops each clip's internal dead air: the manifest gains a per-clip
    `keep` (clip-relative source spans) + `dur` (tightened length) that AE jump-cuts with
    Time Remap, and captions/zoom/overlays are retimed onto that tight timeline. `voiced`
    keeps speech the transcript has no words for (see `select.compress_silence`).
    `cards=False` marks every section `card: False`, so the Master runs clip-to-clip with
    no section title cards (titles stay in the manifest for the outline/debugging).
    `proxy=True` cuts each clip to an all-intra DNxHR LB proxy and points AE at those
    instead of the long-GOP VOD -- much faster previews, at the cost of one transcode
    per clip up front; `proxy=False` keeps AE on the original source.
    `sources`, if given, is the media map [(path, offset_s)] to write verbatim -- the
    caller has already tagged each clip's `src` and no proxying or splitting happens. That
    is how a cut spanning SEVERAL VODs ships: there is no single `video` to slice, so each
    clip points at its own separately-downloaded window and clip times stay absolute on
    whatever timeline the caller laid out (`refire.loud`).
    """
    run_dir = Path(run_dir)
    speech = speech_intervals(words)                 # hold zoom through phrases
    clips = []
    manifest_sections = []
    ci = 0                                            # flattened clip counter
    for sec in sections:
        idxs = []
        for seg in sec["clips"]:
            # Style pass: a beat's role/energy tilts presentation. No role (legacy detect
            # path / flat fallback) -> NEUTRAL, so those manifests stay byte-identical.
            role = seg.get("role", "")
            if role:
                st = style_for(role, seg.get("energy", 3))
                wpl = st["words_per_line"]
                cz_enter = min(0.95, z_enter / st["zoom_sens"])
                cz_exit = min(cz_enter * 0.9, z_exit / st["zoom_sens"])
            else:
                wpl, cz_enter, cz_exit = words_per_line, z_enter, z_exit
            # Dead-air pass: `keep` are the clip-relative source spans worth playing;
            # everything downstream (captions, zoom, overlays) is retimed onto the
            # gap-free timeline `tight` describes, and AE jump-cuts to it via Time Remap.
            keep = tight = None
            if deadspace:
                keep, retimed, cdur = compress_silence(words, seg["start"], seg["end"],
                                                       pad=silence_pad, voiced=voiced)
                tight = _tighten(keep)
                groups = group_words(retimed, 0.0, cdur, wpl)
            else:
                groups = group_words(words, seg["start"], seg["end"], wpl)
            # `captions="emph"` keeps only the lines carrying an emphasized word (a
            # yell, an interjection, an ALL-CAPS transcription). A full subtitle track
            # competes with the cuts in a fast style; sparse text just prevents the
            # confusion the cutting can't. `emph` is already on every word.
            lines = [
                {"text": " ".join(style_text(w["text"], w["emph"]) for w in g),
                 "start": g[0]["start"], "end": g[-1]["end"]}
                for g in groups
                if captions != "emph" or any(w.get("emph") for w in g)
            ]
            # clip-relative speech runs so an episode never releases mid-sentence
            seg_speech = [(max(s, seg["start"]) - seg["start"], min(e, seg["end"]) - seg["start"])
                          for s, e in speech if e > seg["start"] and s < seg["end"]]
            if motion_zoom:
                inten, fps = clip_intensity(video, seg["start"], seg["end"])
                spans = motion_intervals(inten, fps=fps, enter=cz_enter, exit=cz_exit,
                                         speech_intervals=seg_speech)
                if tight:
                    spans = [(tight(a), tight(b)) for a, b in spans]
                episodes = [{"start": round(a, 3), "end": round(b, 3)}
                            for a, b in spans if b - a > 1e-3]
            else:
                episodes = []
            clip = {"start": seg["start"], "end": seg["end"],
                    "captions": lines, "zoom_episodes": episodes}
            # nothing actually cut (silent clip) -> leave the manifest as it was
            if keep and cdur < (seg["end"] - seg["start"]) - 1e-3:
                clip["keep"] = [[round(a, 3), round(b, 3)] for a, b in keep]
                clip["dur"] = round(cdur, 3)
            if role:
                clip["role"] = role                  # informational for AE / debugging
            if "src" in seg:
                clip["src"] = seg["src"]             # caller-supplied media map (see `sources`)
            if overlays_by_clip and ci < len(overlays_by_clip):
                ovs = overlays_by_clip[ci]
                if tight:
                    ovs = [dict(o, start=round(tight(o["start"]), 3)) for o in ovs]
                clip["overlays"] = ovs
            clips.append(clip)
            idxs.append(len(clips) - 1)
            ci += 1
        # hook = minimal: no title card (the cut starts hot). Only flag when False so
        # legacy/neutral sections keep the existing schema.
        manifest_section = {"title": sec["title"], "clip_indices": idxs}
        if not cards or (sec.get("role")
                         and not style_for(sec["role"], sec.get("energy", 3))["card"]):
            manifest_section["card"] = False
        manifest_sections.append(manifest_section)

    # Each clip gets its own all-intra proxy so AE never decodes the long-GOP VOD.
    # `source` stays for the raw-VOD case and older manifests.
    if sources is not None:
        parts = list(sources)               # caller owns the media map and the src tags
    else:
        parts = proxy_clips(video, run_dir / "ae" / "proxies", clips) if proxy else []
        if parts:
            for i, c in enumerate(clips):
                c["src"] = i
        else:
            # raw VOD: AE can only reach 3h into a file, so a long one ships as parts.
            parts = split_source(video, Path(video).with_suffix(".parts"), clips)
            _assign_parts(clips, parts)
            if len(parts) < 2:
                parts = []

    manifest = {
        "source": str(Path(video).resolve()).replace("\\", "/"),
        "fps": FPS, "out_w": OUT_W, "out_h": OUT_H,
        "sections": manifest_sections,
        "clips": clips,
    }
    if len(parts) > 1 or sources is not None:
        manifest["sources"] = [{"path": str(p.resolve()).replace("\\", "/"),
                                "offset": round(off, 3)} for p, off in parts]
    if bgm:
        manifest["bgm"] = str(Path(bgm).resolve()).replace("\\", "/")
        manifest["bgm_db"] = bgm_db
    out_dir = run_dir / "ae"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp = out_dir / "manifest.json"
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return mp
