"""Export an After Effects build manifest: clips + captions + zoom keyframes.

Alternate final stage to the ffmpeg burn-in (`assemble.edit`). reFire writes
`run/ae/manifest.json`; `refire/ae/reFire.jsx` reads it inside After Effects to
build styled, hand-tunable comps. The detector (Stage 1) is untouched — this just
reshapes the already-computed per-clip data for AE.
"""
from __future__ import annotations

import json
from pathlib import Path

from .continuity import organize
from .emphasis import annotate_emphasis, style_text
from .reframe import ENTER, EXIT, motion_intervals
from .score import DEFAULT_MODEL
from .select import select_segments, snap_to_sentences, speech_intervals
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


def export_ae(
    video,
    run_dir="run",
    count: int | None = None,
    min_score: float | None = None,
    order: str = "chrono",
    words_per_line: int = 3,
    topic: str = "",
    model: str = DEFAULT_MODEL,
    zoom_sens: float = 1.0,
) -> Path:
    """Write run_dir/ae/manifest.json for the AE builder. Returns its path.

    zoom_sens scales how readily motion triggers a zoom: >1 = more/earlier punches,
    <1 = fewer. min_score/count control how much footage survives selection.
    """
    # higher sensitivity -> lower motion thresholds -> more zoom episodes
    z_enter = min(0.95, ENTER / max(zoom_sens, 1e-3))
    z_exit = min(z_enter * 0.9, EXIT / max(zoom_sens, 1e-3))
    run_dir = Path(run_dir)
    segments = json.loads((run_dir / "segments.json").read_text(encoding="utf-8"))
    words = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
    annotate_emphasis(words, run_dir / "audio.wav")  # lowercase + caps-on-hype
    speech = speech_intervals(words)                 # hold zoom through phrases

    chosen = select_segments(segments, count=count, min_score=min_score, order=order)
    if not chosen:
        raise SystemExit("No segments selected (check --count/--min-score).")

    # snap onto sentence boundaries (no mid-sentence cuts), then group into sections
    snapped = []
    for seg in chosen:
        s, e = snap_to_sentences(words, seg["start"], seg["end"])
        snapped.append({**seg, "start": s, "end": e})
    sections = organize(snapped, topic, model)

    clips = []
    manifest_sections = []
    for sec in sections:
        idxs = []
        for seg in sec["clips"]:
            groups = group_words(words, seg["start"], seg["end"], words_per_line)
            captions = [
                {"text": " ".join(style_text(w["text"], w["emph"]) for w in g),
                 "start": g[0]["start"], "end": g[-1]["end"]}
                for g in groups
            ]
            # clip-relative speech runs so an episode never releases mid-sentence
            seg_speech = [(max(s, seg["start"]) - seg["start"], min(e, seg["end"]) - seg["start"])
                          for s, e in speech if e > seg["start"] and s < seg["end"]]
            inten, fps = clip_intensity(video, seg["start"], seg["end"])
            episodes = [{"start": round(a, 3), "end": round(b, 3)}
                        for a, b in motion_intervals(inten, fps=fps,
                                                     enter=z_enter, exit=z_exit,
                                                     speech_intervals=seg_speech)]
            clips.append({"start": seg["start"], "end": seg["end"],
                          "captions": captions, "zoom_episodes": episodes})
            idxs.append(len(clips) - 1)
        manifest_sections.append({"title": sec["title"], "clip_indices": idxs})

    manifest = {
        "source": str(Path(video).resolve()).replace("\\", "/"),
        "fps": FPS, "out_w": OUT_W, "out_h": OUT_H,
        "sections": manifest_sections,
        "clips": clips,
    }
    out_dir = run_dir / "ae"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp = out_dir / "manifest.json"
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return mp
