"""Motion-triggered punch zoom, fixed bottom-left anchor (the webcam corner).

Per-frame inter-frame motion -> hysteresis -> zoom intervals. Nearby intervals
(separated by a short calm gap) are merged into one longer zoom so a rapid
in/out/in sequence becomes a single sustained push. Each interval gets a tapered
curve: a quick punch 100%->190%, then a slow creep 190%->200% across the rest of
the moment, then a quick zoom back to 100%. Crop is anchored bottom-left, so
there is no panning and no per-frame jitter.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ZMAX = 2.0            # final zoom (200%)
PUNCH_TO = 1.9        # where the quick initial punch lands (190%)
PUNCH_S = 0.5         # seconds for the 100% -> 190% punch
OUT_S = 0.5           # seconds for the zoom back to 100%
ENTER = 0.45          # normalized motion to START a zoom
EXIT = 0.18           # normalized motion to END a zoom (hysteresis)
SMOOTH_WIN = 9        # frames of rolling-mean on motion
BRIDGE_GAP_S = 1.5    # merge zooms separated by a gap shorter than this
MIN_INTERVAL_S = 0.4  # ignore motion bursts shorter than this


def _smoothstep(a, b, t):
    t = min(1.0, max(0.0, t))
    return a + (b - a) * (t * t * (3 - 2 * t))


def _intervals(active):
    """Contiguous True runs of `active` as [start, end) frame pairs."""
    out, i, n = [], 0, len(active)
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            out.append([i, j])
            i = j
        else:
            i += 1
    return out


def zoom_track(
    intensity,
    fps: float = 30.0,
    zmax: float = ZMAX,
    punch_to: float = PUNCH_TO,
    punch_s: float = PUNCH_S,
    out_s: float = OUT_S,
    enter: float = ENTER,
    exit: float = EXIT,
    smooth_win: int = SMOOTH_WIN,
    bridge_gap_s: float = BRIDGE_GAP_S,
    min_interval_s: float = MIN_INTERVAL_S,
):
    """Motion intensity -> per-frame zoom with tapered punch + merged intervals."""
    inten = np.asarray(intensity, dtype=float)
    n = inten.size
    if n == 0:
        return inten

    ref = np.percentile(inten, 95)
    norm = np.clip(inten / (ref + 1e-9), 0.0, 1.0) if ref > 0 else np.zeros(n)
    if smooth_win > 1:
        norm = np.convolve(norm, np.ones(smooth_win) / smooth_win, mode="same")

    # hysteresis -> active mask
    active = np.zeros(n, dtype=bool)
    on = False
    for i, v in enumerate(norm):
        on = (v >= exit) if on else (v > enter)
        active[i] = on

    # merge intervals separated by a short gap; drop too-short bursts
    bridge = int(bridge_gap_s * fps)
    merged: list[list[int]] = []
    for iv in _intervals(active):
        if merged and iv[0] - merged[-1][1] < bridge:
            merged[-1][1] = iv[1]
        else:
            merged.append(iv[:])
    min_len = int(min_interval_s * fps)
    merged = [iv for iv in merged if iv[1] - iv[0] >= min_len]

    # build tapered curve per interval
    z = np.ones(n)
    pf = max(1, int(punch_s * fps))
    of = max(1, int(out_s * fps))
    for k, (s, e) in enumerate(merged):
        dur = e - s
        creep_len = max(1, dur - pf)
        for local in range(dur):
            if local < pf:
                z[s + local] = _smoothstep(1.0, punch_to, local / pf)
            else:
                z[s + local] = punch_to + (zmax - punch_to) * min(1.0, (local - pf) / creep_len)
        # zoom out after the interval, not past the next interval / end
        nxt = merged[k + 1][0] if k + 1 < len(merged) else n
        for f in range(e, min(e + of, nxt, n)):
            z[f] = _smoothstep(zmax, 1.0, (f - e) / of)
    return z


def reframe_clip(src, dst, out_w=1280, out_h=720):
    """Read src video, write a bottom-left punch-zoomed video to dst (no audio)."""
    import cv2  # heavy/optional dep, import lazily

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    inten: list[float] = []
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        inten.append(0.0 if prev is None else float(cv2.absdiff(small, prev).mean()))
        prev = small
    cap.release()

    zoom = zoom_track(inten, fps=fps)

    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    cap = cv2.VideoCapture(str(src))
    i = n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        Hf, Wf = frame.shape[:2]
        z = float(zoom[i]) if i < len(zoom) else 1.0
        cw, ch = int(round(Wf / z)), int(round(Hf / z))
        crop = frame[Hf - ch:Hf, 0:cw]            # bottom-left anchor
        vw.write(cv2.resize(crop, (out_w, out_h)))
        i += 1
        n += 1
    cap.release()
    vw.release()
    if n == 0:
        raise RuntimeError(f"reframe produced no frames from {src}")
    return Path(dst)
