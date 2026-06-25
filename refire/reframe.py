"""Motion-triggered punch zoom, fixed bottom-left anchor (the webcam corner).

Per-frame inter-frame motion gives an intensity series. A hysteresis state
machine turns sustained motion bursts into a stable binary target (100% or
200%), which is then eased into a smooth, quick zoom ramp. The crop is anchored
to the bottom-left corner, so there is no panning and no per-frame jitter — the
shake from continuous centroid-following / oscillating zoom is gone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ZMAX = 2.0          # punch-in zoom (200%)
ENTER = 0.45        # normalized motion to START a zoom (high -> only real bursts)
EXIT = 0.18         # normalized motion to END a zoom (low -> hysteresis, no flicker)
MIN_HOLD_S = 0.8    # min seconds to stay zoomed once triggered
SMOOTH_WIN = 9      # frames of rolling-mean on motion (kills single-frame spikes)
EASE = 0.20         # zoom ramp speed toward target (quick but smooth)


def zoom_track(
    intensity,
    fps: float = 30.0,
    zmax: float = ZMAX,
    enter: float = ENTER,
    exit: float = EXIT,
    min_hold_s: float = MIN_HOLD_S,
    smooth_win: int = SMOOTH_WIN,
    ease: float = EASE,
):
    """Motion intensity -> stable per-frame zoom (no oscillation -> no shake)."""
    inten = np.asarray(intensity, dtype=float)
    n = inten.size
    if n == 0:
        return inten

    ref = np.percentile(inten, 95)
    norm = np.clip(inten / (ref + 1e-9), 0.0, 1.0) if ref > 0 else np.zeros(n)
    if smooth_win > 1:
        norm = np.convolve(norm, np.ones(smooth_win) / smooth_win, mode="same")

    # hysteresis + minimum hold -> a stable binary target (no rapid toggling)
    target = np.ones(n)
    zoomed = False
    hold = max(1, int(min_hold_s * fps))
    since = 0
    for i, v in enumerate(norm):
        if zoomed:
            since += 1
            if v < exit and since >= hold:
                zoomed = False
        elif v > enter:
            zoomed = True
            since = 0
        target[i] = zmax if zoomed else 1.0

    # ease toward the stable target -> smooth quick ramps, flat holds
    out = np.empty(n)
    z = 1.0
    for i, t in enumerate(target):
        z += (t - z) * ease
        out[i] = z
    return out


def reframe_clip(src, dst, out_w=1280, out_h=720):
    """Read src video, write a bottom-left punch-zoomed video to dst (no audio)."""
    import cv2  # heavy/optional dep, import lazily

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # pass 1: per-frame motion intensity (downscaled for speed)
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

    # pass 2: crop anchored to the bottom-left corner at the eased zoom
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
