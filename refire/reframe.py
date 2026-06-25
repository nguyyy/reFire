"""Action-aware dynamic reframe: pan toward motion, punch-zoom on action, release.

Per-frame inter-frame motion gives an intensity series + a motion centroid.
Intensity drives a zoom envelope with fast attack / slow release (zooms in on
action, eases back out when it calms). The centroid drives the pan, so the frame
follows whichever of streamer/gameplay is moving most. Fast zoom/pan gets a
temporal blend for a motion-blurred, fluid feel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ZMAX = 1.5         # max punch-in zoom (aggressive)
ATTACK = 0.35      # how fast zoom rises toward action (high = aggressive)
RELEASE = 0.06     # how slow it zooms back out (low = lingers, then releases)
PAN_SMOOTH = 0.15  # centroid easing (low = smoother, laggier pan)
BLUR_GAIN = 4.0    # motion-blur sensitivity to zoom/pan velocity
BLUR_MAX = 0.6     # max temporal-blend strength


def zoom_envelope(intensity, zmax=ZMAX, attack=ATTACK, release=RELEASE):
    """Motion intensity -> per-frame zoom, fast attack / slow release."""
    inten = np.asarray(intensity, dtype=float)
    if inten.size == 0:
        return inten
    ref = np.percentile(inten, 95)
    norm = np.clip(inten / (ref + 1e-9), 0.0, 1.0) if ref > 0 else np.zeros_like(inten)
    target = 1.0 + (zmax - 1.0) * norm
    out = np.empty_like(target)
    z = 1.0
    for i, t in enumerate(target):
        z += (t - z) * (attack if t > z else release)
        out[i] = z
    return out


def smooth_path(centroids, coeff=PAN_SMOOTH):
    """EMA-smooth a list of (x,y) centroids in 0..1 space."""
    pts = np.asarray(centroids, dtype=float)
    out = np.empty_like(pts)
    if pts.size == 0:
        return out
    cur = pts[0].copy()
    for i, p in enumerate(pts):
        cur = cur + (p - cur) * coeff
        out[i] = cur
    return out


def reframe_clip(src, dst, out_w=1280, out_h=720):
    """Read src video, write a dynamically reframed video to dst (no audio)."""
    import cv2  # heavy/optional dep, import lazily

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # pass 1: per-frame motion intensity + centroid (downscaled for speed)
    inten: list[float] = []
    cents: list[tuple[float, float]] = []
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        if prev is None:
            inten.append(0.0)
            cents.append((0.5, 0.5))
        else:
            diff = cv2.absdiff(small, prev)
            inten.append(float(diff.mean()))
            m = cv2.moments(diff)
            if m["m00"] > 1e-6:
                cents.append((m["m10"] / m["m00"] / 160.0, m["m01"] / m["m00"] / 90.0))
            else:
                cents.append((0.5, 0.5))
        prev = small
    cap.release()

    zoom = zoom_envelope(inten)
    path = smooth_path(cents)

    # pass 2: crop toward centroid at the envelope zoom, blend for motion blur
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    cap = cv2.VideoCapture(str(src))
    prev_out = None
    pz, pcx, pcy = 1.0, 0.5, 0.5
    i = n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        z = float(zoom[i]) if i < len(zoom) else 1.0
        cx, cy = (float(path[i][0]), float(path[i][1])) if i < len(path) else (0.5, 0.5)
        cw, ch = W / z, H / z
        x0 = min(max(cx * W - cw / 2, 0), W - cw)
        y0 = min(max(cy * H - ch / 2, 0), H - ch)
        crop = frame[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
        out = cv2.resize(crop, (out_w, out_h))

        vel = (abs(z - pz) + np.hypot(cx - pcx, cy - pcy)) * BLUR_GAIN
        a = float(min(BLUR_MAX, vel))
        if prev_out is not None and a > 0:
            out = cv2.addWeighted(out, 1 - a, prev_out, a, 0)

        vw.write(out)
        prev_out = out
        pz, pcx, pcy = z, cx, cy
        i += 1
        n += 1
    cap.release()
    vw.release()
    if n == 0:
        raise RuntimeError(f"reframe produced no frames from {src}")
    return Path(dst)
