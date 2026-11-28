"""Split-screen when a gameplay zoom and a facecam punch overlap (04-sauce.md 4.2 + 6; P5).

Both can't zoom V1 at once, so the overlap becomes one pre-rendered comp on V4 (spec 6: Premiere
masks from script are painful). Nguy's own (Odette video, 0:15-0:21): the gameplay blurred full-frame
behind two floating cards, his face on the left (~1/3), the zoom's window of the gameplay on the right
(~2/3), drop shadows. Card boxes measured off that frame at 1080p. Hard cut in and out like the punch
and the zoom (ponytail: his cards slide in; add a scale-in if he misses it). ffmpeg, video only.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

FACE_BOX = (144, 276, 594, 525)  # x, y, w, h at 1920x1080 (Odette 0:17)
GAME_BOX = (910, 292, 920, 504)
BLUR = 40                        # background gaussian sigma
MIN_LEN = 0.75                   # s; a shorter overlap stays a plain zoom over the punch
PREFIX = "refire:sauce:"


def overlaps(zooms: list[dict], punches: list[tuple[float, float]]) -> list[dict]:
    """each zoom x punch overlap long enough to show -> a split event (zoom's window kept)."""
    out = []
    for z in zooms:
        for a, b in punches:
            t0, t1 = max(z["t"], a), min(z["end"], b)
            if t1 - t0 >= MIN_LEN:
                out.append({"type": "split", "t": round(t0, 3), "end": round(t1, 3), "zoom": z["id"],
                            "anchor": z["anchor"], "scale": z["scale"], "confidence": z["confidence"],
                            "text": z.get("text", ""), "reason": f"zoom {z['id']} meets a facecam punch"})
    for i, e in enumerate(sorted(out, key=lambda e: e["t"]), 1):
        e["id"] = f"s{i:02d}"
    return out


FACE_H = 0.8                     # of the facecam box's height: its box runs ~10% into the gameplay above (seen live)


def crops(anchor: list[float], scale: float, cam: list[float], W: int, H: int,
          face_center: list[float] | None = None) -> tuple[list[int], list[int]]:
    """(gameplay crop, face crop) as [x, y, w, h] px of the source frame, each with its card's aspect
    and kept inside the frame. cam: the facecam box [x0, y0, x1, y1] normalized; face_center: his
    face in it (normalized), the face crop centers there."""

    def box(cx, cy, h, aspect, lim):                      # lim: [x0, y0, x1, y1] px it must stay inside
        h = min(h, lim[3] - lim[1], (lim[2] - lim[0]) / aspect)
        w = h * aspect
        x = min(max(cx - w / 2, lim[0]), lim[2] - w)
        y = min(max(cy - h / 2, lim[1]), lim[3] - h)
        return [round(x), round(y), round(w), round(h)]

    game = box(anchor[0] * W, anchor[1] * H, H / scale, GAME_BOX[2] / GAME_BOX[3], [0, 0, W, H])
    c = [cam[0] * W, cam[1] * H, cam[2] * W, cam[3] * H]
    fc = [face_center[0] * W, face_center[1] * H] if face_center else [(c[0] + c[2]) / 2, (c[1] + c[3]) / 2]
    face = box(fc[0], fc[1], FACE_H * (c[3] - c[1]), FACE_BOX[2] / FACE_BOX[3], [c[0], c[1] + (1 - FACE_H) * (c[3] - c[1]), c[2], c[3]])
    return game, face


def render(media: str, t0: float, dur: float, game: list[int], face: list[int], out: Path, fps: float,
           size=(1920, 1080)) -> Path:
    """the comp, from the source file's local time t0, as ProRes 422 (video only)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    k = size[0] / 1920

    def card(label, crop, b):
        x, y, w, h = (round(v * k) for v in b)
        return (f"[{label}]crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]},scale={w}:{h},setsar=1[{label}c];"
                f"color=black@0.6:s={w}x{h}:d=1,format=rgba,pad={w + 60}:{h + 60}:30:30:color=black@0,"
                f"gblur=sigma=14,loop=-1:1[{label}s];"), (x, y)
    fcard, (fx, fy) = card("f", face, FACE_BOX)
    gcard, (gx, gy) = card("g", game, GAME_BOX)
    fc = (f"[0:v]split=3[bg][f][g];[bg]scale={size[0]}:{size[1]},setsar=1,gblur=sigma={BLUR},eq=brightness=-0.06[b];"
          f"{fcard}{gcard}"
          f"[b][fs]overlay={fx - 22}:{fy - 18}:shortest=1[b1];[b1][fc]overlay={fx}:{fy}[b2];"
          f"[b2][gs]overlay={gx - 22}:{gy - 18}:shortest=1[b3];[b3][gc]overlay={gx}:{gy},fps={fps:g}[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", media,
                    "-filter_complex", fc, "-map", "[v]", "-an", "-c:v", "prores_ks", "-profile:v", "2",
                    "-pix_fmt", "yuv422p10le", str(out)], check=True)
    return out
