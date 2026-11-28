"""Split-screen when a gameplay zoom and a facecam punch overlap (04-sauce.md 4.2 + 6; P5).

Both can't zoom V1 at once, so the overlap becomes one pre-rendered comp on V4 (spec 6: Premiere
masks from script are painful): the zoom's window of the gameplay on GAME_W of the frame, his face
(the facecam box, cropped tall) on the rest, hard cut in and out like the punch and the zoom.
Nguy: 2/3 + 1/3. Rendered from the source footage with ffmpeg, video only (V1 keeps the sound).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

GAME_W = 2 / 3                   # gameplay share of the width (Nguy: 1/3 2/3)
GAME_SIDE = "left"               # which side the gameplay sits on
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


def crops(anchor: list[float], scale: float, cam: list[float], W: int, H: int, size=(1920, 1080),
          game_w: float = GAME_W, face_center: list[float] | None = None) -> tuple[list[int], list[int]]:
    """(gameplay crop, face crop) as [x, y, w, h] px of the source frame, each with its pane's aspect
    and kept inside the frame. cam: the facecam box [x0, y0, x1, y1] normalized; face_center: his
    face in it (normalized), the face crop centers there."""
    gw, fw, ph = round(size[0] * game_w), size[0] - round(size[0] * game_w), size[1]

    def box(cx, cy, h, aspect, lim):                      # lim: [x0, y0, x1, y1] px it must stay inside
        h = min(h, lim[3] - lim[1], (lim[2] - lim[0]) / aspect)
        w = h * aspect
        x = min(max(cx - w / 2, lim[0]), lim[2] - w)
        y = min(max(cy - h / 2, lim[1]), lim[3] - h)
        return [round(x), round(y), round(w), round(h)]

    game = box(anchor[0] * W, anchor[1] * H, H / scale, gw / ph, [0, 0, W, H])
    c = [cam[0] * W, cam[1] * H, cam[2] * W, cam[3] * H]
    fc = [face_center[0] * W, face_center[1] * H] if face_center else [(c[0] + c[2]) / 2, (c[1] + c[3]) / 2]
    face = box(fc[0], fc[1], FACE_H * (c[3] - c[1]), fw / ph, [c[0], c[1] + (1 - FACE_H) * (c[3] - c[1]), c[2], c[3]])
    return game, face


def render(media: str, t0: float, dur: float, game: list[int], face: list[int], out: Path, fps: float,
           size=(1920, 1080), game_w: float = GAME_W, side: str = GAME_SIDE) -> Path:
    """the comp, from the source file's local time t0, as ProRes 422 (video only)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    gw, fw, ph = round(size[0] * game_w), size[0] - round(size[0] * game_w), size[1]
    order = "[g][f]" if side == "left" else "[f][g]"
    fc = (f"[0:v]split[a][b];[a]crop={game[2]}:{game[3]}:{game[0]}:{game[1]},scale={gw}:{ph},setsar=1[g];"
          f"[b]crop={face[2]}:{face[3]}:{face[0]}:{face[1]},scale={fw}:{ph},setsar=1[f];{order}hstack=2,fps={fps:g}[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", media,
                    "-filter_complex", fc, "-map", "[v]", "-an", "-c:v", "prores_ks", "-profile:v", "2",
                    "-pix_fmt", "yuv422p10le", str(out)], check=True)
    return out
