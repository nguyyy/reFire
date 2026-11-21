"""Frames from a run -> assets + spec.json for a layout (docs/specs/03-thumb.md steps 3, 5, 6).

Layout math lives here, the look lives in house.css, the renderer only draws what the spec
says. A candidate folder is self-contained: <run_dir>/thumb/<A|B|C>/{spec.json, assets/}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CANVAS = (1280, 720)
FIGURE = (0.80, 0.90)           # head top -> frame bottom / canvas height (Nguy, T6)
FIGURE_TARGET = 0.88
FACE_MAX = 0.62                 # never let a tight cam crop blow the face past this
BADGE = (0.16, 0.12)            # youtube duration badge: bottom-right width x height


def face_box(img_bgr: np.ndarray, detector=None) -> list[float] | None:
    """largest confident face in an image -> [x, y, w, h] pixels."""
    from ..facecam import MIN_SCORE, YuNet
    det = detector or YuNet()
    h, w = img_bgr.shape[:2]
    faces = [f for f in det(np.ascontiguousarray(img_bgr[:, :, :3])) if f[4] >= MIN_SCORE]
    if not faces:
        return None
    x, y, fw, fh, _ = max(faces, key=lambda f: f[2] * f[3])
    return [float(x * w), float(y * h), float(fw * w), float(fh * h)]


def head_top(cut_bgra: np.ndarray, frac: float = 0.02) -> int:
    """first row of the cutout with real figure in it (alpha > 50% across >= frac of the width)."""
    a = cut_bgra[:, :, 3] > 128
    rows = np.flatnonzero(a.sum(axis=1) >= frac * a.shape[1])
    return int(rows[0]) if rows.size else 0


def place_face(cut_wh: tuple[int, int], box: list[float], top: float, canvas=CANVAS, side: str = "right",
               target: float = FIGURE_TARGET) -> dict:
    """zy0x fills `target` of the frame from the top of his head down, running off the bottom
    (refs, Nguy: ~80-90%); face centered on the right (or left) third."""
    W, H = canvas
    cw, ch = cut_wh
    fx, fy, fw, fh = box
    s = target * H / max(1.0, ch - top)
    s = min(s, FACE_MAX * H / fh)
    y = H - ch * s                                           # the cam crop's cut edge sits on the frame bottom
    cx = (0.74 if side == "right" else 0.26) * W
    x = cx - (fx + fw / 2) * s
    return {"x": round(x, 1), "y": round(y, 1), "w": round(cw * s, 1),
            "face": [round(x + fx * s, 1), round(y + fy * s, 1), round(fw * s, 1), round(fh * s, 1)]}


def place_subject(wh: tuple[int, int], canvas=CANVAS, side: str = "left", max_w: float = 0.36,
                  max_h: float = 0.60) -> dict:
    W, H = canvas
    w, h = wh
    s = min(max_w * W / w, max_h * H / h)
    x = 0.05 * W if side == "left" else W - 0.05 * W - w * s
    return {"x": round(x, 1), "y": round(0.06 * H, 1), "w": round(w * s, 1)}


def badge_clear(box: list[float], canvas=CANVAS) -> bool:
    """True when a pixel box stays out of the bottom-right duration-badge zone."""
    W, H = canvas
    x, y, w, h = box
    return x + w <= W * (1 - BADGE[0]) or y + h <= H * (1 - BADGE[1])


def reaction_spec(face_cut: Path, box: list[float], subject: Path | None, bg: Path, hook: dict,
                  canvas=CANVAS, glow: str = "#ffffff") -> dict:
    """the `reaction` layout: face large right, subject small left, hook bottom-left."""
    from PIL import Image
    im = Image.open(face_cut)
    f = place_face(im.size, box, head_top(np.asarray(im.convert("RGBA"))[:, :, [2, 1, 0, 3]]), canvas)
    layers = []
    if subject:
        s = place_subject(Image.open(subject).size, canvas)
        layers.append({"kind": "subject", "src": f"assets/{subject.name}", **s, "rotate": -3, "shadow": 0.6})
    layers.append({"kind": "face", "src": f"assets/{face_cut.name}", "x": f["x"], "y": f["y"], "w": f["w"],
                   "glow": glow, "face_box": f["face"]})
    return {"version": 1, "layout": "reaction", "canvas": list(canvas),
            "background": {"src": f"assets/{bg.name}", "blur": 12, "brightness": 0.62, "saturate": 1.15,
                           "scale": 1.08, "vignette": 0.55},
            "layers": layers, "text": {"anchor": "bottom-left", **hook}}


class RunFrames:
    """full-res frames and facecam crops of a run at timeline times (proxies on V1)."""

    def __init__(self, run_dir, timeline=None):
        from ..facecam import Facecams
        from ..timeline_map import TimelineMap
        mp = Path(run_dir) / "ae" / "manifest.json"
        self.tmap = TimelineMap.load(mp, timeline)
        self.cams = Facecams(mp, timeline)

    def frame(self, t: float) -> np.ndarray:
        hit = self.tmap.timeline_to_source(t)
        if not hit:
            raise ValueError(f"nothing of ours on V1 at {t:.2f}s")
        fr = self.cams.frames(hit[0], [hit[1]], width=0)
        if not fr:
            raise ValueError(f"could not decode a frame at {t:.2f}s")
        return fr[0]

    def facecam(self, t: float) -> np.ndarray:
        f = self.frame(t)
        cam = self.cams.detect(self.tmap.timeline_to_source(t)[0])
        if cam is None:
            raise ValueError("no facecam found for this source")
        h, w = f.shape[:2]
        x0, y0, x1, y1 = cam.box
        return f[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))]


def draft_reaction(run_dir, letter: str, face_t: float, hook: dict, subject_t: float | None = None,
                   subject_box: list[float] | None = None, timeline=None, model: str = "lite") -> Path:
    """one `reaction` candidate from explicit choices -> <run_dir>/thumb/<letter>/spec.json.
    (the concept step picks face_t/subject/hook; this is the deterministic half.)"""
    import cv2

    from .cutout import cutout
    rf = RunFrames(run_dir, timeline)
    d = Path(run_dir) / "thumb" / letter
    (d / "assets").mkdir(parents=True, exist_ok=True)
    cut = cutout(rf.facecam(face_t), model=model)
    box = face_box(cut)
    if box is None:
        raise ValueError(f"no face in the facecam at {face_t:.2f}s")
    cv2.imwrite(str(d / "assets" / "face.png"), cut)
    sub = None
    if subject_t is not None:
        fr = rf.frame(subject_t)
        if subject_box:
            h, w = fr.shape[:2]
            x0, y0, x1, y1 = subject_box
            fr = fr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        sub = d / "assets" / "subject.png"
        cv2.imwrite(str(sub), fr)
    bg = d / "assets" / "bg.jpg"
    cv2.imwrite(str(bg), rf.frame(subject_t if subject_t is not None else face_t), [cv2.IMWRITE_JPEG_QUALITY, 90])
    spec = reaction_spec(d / "assets" / "face.png", box, sub, bg, hook)
    (d / "spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    return d / "spec.json"
