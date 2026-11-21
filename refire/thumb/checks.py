"""Deterministic thumbnail checks (docs/specs/03-thumb.md step 7, "fail = must fix").

Each check reads a rendered candidate folder (spec.json + thumb.png + assets/hook.png) and
returns {name, ok, value, need}. No LLM; the vision critic scores on top of these.
"""
from __future__ import annotations

import colorsys
import json
from pathlib import Path

import numpy as np

from .compose import BADGE
from .text import PALETTE

SMALL = ((246, 138), (168, 94))      # feed sizes the hook must survive
PRIMARY = {"face", "subject", "hook", "bubble", "glyph", "art", "inset"}


def _r(name, ok, value, need):
    return {"name": name, "ok": bool(ok), "value": value, "need": need}


def hook_text(spec: dict) -> str:
    return " ".join(t if isinstance(t, str) else t["text"] for t in (spec.get("text") or {}).get("lines", []))


def readable(png: Path, text: str, ocr=None) -> dict:
    """the hook still OCRs at feed size: char accuracy >= 0.80 at both small sizes."""
    import difflib

    import cv2
    if not text:
        return _r("hook readable small", True, None, ">= 0.80")
    if ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        eng = RapidOCR()
        ocr = lambda im: " ".join(r[1] for r in (eng(im, use_cls=False)[0] or []))   # noqa: E731
    img = cv2.imread(str(png))
    want = "".join(text.upper().split())
    scores = []
    for w, h in SMALL:
        small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        got = "".join(ocr(cv2.resize(small, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)).upper().split())
        m = difflib.SequenceMatcher(None, want, got, autojunk=False)
        scores.append(round(sum(b.size for b in m.get_matching_blocks()) / max(1, len(want)), 2))
    return _r("hook readable small", min(scores) >= 0.80, scores, ">= 0.80")


def _face(spec: dict) -> list | None:
    return next((L.get("face_box") for L in spec.get("layers", []) if L.get("kind") == "face"), None)


def face_size(spec: dict) -> dict:
    fb, H = _face(spec), spec.get("canvas", [1280, 720])[1]
    v = round(fb[3] / H, 3) if fb else 0
    return _r("face size", v >= 0.22, v, ">= 0.22")


def _overlap(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy


def face_clear(spec: dict, hook_box: list | None) -> dict:
    """nothing drawn over the eyes/mouth: the face box's inner 60% vs later layers + the hook."""
    fb = _face(spec)
    if not fb:
        return _r("face clear", False, "no face", "no overlap")
    x, y, w, h = fb
    core = [x + 0.2 * w, y + 0.2 * h, 0.6 * w, 0.7 * h]
    after, seen = [], False
    for L in spec.get("layers", []):
        if L.get("kind") == "face":
            seen = True
            continue
        if seen and "w" in L:
            after.append([L["x"], L["y"], L["w"], L.get("h", L["w"])])
    if hook_box:
        after.append(hook_box)
    hit = max((_overlap(core, b) / (core[2] * core[3]) for b in after), default=0)
    return _r("face clear", hit < 0.02, round(hit, 3), "< 0.02 of the face core covered")


def badge_zone(spec: dict, hook_box: list | None) -> dict:
    W, H = spec.get("canvas", [1280, 720])
    zone = [W * (1 - BADGE[0]), H * (1 - BADGE[1]), W * BADGE[0], H * BADGE[1]]
    fb = _face(spec)
    boxes = [b for b in (hook_box, fb and [fb[0] + 0.2 * fb[2], fb[1] + 0.2 * fb[3], 0.6 * fb[2], 0.7 * fb[3]]) if b]
    hit = max((_overlap(zone, b) for b in boxes), default=0)
    return _r("badge zone", hit == 0, round(hit), "0 px of hook/face core in the bottom-right")


def dominant_hue(img_rgb: np.ndarray) -> float:
    """saturation-weighted circular mean hue (degrees) of an image."""
    px = img_rgb.reshape(-1, 3)[::7] / 255.0
    hs = np.array([colorsys.rgb_to_hsv(*p)[:2] for p in px])
    a = hs[:, 0] * 2 * np.pi
    w = hs[:, 1] ** 2 + 1e-6
    return float(np.degrees(np.arctan2((np.sin(a) * w).sum(), (np.cos(a) * w).sum())) % 360)


def accent_contrast(spec: dict, bg_rgb: np.ndarray) -> dict:
    t = spec.get("text") or {}
    if not t.get("accent_phrase"):
        return _r("accent vs background", True, None, ">= 60 deg")
    base = PALETTE.get(t.get("accent", "gold"), PALETTE["gold"])[0]
    r, g, b = (int(base[i:i + 2], 16) / 255 for i in (1, 3, 5))
    ah = colorsys.rgb_to_hsv(r, g, b)[0] * 360
    d = abs(ah - dominant_hue(bg_rgb)) % 360
    d = round(min(d, 360 - d), 1)
    return _r("accent vs background", d >= 60, d, ">= 60 deg")


def clutter(spec: dict) -> dict:
    n = sum(1 for L in spec.get("layers", []) if L.get("kind") in PRIMARY)
    n += len((spec.get("scene") or {}).get("cards", []))
    n += 1 if hook_text(spec) else 0
    return _r("clutter", n <= 4, n, "<= 4 primary elements")


def phash(png: Path) -> int:
    """64-bit dct perceptual hash."""
    import cv2
    g = cv2.resize(cv2.imread(str(png), cv2.IMREAD_GRAYSCALE), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8].flatten()
    bits = d > np.median(d[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def candidate(d: Path, ocr=None) -> list[dict]:
    """every single-candidate check for a rendered folder."""
    import cv2
    spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
    png = next(p for p in (d / "thumb.png", d / "thumb.jpg") if p.exists())
    hook = None
    hj = d / "assets" / "hook.json"
    if hj.exists() and hook_text(spec):
        i = json.loads(hj.read_text(encoding="utf-8"))
        hook = [i["x"], i["y"], i["w"], i["h"]]
    rgb = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
    return [readable(png, hook_text(spec), ocr), face_size(spec), face_clear(spec, hook), badge_zone(spec, hook),
            accent_contrast(spec, rgb), clutter(spec)]


def distinct(dirs: list[Path], types: list[str] | None = None) -> list[dict]:
    """across candidates: different concept types, pairwise pHash distance >= 16."""
    out = []
    if types is not None:
        out.append(_r("three concept types", len(set(types)) == len(types), types, "all different"))
    hs = [phash(next(p for p in (d / "thumb.png", d / "thumb.jpg") if p.exists())) for d in dirs]
    pairs = [hamming(a, b) for i, a in enumerate(hs) for b in hs[i + 1:]]
    out.append(_r("distinct images", min(pairs, default=64) >= 16, pairs, "pairwise pHash >= 16"))
    return out
