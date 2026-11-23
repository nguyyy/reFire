"""Cheap vision critic (03-thumb section 7, one round): one low-effort call looks at a run's three
rendered candidates beside one of Nguy's thumbnails, with each candidate's measured framing and the
list of mistakes this pipeline has made before (MISTAKES: what it looked like, how to fix it). It
returns bounded moves for the left element and zy0x; they are applied to spec.json and the
candidate is re-rendered. What moves can't fix (a bad cutout, a readable answer) is written to
critic.json as issues, which report.md lists. ponytail: one round, moves only; the spec's 2-round
re-score loop and hook/face-frame swaps come when moves stop being enough.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

W, H = 1280, 720
REF = Path(__file__).with_name("refs") / "stronboxthumbnail - Copy.png"
MODEL = "claude-sonnet-5-5"          # cheap: placement judgement, not story
MOVE = {"dx": 200.0, "dy": 200.0, "scale": (0.8, 1.35)}     # bounds on what one round may do

# every mistake nguy or the measurements caught so far, and the move that fixes it
MISTAKES = [
    ("left element small: its solid part covers well under a quarter of the frame (his: 24-42%)",
     "left.scale 1.1-1.3"),
    ("left element sits in the middle, empty space between it and the left edge (his hug the left and "
     "top edges)", "left.dx negative"),
    ("left element too high: its top or a card's title cut off by the frame", "left.dy positive"),
    ("left element too low: a character's face or a card's key rows behind the hook text", "left.dy negative"),
    ("zy0x small: the top of his head well below ~12% of the height (his 80-90 rule: he fills 88-93%)",
     "face.scale up"),
    ("zy0x not at the right edge: a gap between his far shoulder and the edge, or his face pushed past "
     "~85% of the width (his face centers sit at 72-79% of the width)", "face.dx"),
    ("zy0x's face covered by the hook text or the ??? bubble", "face.dx / face.dy"),
    ("cutout flaws: his head sliced flat by the facecam edge, chunks of wall or chair kept", "issue: another frame"),
    ("a reveal whose answer is still readable (\"Challenge Completed\" under \"did he clear it?\")",
     "issue: hide it"),
    ("a dark left element or background (a stream screen, a dark gameplay frame)",
     "issue: bright art, a designed card, a real game backdrop"),
    ("the left element isn't what the hook is about (the banner instead of the character chased)",
     "issue: the subject"),
]

SYSTEM = f"""You check YouTube thumbnails for zy0x's Genshin channel against his editor's own style (the
REF tile is one of his). You see a sheet with tiles A, B, C (candidates) and REF, plus measurements.
Judge placement and framing only: the hook's words, font and colors are already fixed.

Mistakes this pipeline has made before, and the fix for each:
{chr(10).join(f"- {m} -> {f}" for m, f in MISTAKES)}

Moves you can make per candidate (screen px on the 1280x720 frame; 0 / 1.0 = leave it):
- left: {{"dx", "dy", "scale"}} moves/resizes the left element (art, card or stream screen)
- face: {{"dx", "dy", "scale"}} moves/resizes zy0x (his cut edge always stays on the frame bottom)
dx and dy within +-{MOVE['dx']:.0f}, scale within {MOVE['scale'][0]}-{MOVE['scale'][1]}. Only move what is
wrong; a good candidate gets zeros. Anything a move can't fix goes in issues, short.

Reply with ONLY JSON:
{{"candidates": [{{"id": "A", "score": 0-100, "scores": {{"placement": 1-5, "framing": 1-5, "readability": 1-5}},
  "issues": ["..."], "left": {{"dx": 0, "dy": 0, "scale": 1.0}}, "face": {{"dx": 0, "dy": 0, "scale": 1.0}}}}]}}"""


def _models():
    from pydantic import BaseModel

    class Move(BaseModel):
        dx: float = 0.0
        dy: float = 0.0
        scale: float = 1.0

    class Verdict(BaseModel):
        id: str
        score: int = 0
        scores: dict = {}
        issues: list[str] = []
        left: Move = Move()
        face: Move = Move()

    class Out(BaseModel):
        candidates: list[Verdict]

    return Out


def clamp(m: dict) -> dict:
    lo, hi = MOVE["scale"]
    return {"dx": max(-MOVE["dx"], min(MOVE["dx"], float(m.get("dx", 0)))),
            "dy": max(-MOVE["dy"], min(MOVE["dy"], float(m.get("dy", 0)))),
            "scale": max(lo, min(hi, float(m.get("scale", 1.0))))}


def _img_hw(p: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(p) as im:
        return im.height, im.width


def move_left(spec: dict, d: Path, m: dict) -> None:
    """shift/scale the scene card by screen px about its center (world units = screen / k)."""
    from .pipeline import FOV
    cards = (spec.get("scene") or {}).get("cards") or []
    if not cards:
        return
    c = cards[0]
    k = (H / 2) / math.tan(math.radians(FOV / 2))
    k = k / (k - c.get("z", 0))
    ih, iw = _img_hw(d / c["src"])
    h = c.get("h") or c["w"] * ih / iw
    cx, cy = c["x"] + c["w"] / 2 + m["dx"] / k, c["y"] + h / 2 + m["dy"] / k
    c["w"] *= m["scale"]
    if "h" in c:
        c["h"] *= m["scale"]
    h *= m["scale"]
    c["x"], c["y"] = round(cx - c["w"] / 2, 1), round(cy - h / 2, 1)
    c["w"] = round(c["w"], 1)


def move_face(spec: dict, d: Path, m: dict) -> None:
    """shift/scale zy0x about his face center; his cam crop's cut edge never rises above the frame bottom."""
    L = next((L for L in spec.get("layers", []) if L.get("kind") == "face"), None)
    if L is None:
        return
    ih, iw = _img_hw(d / L["src"])
    fb = L.get("face_box") or [L["x"], L["y"], L["w"], L["w"] * ih / iw]
    fcx, fcy = fb[0] + fb[2] / 2, fb[1] + fb[3] / 2
    s = m["scale"]
    x, y, w = fcx - (fcx - L["x"]) * s + m["dx"], fcy - (fcy - L["y"]) * s + m["dy"], L["w"] * s
    bottom = y + w * ih / iw
    if bottom < H:                                # never show the crop's cut edge
        m = {**m, "dy": m["dy"] + H - bottom}
        y += H - bottom
    L.update(x=round(x, 1), y=round(y, 1), w=round(w, 1))
    L["face_box"] = [round(fcx - (fcx - fb[0]) * s + m["dx"], 1), round(fcy - (fcy - fb[1]) * s + m["dy"], 1),
                     round(fb[2] * s, 1), round(fb[3] * s, 1)]


def measure(d: Path, spec: dict) -> str:
    """the framing numbers the critic gets (the same ones the refs were measured with)."""
    import cv2

    from .pipeline import project
    out = []
    cards = (spec.get("scene") or {}).get("cards") or []
    if cards:
        c = cards[0]
        im = cv2.imread(str(d / c["src"]), cv2.IMREAD_UNCHANGED)
        ih, iw = im.shape[:2]
        x0, y0, x1, y1 = project(c, c["w"] / c["h"] if "h" in c else iw / ih)
        a = im[:, :, 3] if im.shape[2] == 4 else np.full((ih, iw), 255, np.uint8)
        bw, bh = max(1, round(x1 - x0)), max(1, round(y1 - y0))
        mask = cv2.resize(a, (bw, bh)) > 128
        frame = np.zeros((H, W), bool)
        X, Y = round(x0), round(y0)
        fx0, fy0, fx1, fy1 = max(0, X), max(0, Y), min(W, X + bw), min(H, Y + bh)
        if fx1 > fx0 and fy1 > fy0:
            frame[fy0:fy1, fx0:fx1] = mask[fy0 - Y:fy1 - Y, fx0 - X:fx1 - X]
        ys, xs = np.where(frame)
        if len(xs):
            out.append(f"left element covers {100 * frame.mean():.0f}% of the frame, solid part x {xs.min()}-{xs.max()}, "
                       f"y {ys.min()}-{ys.max()}")
    L = next((L for L in spec.get("layers", []) if L.get("kind") == "face"), None)
    if L and L.get("face_box"):
        im = cv2.imread(str(d / L["src"]), cv2.IMREAD_UNCHANGED)
        s = L["w"] / im.shape[1]
        rows = np.where((im[:, :, 3] > 128).mean(axis=1) >= 0.02)[0]
        top = L["y"] + (rows.min() if len(rows) else 0) * s
        fb = L["face_box"]
        out.append(f"zy0x: head top y {top:.0f} (fills {100 * (H - top) / H:.0f}% of the height), face center x "
                   f"{fb[0] + fb[2] / 2:.0f} ({100 * (fb[0] + fb[2] / 2) / W:.0f}% of the width), face height "
                   f"{100 * fb[3] / H:.0f}% of the frame")
    return "; ".join(out)


def sheet(root: Path, ids: list[str]) -> Path:
    """A, B, C and his REF on one 2x2 sheet, labelled: one image to read."""
    from PIL import Image, ImageDraw
    tiles = [(k, root / k / "thumb.png") for k in ids] + [("REF", REF)]
    out = Image.new("RGB", (2 * W, 2 * H), "black")
    for i, (k, p) in enumerate(tiles[:4]):
        t = Image.open(p).convert("RGB").resize((W, H))
        dr = ImageDraw.Draw(t)
        dr.rectangle([0, 0, 150, 70], fill="black")
        dr.text((14, 10), k, fill="yellow", font_size=48)
        out.paste(t, ((i % 2) * W, (i // 2) * H))
    p = root / "critic_sheet.jpg"
    out.save(p, quality=88)
    return p


def review(root: Path, browser=None, model: str = MODEL, ask=None, log=print) -> list[dict]:
    """one call over a run's candidates -> moves applied + re-rendered, critic.json per candidate."""
    from .. import director
    from . import render
    root = Path(root)
    ids = [k for k in "ABC" if (root / k / "spec.json").exists()]
    if not ids:
        return []
    specs = {k: json.loads((root / k / "spec.json").read_text(encoding="utf-8")) for k in ids}
    p = sheet(root, ids)
    facts = "\n".join(f"{k}: {measure(root / k, specs[k])}" for k in ids)
    user = f"Read {p.name} (tiles {', '.join(ids)}, REF). Measurements:\n{facts}"
    Out = _models()
    try:
        out = ask(user, Out) if ask else director._complete_cli(
            model, SYSTEM, [{"type": "text", "text": user}], Out, tag="thumb critic", effort="low", live=False,
            tools="Read", cwd=root)
    except (RuntimeError, OSError) as e:                 # no paid fallback: the critic is optional polish
        log(f"[thumb critic] skipped ({str(e)[:120]})")
        return []
    done = []
    for v in out.candidates:
        if v.id not in specs:
            continue
        d, spec = root / v.id, specs[v.id]
        left, face = clamp(v.left.model_dump()), clamp(v.face.model_dump())
        moved = any(abs(m["dx"]) > 0.5 or abs(m["dy"]) > 0.5 or abs(m["scale"] - 1) > 1e-3 for m in (left, face))
        if moved:
            move_left(spec, d, left)
            move_face(spec, d, face)
            (d / "spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
            render.render(d / "spec.json", browser=browser)
        rec = {"score": v.score, "scores": v.scores, "issues": v.issues, "left": left, "face": face, "moved": moved}
        (d / "critic.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        log(f"[thumb critic] {v.id}: {v.score} {'moved' if moved else 'kept'}; {'; '.join(v.issues)[:140]}")
        done.append({"id": v.id, **rec})
    return done
