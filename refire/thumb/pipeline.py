"""`refire thumb <run_dir>`: three thumbnail candidates from a run folder alone (03-thumb.md).

context (outline beats + the cut's loudest/exclamatory moments) -> ONE Claude call writing
the brief and three concepts (spec: two calls; merged to halve subscription use, both files
still written) -> per concept: best facecam frame near its moment, zy0x cutout, subject
(stream UI region or official art), layout in the 3D scene, accent auto-fixed against the
background hue -> render -> deterministic checks -> report.md. Each run lands in
<run_dir>/thumb/v<k>/ so repeated runs sit side by side; later runs are told the earlier
hooks so they bet differently.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

TYPES = ("reaction", "subject", "reveal")
LINE_MAX = 13         # hook characters per line, spaces included (refs: 6-13, ~10; nguy: "around 10")
REGIONS = {"full": (0, 0, 1, 1), "left-half": (0, 0, 0.5, 1), "right-half": (0.5, 0, 1, 1),
           "center": (0.2, 0.08, 0.8, 0.92), "right-panel": (0.64, 0.06, 0.99, 0.94),
           "left-panel": (0.01, 0.06, 0.36, 0.94), "top-half": (0, 0, 1, 0.5), "bottom-half": (0, 0.5, 1, 1)}
PALETTE_THEMES = "gold: artifacts/rewards; sky: Cryo/Snezhnaya; green or mint: easy/right/wins; cyan: people/reviews; blue: series/lists"

SYSTEM = f"""You direct thumbnails for zy0x's Genshin Impact stream edits (YouTube channel). You pick
three thumbnail concepts for YouTube Test & Compare, which picks a winner by WATCH TIME, so the
three must be three different reasons to click, and each must be paid off by the cut (cite the
outline beats that deliver it; a concept without payoff beats is rejected).

House style (his thumbnails run 8-11% CTR, keep to it): a two-line ALL-CAPS hook, 2-4 words
ideal, 5 max, about 10 characters per line (12 at most, counting spaces and punctuation; his:
"EASIEST / ABYSS?", "MORE / TIER LISTS", "STRONGBOX... / GONE RIGHT?"), no swearing;
questions/teasers end in "?" or "..."; one accent phrase in one accent color ({PALETTE_THEMES});
a small lowercase annotation is rare, only when it adds a fact, e.g. "(birthday)".
zy0x's face is huge on the right (shock, deadpan, skeptical, hype, laughing). The left side holds
ONE clear item that says what the video is about, big: whatever conveys it best, not only
characters or artifacts (a weapon, a boss, a wish result, a tier list, an artifact card, ...).

Concept types, use each exactly once:
- reaction: the strongest asset is his face at a moment.
- subject: the video is about a thing (an artifact, a character, a tier list); subject big.
- reveal: hide the answer (a stat, a tier placement, a pull result) behind a pixel mosaic with a
  "???" bubble; the hook is a question.

Pick face.t ONLY from the given moments (source seconds) and say the expression. The subject
must be deliberate, one item: either a game UI panel that was on screen (source "panel": t = source
seconds when the panel (artifact card, tier list, wish result, character grid) is showing, region =
where it sits: {", ".join(REGIONS)}), or official art of ONE thing (source "art": query = one exact
in-game name: a character, weapon, material, enemy or boss, e.g. "Skirk", "Primogem", "Lord of the
Hidden Depths: Whisperer of Nightmares"), or a graphic you design (source "made": made =
{{"kind": "tierlist", "rows": [{{"label": "GOD TIER", "color": "#FF7F7F", "chars": ["Furina", ...]}}, ...],
"hidden": [[row, col]]}} with 3 rows x up to 4 Genshin characters (the ones the video is about), labels like the stream's own
(GOD TIER / OKAY / RUN AWAY or T0 / T0.5 / T67); or {{"kind": "stat_card", "title": exact artifact
piece name, "set": its artifact set, "subtitle": slot (Flower of Life / Plume of Death / Sands of Eon /
Goblet of Eonothem / Circlet of Logos), "main": [stat, value], "stars": 5, "level": "+20",
"rows": [[stat, value], ...] (4 substats), "hidden": [row indexes, the first rows: the lower ones sit
behind the hook]}}: rendered one-to-one as the in-game artifact card with its icon). Design it from what was said on stream; prefer it when no
clean panel was shown. Never chat, never a random shot.
Title <= 60 characters, paired with the hook.

Reply with ONLY JSON:
{{"brief": {{"promise": "...", "entities": ["..."], "reveal": {{"what": "...", "t": 0, "beat_id": 0}} or null,
  "moments": [{{"t": 0, "kind": "reaction|subject|reveal", "why": "..."}}], "hook_ideas": ["..."]}},
 "concepts": [{{"type": "reaction|subject|reveal", "bet": "the one-line reason to click",
  "hook": {{"lines": ["LINE ONE", "LINE TWO"], "accent_phrase": "...", "accent_color": "gold|sky|green|mint|blue|cyan",
            "annotation": null}},
  "face": {{"t": 0, "expression": "shock|deadpan|skeptical|hype|laughing"}},
  "subject": {{"source": "panel|art|made", "t": 0, "region": "right-panel", "query": null, "made": null}},
  "title": "...", "payoff_beats": [0]}}]}}"""


# --------------------------------------------------------------------------- context

def context(run_dir: Path) -> dict:
    """what the director knows: outline beats + the cut's loudest / most exclamatory moments."""
    from .. import punch
    from ..emphasis import annotate_emphasis
    from ..loud import spike_signal
    from ..srt import _run_dir
    from ..timeline_map import TimelineMap
    mp = run_dir / "ae" / "manifest.json"
    tmap = TimelineMap.from_manifest(mp) if not (mp.parent / "timeline.tsv").exists() else TimelineMap.load(mp)
    vod = _run_dir(mp)
    words = sorted(json.loads((vod / "transcript.json").read_text(encoding="utf-8")), key=lambda w: w["start"])
    annotate_emphasis(words, vod / "audio.wav")
    loud = punch.Signal(spike_signal(vod / "audio.wav", window=0.5))
    phr = punch.phrases(punch.timeline_words(words, tmap), tmap)
    lex = punch.load_lexicon()
    scored = []
    for p in phr:
        f = punch.features(p, loud, punch.Signal([]), lex, [])
        scored.append((f["z_loud_peak"] + 1.0 * f["lex"] + 0.5 * f["caps"], p))
    moments = [{"t": round(p["src_t0"], 2), "text": p["text"][:90]}
               for s, p in sorted(scored, key=lambda x: -x[0])[:16] if not p["straddle"]]
    out = json.loads((run_dir / "outline.json").read_text(encoding="utf-8"))
    beats = [{"id": i, "title": b.get("title"), "role": b.get("role"), "viewer_question": b.get("viewer_question"),
              "turn": b.get("turn"), "energy": b.get("energy"), "start": b.get("start"), "end": b.get("end"),
              "text": (b.get("text") or "")[:320]} for i, b in enumerate(out.get("beats", []))]
    gloss = sorted({t for g in vod.glob("glossary_*.json") for t in json.loads(g.read_text(encoding="utf-8"))})
    return {"central_idea": out.get("central_idea"), "viewer_promise": out.get("viewer_promise"),
            "beats": beats, "moments": moments, "glossary": gloss}


# --------------------------------------------------------------------------- director

def _models():
    from pydantic import BaseModel, field_validator

    class Hook(BaseModel):
        lines: list[str]
        accent_phrase: str | None = None
        accent_color: str = "gold"
        annotation: str | None = None

    class Face(BaseModel):
        t: float
        expression: str = "shock"

    class Subject(BaseModel):
        source: str = "video"
        t: float | None = None
        region: str = "right-panel"
        query: str | None = None
        made: dict | None = None

    class Concept(BaseModel):
        type: str
        bet: str = ""
        hook: Hook
        face: Face
        subject: Subject
        title: str = ""
        payoff_beats: list[int]

        @field_validator("payoff_beats")
        @classmethod
        def paid(cls, v):
            if not v:
                raise ValueError("a concept without payoff beats is rejected")
            return v

    class Out(BaseModel):
        brief: dict
        concepts: list[Concept]

    return Out


def validate(out, n_beats: int) -> list[str]:
    """house rules the schema can't express -> problems (empty = ok)."""
    errs = []
    types = [c.type for c in out.concepts]
    if sorted(types) != sorted(TYPES):
        errs.append(f"need exactly one of each type {TYPES}, got {types}")
    for c in out.concepts:
        words = " ".join(c.hook.lines).split()
        if len(words) > 5:
            errs.append(f"hook too long ({len(words)} words): {c.hook.lines}")
        long = [ln for ln in c.hook.lines if len(ln) > LINE_MAX]
        if long:
            errs.append(f"hook lines over {LINE_MAX} characters (aim for ~10): {long}")
        if any(b < 0 or b >= n_beats for b in c.payoff_beats):
            errs.append(f"payoff beat out of range: {c.payoff_beats}")
        if len(c.title) > 60:
            errs.append(f"title over 60 chars: {c.title}")
    return errs


def direct(ctx: dict, previous_hooks: list[str], variant: int, backend: str = "cli",
           model: str = "claude-opus-5", effort: str = "medium", ask=None):
    """one call (plus one corrective retry) -> validated Out."""
    from .. import director
    Out = _models()
    note = (f"\n\nThis is run {variant}. Bet differently from these earlier hooks: {previous_hooks}"
            if previous_hooks else "")
    user = json.dumps(ctx, ensure_ascii=False) + note

    def call(text):
        if ask:
            return ask(text, Out)
        content = [{"type": "text", "text": text}]
        if backend == "cli":
            err = None
            for _ in range(2):          # the cli now and then returns unparseable json: one more try first
                try:
                    return director._complete_cli(model, SYSTEM, content, Out, tag="thumb director", effort=effort,
                                                  live=False)
                except (RuntimeError, OSError) as e:
                    err = e
            director._api_fallback("thumb director", err)
        import anthropic
        return director._complete(anthropic.Anthropic(), model, SYSTEM, content, Out, tag="thumb director",
                                  effort=effort)

    out = call(user)
    errs = validate(out, len(ctx["beats"]))
    if errs:
        out = call(user + "\n\nYour last answer broke these rules, fix them: " + "; ".join(errs))
        errs = validate(out, len(ctx["beats"]))
        if errs:
            raise ValueError("thumb director: " + "; ".join(errs))
    return out


# --------------------------------------------------------------------------- frames

def best_face(cams, sid: int, t: float, expression: str, window: float = 1.5, fps: float = 6.0):
    """score facecam frames around t (spec q = .35 sharp + .35 expression + .15 size + .15 open);
    -> (crop BGR, info). expression proxy: mouth-open darkness from yunet's mouth corners.
    ponytail: MediaPipe blendshapes would score expression properly (needs the opencv override)."""
    import cv2
    cam = cams.detect(sid)
    if cam is None:
        raise ValueError("no facecam for this source")
    times = list(np.arange(t - window, t + window + 1e-6, 1 / fps))
    frames = cams.frames(sid, times, width=0)
    cand, prev = [], None
    for f in frames:
        h, w = f.shape[:2]
        x0, y0, x1, y1 = cam.box
        crop = f[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))]
        rows = cams.detector.raw(crop)
        if not len(rows):
            prev = crop
            continue
        r = max(rows, key=lambda r: r[2] * r[3])
        fx, fy, fw, fh = (int(v) for v in r[:4])
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        face = g[max(0, fy):fy + fh, max(0, fx):fx + fw]
        sharp = float(cv2.Laplacian(face, cv2.CV_64F).var()) if face.size else 0.0
        (mrx, mry), (mlx, mly) = r[10:12], r[12:14]
        my = int((mry + mly) / 2)
        mouth = g[my:my + max(2, int(0.18 * fh)), int(min(mrx, mlx)):int(max(mrx, mlx))]
        dark = 1.0 - float(mouth.mean()) / max(1.0, float(face.mean())) if mouth.size and face.size else 0.0
        motion = float(np.abs(cv2.absdiff(crop, prev)).mean()) if prev is not None and prev.shape == crop.shape else 0.0
        prev = crop
        cand.append({"crop": crop, "sharp": sharp, "size": fh, "conf": float(r[14]), "mouth": dark, "motion": motion})
    if not cand:
        raise ValueError(f"no face in the facecam around {t:.1f}s")
    cut = np.percentile([c["motion"] for c in cand], 80) if len(cand) > 4 else float("inf")
    keep = [c for c in cand if c["motion"] <= cut] or cand

    def norm(k):
        v = np.array([c[k] for c in keep], float)
        return (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.ones_like(v)
    S, A, O, M = norm("sharp"), norm("size"), norm("conf"), norm("mouth")
    E = 1.0 - M if expression in ("deadpan", "skeptical") else M
    q = 0.35 * S + 0.35 * E + 0.15 * A + 0.15 * O
    i = int(np.argmax(q))
    return keep[i]["crop"], {"q": round(float(q[i]), 3), "frames": len(cand)}


def art_crop(png: Path, out: Path) -> Path:
    """official splash -> the figure, not the whole effects swirl: alpha bbox, center 60%."""
    from PIL import Image
    im = Image.open(png).convert("RGBA")
    x0, y0, x1, y1 = im.getchannel("A").point(lambda a: 255 if a > 200 else 0).getbbox() or (0, 0, *im.size)
    cx, w = (x0 + x1) / 2, (x1 - x0) * 0.6
    im.crop((int(max(0, cx - w / 2)), y0, int(min(im.width, cx + w / 2)), y1)).save(out)
    return out


# --------------------------------------------------------------------------- compose

def _accent(name: str, bg_rgb: np.ndarray) -> str:
    """keep the director's accent unless it sits within 60 deg of the background hue."""
    import colorsys

    from .checks import dominant_hue
    from .text import PALETTE
    bh = dominant_hue(bg_rgb)

    def dist(n):
        base = PALETTE[n][0]
        h = colorsys.rgb_to_hsv(*(int(base[i:i + 2], 16) / 255 for i in (1, 3, 5)))[0] * 360
        d = abs(h - bh) % 360
        return min(d, 360 - d)
    name = name if name in PALETTE else "gold"
    return name if dist(name) >= 60 else max(PALETTE, key=dist)


CAP_FRAC = 0.152  # hook cap height / frame height: the refs' lines sit at 90-117 px caps (avg ~108)
FOV = 50          # a wider lens than 30 so a tilted card actually reads as 3D (nguy: fix the focal length)
SPEEDLINES = Path(__file__).with_name("assets") / "speedlines.png"


def _fit(p: Path, max_w: float, max_h: float) -> float:
    """card width in px so it fits a max_w x max_h box at its own aspect."""
    from PIL import Image
    w, h = Image.open(p).size
    return round(min(max_w, max_h * w / h), 1)


# the left element (nguy): one clear item, ~50% of the left half, centered at mid-height, its
# edges blended into the scene. screen px; the card is set back in depth and swung toward zy0x.
LEFT_CENTER = (300, 300)
# per element: a tall card sits a bit higher so its top rows clear the hook; a wide tier list
# higher still (his MORE TIER LISTS); a cutout figure lower (it can stand behind the hook)
CENTERS = {"stat_card": (300, 285), "tierlist": (310, 270), "panel": (300, 275), "art": (300, 340)}
# his strongbox card is drawn ~0.85 as tall as the game's proportions (header 235 px for a 513 px
# wide card vs 283:492 in game): the free-transform squash every editor does, so the rows clear the hook
SQUASH = {"stat_card": 0.86}
LEFT_AREA = 0.6 * 640 * 720         # nguy "approx 50%"; his strongbox card measures ~59%
LEFT_MAX = (620, 560)
LEFT_RIGHT = 680          # the element's right edge stays left of this (zy0x starts ~ here)
FADES = {"panel": (0.08, 0.08, 0.08, 0.14), "stat_card": (0.05, 0.02, 0.03, 0.18),
         "tierlist": (0.05, 0.03, 0.04, 0.16), "art": (0.05, 0.03, 0.05, 0.18)}


def screen_size(wh: tuple[int, int], area: float = LEFT_AREA, max_wh: tuple = LEFT_MAX) -> tuple[float, float]:
    """on-screen (w, h) for an image of this size: `area` px at its own aspect, capped."""
    a = wh[0] / wh[1]
    sw, sh = (area * a) ** 0.5, (area / a) ** 0.5
    k = min(1.0, max_wh[0] / sw, max_wh[1] / sh)
    return sw * k, sh * k


def project(card: dict, aspect: float, canvas=(1280, 720)) -> tuple[float, float, float, float]:
    """screen bbox (x0, y0, x1, y1) of a scene.js card: plane w x w/aspect centered at
    (x + w/2, -(y + h/2), z), rotated euler YXZ (three.js), pinhole camera on the frame center."""
    import math
    W, H = canvas
    dist = (H / 2) / math.tan(math.radians(FOV / 2))
    w = card["w"]
    h = w / aspect
    rx, ry, rz = (math.radians(v) for v in card.get("rot", [0, 0, 0]))
    cx, cy, cz = card["x"] + w / 2, -(card["y"] + h / 2), card.get("z", 0)
    pts = []
    for px, py in ((-w / 2, h / 2), (w / 2, h / 2), (w / 2, -h / 2), (-w / 2, -h / 2)):
        x, y = px * math.cos(rz) - py * math.sin(rz), px * math.sin(rz) + py * math.cos(rz)     # Rz
        y, z = y * math.cos(rx), y * math.sin(rx)                                                # Rx
        x, z = x * math.cos(ry) + z * math.sin(ry), -x * math.sin(ry) + z * math.cos(ry)         # Ry
        X, Y, Z = cx + x, cy + y, cz + z
        k = dist / (dist - Z)
        pts.append((W / 2 + (X - W / 2) * k, H / 2 - (Y + H / 2) * k))
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    return min(xs), min(ys), max(xs), max(ys)


def fit_card(src: str, aspect: float, box: tuple, z: float, rot: list, canvas=(1280, 720)) -> dict:
    """a card whose projected bbox fills `box` (x0, y0, x1, y1 screen px) at its aspect:
    scale to the box, then shift the center, a few rounds (perspective isn't linear)."""
    import math
    W, H = canvas
    dist = (H / 2) / math.tan(math.radians(FOV / 2))
    k = dist / (dist - z)
    bw, bh = box[2] - box[0], box[3] - box[1]
    card = {"src": src, "x": 0.0, "y": 0.0, "w": min(bw, bh * aspect) / k, "z": z, "rot": rot}
    tx, ty = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    card["x"] = W / 2 + (tx - W / 2) / k - card["w"] / 2
    card["y"] = H / 2 + (ty - H / 2) / k - card["w"] / aspect / 2
    for _ in range(6):
        x0, y0, x1, y1 = project(card, aspect, canvas)
        s_ = min(bw / (x1 - x0), bh / (y1 - y0))
        cxw, cyw = card["x"] + card["w"] / 2, card["y"] + card["w"] / aspect / 2
        card["w"] *= s_
        cxw += (tx - (x0 + x1) / 2) / k
        cyw += (ty - (y0 + y1) / 2) / k
        card["x"], card["y"] = cxw - card["w"] / 2, cyw - card["w"] / aspect / 2
    card.update(x=round(card["x"], 1), y=round(card["y"], 1), w=round(card["w"], 1))
    return card


def left_box(sw: float, sh: float, center=LEFT_CENTER, margin=(40, 30), canvas=(1280, 720),
             right: float = LEFT_RIGHT) -> tuple:
    """the element's screen box: sw x sh around `center`, pushed inside the frame margins."""
    cx = max(margin[0] + sw / 2, min(center[0], right - sw / 2))      # in frame, off zy0x
    cy = max(margin[1] + sh / 2, min(center[1], canvas[1] - margin[1] - sh / 2))
    return cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2


def cards_for(c, subject, kind: str, made_kind: str | None = None) -> tuple[list[dict], tuple]:
    """the subject card for a layout + its screen box (x0, y0, x1, y1). one item only (nguy:
    one clear theme). the png gets soft edges (FADES) written beside it as *_soft.png."""
    from PIL import Image

    from . import fx
    p = subject[0] if isinstance(subject, list) else subject
    if not p:
        return [], None
    fk = "art" if kind == "art" else (made_kind or "panel") if kind == "made" else "panel"
    soft = p.with_name(p.stem + "_soft.png")
    fx.fade(Image.open(p), FADES.get(fk, FADES["panel"])).save(soft)
    w, h = Image.open(p).size
    aspect = w / h / SQUASH.get(fk, 1.0)
    if kind == "art":       # a cutout figure: taller, nearly flat on, may run behind the hook
        sw, sh = screen_size((aspect, 1), area=640 * 720, max_wh=(560, 640))     # sized by height
        box = left_box(sw, sh, CENTERS["art"], margin=(30, 20))
        rot = [0, 8, 0]
    else:
        if fk == "tierlist":    # his tier lists run on behind zy0x (only ~4 icons per row show)
            sw, sh = screen_size((aspect, 1), area=0.75 * 640 * 720, max_wh=(700, 560))
            box = left_box(sw, sh, CENTERS[fk], right=LEFT_RIGHT + 60)
        else:
            sw, sh = screen_size((aspect, 1))
            box = left_box(sw, sh, CENTERS.get(fk, LEFT_CENTER))
        rot = [3, 14 if c.type == "subject" else 16, -2]
    card = fit_card(f"assets/{soft.name}", aspect, box, -70, rot)
    if fk in SQUASH:
        card["h"] = round(card["w"] / aspect, 1)
    return [card], box


def build_spec(c, d: Path, face_png: Path, subject_png, subject_kind: str, bg_rgb, seed: int,
               made_kind: str | None = None, hides: bool = False) -> dict:
    """subject_kind: art | made | panel. hides: a made graphic that already hides its own values."""
    from PIL import Image

    from . import compose
    im = Image.open(face_png)
    arr = np.asarray(im.convert("RGBA"))[:, :, [2, 1, 0, 3]]
    box = compose.face_box(arr)
    if box is None:
        raise ValueError("no face in the cutout")
    target = 0.80 if c.type == "subject" else compose.FIGURE_TARGET
    import shutil
    shutil.copy(SPEEDLINES, d / "assets" / "speedlines.png")
    f = compose.place_face(im.size, box, compose.head_top(arr), target=target)
    fx, fy, fw, fh = f["face"]
    cards, sbox = cards_for(c, subject_png, subject_kind, made_kind)
    spec = {"version": 1, "layout": c.type, "canvas": [1280, 720], "background": {"vignette": 0.55},
            "layers": [{"kind": "face", "src": f"assets/{face_png.name}", "x": f["x"], "y": f["y"], "w": f["w"],
                        "glow": "#ffffff", "face_box": f["face"]}],
            "scene": {"seed": seed, "fov": FOV, "bg": {"raw": "assets/bg.jpg", "blur": 12, "brightness": 0.6,
                                                       "saturate": 1.15, "z": -400},
                      # nguy's editing-pack speed lines, 50% opacity, converging on zy0x's face
                      "overlays": [{"src": "assets/speedlines.png", "cx": fx + fw / 2, "cy": fy + fh / 2,
                                    "w": 1800, "z": -150, "opacity": 0.5}],
                      "cards": cards,
                      "sparkles": {"count": 26, "x0": 0, "x1": 700, "y0": 0, "y1": 420, "size": 9},
                      # bright splash art blooms to white at the gameplay setting
                      "bloom": ({"strength": 0.25, "radius": 0.35, "threshold": 0.92} if subject_kind == "art"
                                else {"strength": 0.5, "radius": 0.4, "threshold": 0.82})},
            "text": {"lines": [ln.upper() for ln in c.hook.lines], "accent_phrase": (c.hook.accent_phrase or "").upper() or None,
                     "accent": _accent(c.hook.accent_color, bg_rgb), "annotation": c.hook.annotation,
                     "cap_frac": CAP_FRAC, "width_frac": 0.74, **({"skew": -10} if c.type == "reveal" else {})}}
    if c.type == "reveal" and cards and subject_kind != "art":
        # a made graphic hides its own values; a stream panel gets the mosaic band
        compose.reveal(spec, d, boxes=None if hides else [(0.06, 0.30, 0.94, 0.62)],
                       bubble_at=(round(min(sbox[2] - 70, 560)), round(max(12, sbox[1] - 40))))
    return spec


# --------------------------------------------------------------------------- run

def gameplay(region: tuple, cam_box) -> tuple:
    """keep a crop off the stream's own facecam + the chat stacked above it: a left cam trims
    the region's left edge to the cam's right edge (a right cam, the other way)."""
    x0, y0, x1, y1 = region
    if cam_box:
        if cam_box[0] < 0.05:
            x0 = max(x0, min(cam_box[2] + 0.01, x1 - 0.2))
        elif cam_box[2] > 0.95:
            x1 = min(x1, max(cam_box[0] - 0.01, x0 + 0.2))
    return x0, y0, x1, y1


def snap(cams, sid: int, t: float) -> float:
    """a time the cut actually shows (beats can include stretches the edit trimmed away)."""
    spans = [(a, b) for s_, _, a, b, _ in cams._ranges if s_ == sid]
    if any(a <= t < b for a, b in spans) or not spans:
        return t
    a, b = min(spans, key=lambda r: min(abs(t - r[0]), abs(t - r[1])))
    return min(max(t, a + 0.05), b - 0.05)


def _ocr():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def _clusters(boxes: list[list[float]], gap: float) -> list[list[list[float]]]:
    """group text boxes (x0, y0, x1, y1) whose gap-padded rects overlap (union-find)."""
    parent = list(range(len(boxes)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            if a[0] - gap < b[2] and b[0] - gap < a[2] and a[1] - gap < b[3] and b[1] - gap < a[3]:
                parent[root(i)] = root(j)
    groups: dict[int, list] = {}
    for i, bx in enumerate(boxes):
        groups.setdefault(root(i), []).append(bx)
    return list(groups.values())


def find_panel(cams, sid: int, t: float, cam_box, hint: str | None = None, window: float = 2.0, n: int = 5,
               ocr=None):
    """the game UI panel on screen near t (artifact card, tier list, character grid), or None.
    text boxes (rapidocr detection) right of the stream's cam/chat column, grouped into blocks;
    the densest block is the panel; the sharpest frame wins. never a blind region crop."""
    import cv2
    eng = ocr or _ocr()
    gx0, _, gx1, _ = gameplay((0, 0, 1, 1), cam_box)
    hx = None
    if hint in REGIONS:
        r = REGIONS[hint]
        hx = (r[0] + r[2]) / 2
    best = None
    for tt in np.linspace(t - window, t + window, n):
        fr = cams.frames(sid, [snap(cams, sid, float(tt))], width=0)
        if not fr:
            continue
        f = fr[0]
        H, W = f.shape[:2]
        sw, sh = 960, 540
        res = eng(cv2.resize(f, (sw, sh)), use_det=True, use_cls=False, use_rec=False)[0] or []
        boxes = []
        for q in res:
            xs, ys = [pt[0] / sw for pt in q], [pt[1] / sh for pt in q]
            bx = [min(xs), min(ys), max(xs), max(ys)]
            if gx0 <= (bx[0] + bx[2]) / 2 <= gx1:
                boxes.append(bx)
        if len(boxes) < 5:
            continue
        for g in _clusters(boxes, 0.035):
            if len(g) < 5:
                continue
            x0, y0 = min(b[0] for b in g), min(b[1] for b in g)
            x1, y1 = max(b[2] for b in g), max(b[3] for b in g)
            m = 0.03
            x0, y0, x1, y1 = max(gx0, x0 - m), max(0, y0 - m), min(gx1, x1 + m), min(1, y1 + m)
            w, h = x1 - x0, y1 - y0
            if w * h < 0.06 or not 0.3 <= (w * W) / (h * H) <= 3.2:
                continue
            crop = f[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
            sharp = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            near = 1.0 if hx is None else 1.0 - min(1.0, abs((x0 + x1) / 2 - hx))
            score = len(g) * np.sqrt(sharp) * (0.5 + 0.5 * near)
            if best is None or score > best[0]:
                best = (score, crop, len(g))
    return None if best is None else best[1]


def art_card(query: str, d: Path, log=print) -> Path | None:
    """one official image for the left element -> the figure cut out of it, in assets/.
    a character -> its wish splash; anything else (weapon, item, enemy, boss) -> its wiki image."""
    import shutil

    from . import art
    q = query.split(",")[0].strip()
    if not q:
        return None
    try:
        roster = art.characters(art.Fetcher())
        if art.resolve(q, roster):
            src = art.splash(q)
        else:
            try:
                src = art.item(q)
            except art.ArtError:
                from . import made                      # an artifact set: its official flower icon
                piece = made.artifact({"set": q, "subtitle": "flower"})
                if not piece:
                    raise
                src = piece["icon"]
                src.with_suffix(".json").write_text(json.dumps({"name": piece["set"], "source": "yatta",
                                                                "icon": src.name}), encoding="utf-8")
        fig = art.figure(src) if src.name.endswith(("_wish.png", "_item.png")) else src
    except art.ArtError as e:
        log(f"[thumb] art for {q!r} unavailable ({e})")
        return None
    out = d / "assets" / "art0.png"
    shutil.copy(fig, out)
    (d / "assets" / "art0.art.json").write_text(src.with_suffix(".json").read_text(encoding="utf-8"), encoding="utf-8")
    return out


def characters_in(names: list[str]) -> list[str]:
    """which of these brief entities are genshin characters (resolved against the roster)."""
    from . import art
    try:
        roster = art.characters(art.Fetcher())
    except art.ArtError:
        return []
    return [r for r in (art.resolve(n, roster) for n in names) if r]


def assemble(c, k: str, root: Path, cams, browser, variant: int, entities: list[str], ocr=None, log=print) -> None:
    """one concept -> <root>/<k>/{assets, spec.json, thumb.png}."""
    import cv2

    from . import render
    from .cutout import cutout
    sid = 0
    d = root / k
    (d / "assets").mkdir(parents=True, exist_ok=True)
    c.face.t = snap(cams, sid, c.face.t)
    if c.subject.t is not None:
        c.subject.t = snap(cams, sid, c.subject.t)
    crop, info = best_face(cams, sid, c.face.t, c.face.expression)
    cv2.imwrite(str(d / "assets" / "face.png"), cutout(crop))
    cam_box = cams.detect(sid).box
    sub, kind = None, c.subject.source
    if kind == "made" and c.subject.made:
        from . import made
        try:
            sub = made.render(c.subject.made.get("kind", "tierlist"), c.subject.made, d / "assets" / "made.png",
                              browser=browser)
            kind = "made"
        except Exception as e:                           # a bad made spec falls back to art
            log(f"[thumb] {k}: made graphic failed ({e}); trying character art")
            kind = "art"
    if kind in ("video", "panel") and c.subject.t is not None:
        panel = find_panel(cams, sid, c.subject.t, cam_box, c.subject.region, ocr=ocr)
        if panel is not None:
            sub, kind = d / "assets" / "subject.png", "panel"
            cv2.imwrite(str(sub), panel)
        else:
            log(f"[thumb] {k}: no UI panel near {c.subject.t:.0f}s; trying character art")
            kind = "art"
    if sub is None:                                      # nguy: there is always a left element
        kind = "art"
    if kind == "art":
        sub = art_card(c.subject.query or next(iter(characters_in(entities)), ""), d, log)
        if sub is None and c.subject.query:
            sub = art_card(next(iter(characters_in(entities)), ""), d, log)
    t_bg = c.subject.t if c.subject.t is not None and kind == "panel" else c.face.t
    made_hides = kind == "made" and bool((c.subject.made or {}).get("hidden"))
    bg = cams.frames(sid, [t_bg], width=0)[0]
    gx0, _, gx1, _ = gameplay((0, 0, 1, 1), cam_box)
    bg = bg[:, int(gx0 * bg.shape[1]):int(gx1 * bg.shape[1])]
    cv2.imwrite(str(d / "assets" / "bg.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 90])
    bg_rgb = cv2.cvtColor(cv2.GaussianBlur(bg, (0, 0), 12), cv2.COLOR_BGR2RGB)
    spec = build_spec(c, d, d / "assets" / "face.png", sub, kind if kind in ("art", "made") else "panel",
                      bg_rgb, seed=variant * 10 + "ABC".index(k),
                      made_kind=(c.subject.made or {}).get("kind") if kind == "made" else None, hides=made_hides)
    (d / "spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    render.render(d / "spec.json", browser=browser)
    log(f"[thumb] {k} {c.type}: {' / '.join(c.hook.lines)}  (face q {info['q']}, subject {kind if sub else 'none'})")


def _render_all(run_dir: Path, root: Path, concepts: list, variant: int, entities: list[str], log=print) -> Path:
    from ..facecam import Facecams
    from . import render, report
    cams = Facecams(run_dir / "ae" / "manifest.json")
    ocr = _ocr()
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = render.launch(p)
        try:
            for k, c in zip("ABC", concepts):
                assemble(c, k, root, cams, b, variant, entities, ocr, log)
        finally:
            b.close()
    return report.build(run_dir, root=root)


def run(run_dir, variant: int = 1, previous_hooks: list[str] | None = None, backend: str = "cli",
        model: str = "claude-opus-5", ask=None, log=print) -> Path:
    """one full candidate run -> <run_dir>/thumb/v<variant>/report.md."""
    run_dir = Path(run_dir)
    root = run_dir / "thumb" / f"v{variant}"
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    ctx = context(run_dir)
    log(f"[thumb] context: {len(ctx['beats'])} beats, {len(ctx['moments'])} moments ({time.monotonic() - t0:.0f}s)")
    t1 = time.monotonic()
    out = direct(ctx, previous_hooks or [], variant, backend, model, ask=ask)
    log(f"[thumb] director: {[c.hook.lines for c in out.concepts]} ({time.monotonic() - t1:.0f}s)")
    (root / "brief.json").write_text(json.dumps(out.brief, indent=1, ensure_ascii=False), encoding="utf-8")
    order = sorted(out.concepts, key=lambda c: TYPES.index(c.type))
    concepts = [{"id": k, **c.model_dump()} for k, c in zip("ABC", order)]
    (root / "concepts.json").write_text(json.dumps({"concepts": concepts}, indent=1, ensure_ascii=False), encoding="utf-8")
    rep = _render_all(run_dir, root, order, variant, out.brief.get("entities", []), log)
    log(f"[thumb] v{variant}: {rep}  ({time.monotonic() - t0:.0f}s)")
    return rep


def recompose(run_dir, variant: int, log=print) -> Path:
    """rebuild a run's candidates from its saved brief + concepts (no director call): for
    layout/asset changes."""
    run_dir = Path(run_dir)
    root = run_dir / "thumb" / f"v{variant}"
    Concept = _models().model_fields["concepts"].annotation.__args__[0]
    concepts = [Concept.model_validate({k: v for k, v in c.items() if k != "id"})
                for c in json.loads((root / "concepts.json").read_text(encoding="utf-8"))["concepts"]]
    brief = json.loads((root / "brief.json").read_text(encoding="utf-8"))
    return _render_all(run_dir, root, concepts, variant, brief.get("entities", []), log)
