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
REGIONS = {"full": (0, 0, 1, 1), "left-half": (0, 0, 0.5, 1), "right-half": (0.5, 0, 1, 1),
           "center": (0.2, 0.08, 0.8, 0.92), "right-panel": (0.64, 0.06, 0.99, 0.94),
           "left-panel": (0.01, 0.06, 0.36, 0.94), "top-half": (0, 0, 1, 0.5), "bottom-half": (0, 0.5, 1, 1)}
PALETTE_THEMES = "gold: artifacts/rewards; sky: Cryo/Snezhnaya; green or mint: easy/right/wins; cyan: people/reviews; blue: series/lists"

SYSTEM = f"""You direct thumbnails for zy0x's Genshin Impact stream edits (YouTube channel). You pick
three thumbnail concepts for YouTube Test & Compare, which picks a winner by WATCH TIME, so the
three must be three different reasons to click, and each must be paid off by the cut (cite the
outline beats that deliver it; a concept without payoff beats is rejected).

House style (his thumbnails run 8-11% CTR, keep to it): a two-line ALL-CAPS hook, 2-4 words
ideal, 5 max, no swearing; questions/teasers end in "?" or "..."; one accent phrase in one accent
color ({PALETTE_THEMES}); a small lowercase annotation is optional, e.g. "(birthday)".
zy0x's face is huge on the right (shock, deadpan, skeptical, hype, laughing). The left side holds
the subject: a game UI moment from the stream (artifact card, tier list, wish screen) or a
character's official splash art.

Concept types, use each exactly once:
- reaction: the strongest asset is his face at a moment; subject small.
- subject: the video is about a thing (an artifact, a character, a tier list); subject big.
- reveal: hide the answer (a stat, a tier placement, a pull result) behind a pixel mosaic with a
  "???" bubble; the hook is a question.

Pick face.t ONLY from the given moments (source seconds) and say the expression. For a video
subject give t (source seconds inside a beat) and a region: {", ".join(REGIONS)}. For art give
query = a character name. Title <= 60 characters, paired with the hook.

Reply with ONLY JSON:
{{"brief": {{"promise": "...", "entities": ["..."], "reveal": {{"what": "...", "t": 0, "beat_id": 0}} or null,
  "moments": [{{"t": 0, "kind": "reaction|subject|reveal", "why": "..."}}], "hook_ideas": ["..."]}},
 "concepts": [{{"type": "reaction|subject|reveal", "bet": "the one-line reason to click",
  "hook": {{"lines": ["LINE ONE", "LINE TWO"], "accent_phrase": "...", "accent_color": "gold|sky|green|mint|blue|cyan",
            "annotation": null}},
  "face": {{"t": 0, "expression": "shock|deadpan|skeptical|hype|laughing"}},
  "subject": {{"source": "video|art|none", "t": 0, "region": "right-panel", "query": null}},
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
            try:
                return director._complete_cli(model, SYSTEM, content, Out, tag="thumb director", effort=effort,
                                              live=False)
            except (RuntimeError, OSError) as e:
                director._api_fallback("thumb director", e)
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


def build_spec(c, d: Path, face_png: Path, subject_png: Path | None, subject_kind: str, bg_rgb, seed: int) -> dict:
    from PIL import Image

    from . import compose
    im = Image.open(face_png)
    arr = np.asarray(im.convert("RGBA"))[:, :, [2, 1, 0, 3]]
    box = compose.face_box(arr)
    if box is None:
        raise ValueError("no face in the cutout")
    target = 0.80 if c.type == "subject" else compose.FIGURE_TARGET
    f = compose.place_face(im.size, box, compose.head_top(arr), target=target)
    fx, fy, fw, fh = f["face"]
    cards = []
    if subject_png:
        if subject_kind == "art":
            cards.append({"src": f"assets/{subject_png.name}", "x": -60, "y": -20, "w": 640, "z": -80, "rot": [0, 10, 0]})
        elif c.type == "subject":
            cards.append({"src": f"assets/{subject_png.name}", "x": 30, "y": 20, "w": 600, "z": 10, "rot": [3, 12, -3],
                          "shadow": 0.75, "glow": "#ffffff", "glow_alpha": 0.35})
        else:
            cards.append({"src": f"assets/{subject_png.name}", "x": 50, "y": 30, "w": 430, "z": 10, "rot": [4, 18, -4],
                          "shadow": 0.75, "glow": "#ffffff", "glow_alpha": 0.35})
    spec = {"version": 1, "layout": c.type, "canvas": [1280, 720], "background": {"vignette": 0.55},
            "layers": [{"kind": "face", "src": f"assets/{face_png.name}", "x": f["x"], "y": f["y"], "w": f["w"],
                        "glow": "#ffffff", "face_box": f["face"]}],
            "scene": {"seed": seed, "fov": 30, "bg": {"raw": "assets/bg.jpg", "blur": 12, "brightness": 0.6,
                                                      "saturate": 1.15, "z": -400},
                      "rays": {"x": fx + fw / 2, "y": fy + fh * 0.4, "color": "#fff3d6", "strength": 0.22,
                               "count": 22, "z": -120},
                      "cards": cards,
                      "sparkles": {"count": 26, "x0": 0, "x1": 700, "y0": 0, "y1": 420, "size": 9},
                      "bloom": {"strength": 0.5, "radius": 0.4, "threshold": 0.82}},
            "text": {"lines": [ln.upper() for ln in c.hook.lines], "accent_phrase": (c.hook.accent_phrase or "").upper() or None,
                     "accent": _accent(c.hook.accent_color, bg_rgb), "annotation": c.hook.annotation,
                     "width_frac": 0.6, **({"skew": -10} if c.type == "reveal" else {})}}
    if c.type == "reveal" and cards and subject_kind == "video":
        compose.reveal(spec, d, boxes=[(0.06, 0.30, 0.94, 0.62)], bubble_at=(cards[0]["x"] + 280, 18))
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


def run(run_dir, variant: int = 1, previous_hooks: list[str] | None = None, backend: str = "cli",
        model: str = "claude-opus-5", ask=None, log=print) -> Path:
    """one full candidate run -> <run_dir>/thumb/v<variant>/report.md."""
    import cv2

    from ..facecam import Facecams
    from . import art, render, report
    from .cutout import cutout
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

    mp = run_dir / "ae" / "manifest.json"
    cams = Facecams(mp)
    sid = 0
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = render.launch(p)
        try:
            for k, c in zip("ABC", order):
                d = root / k
                (d / "assets").mkdir(parents=True, exist_ok=True)
                c.face.t = snap(cams, sid, c.face.t)
                if c.subject.t is not None:
                    c.subject.t = snap(cams, sid, c.subject.t)
                crop, info = best_face(cams, sid, c.face.t, c.face.expression)
                cv2.imwrite(str(d / "assets" / "face.png"), cutout(crop))
                cam_box = cams.detect(sid).box
                bg = cams.frames(sid, [c.subject.t if c.subject.source == "video" and c.subject.t else c.face.t], width=0)
                hh, ww = bg[0].shape[:2]
                gx0, _, gx1, _ = gameplay((0, 0, 1, 1), cam_box)
                bg = [bg[0][:, int(gx0 * ww):int(gx1 * ww)]]
                cv2.imwrite(str(d / "assets" / "bg.jpg"), bg[0], [cv2.IMWRITE_JPEG_QUALITY, 90])
                sub, kind = None, c.subject.source
                if kind == "video" and c.subject.t is not None:
                    fr = cams.frames(sid, [c.subject.t], width=0)
                    if fr:
                        h, w = fr[0].shape[:2]
                        x0, y0, x1, y1 = gameplay(REGIONS.get(c.subject.region, REGIONS["right-panel"]), cam_box)
                        sub = d / "assets" / "subject.png"
                        cv2.imwrite(str(sub), fr[0][int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)])
                elif kind == "art" and c.subject.query:
                    try:
                        sp = art.splash(c.subject.query)
                        sub = art_crop(sp, d / "assets" / "art.png")
                        (d / "assets" / "art.art.json").write_text(sp.with_suffix(".json").read_text(encoding="utf-8"),
                                                                   encoding="utf-8")
                    except art.ArtError as e:
                        log(f"[thumb] {k}: art for {c.subject.query!r} unavailable ({e}); no subject")
                bg_rgb = cv2.cvtColor(cv2.GaussianBlur(bg[0], (0, 0), 12), cv2.COLOR_BGR2RGB)
                spec = build_spec(c, d, d / "assets" / "face.png", sub, kind, bg_rgb, seed=variant * 10 + "ABC".index(k))
                (d / "spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
                render.render(d / "spec.json", browser=b)
                log(f"[thumb] {k} {c.type}: {' / '.join(c.hook.lines)}  (face q {info['q']} of {info['frames']} frames)")
        finally:
            b.close()
    rep = report.build(run_dir, root=root)
    log(f"[thumb] v{variant}: {rep}  ({time.monotonic() - t0:.0f}s)")
    return rep
