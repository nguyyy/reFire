"""`refire thumb <run_dir>`: three thumbnail candidates from a run folder alone (03-thumb.md).

context (outline beats + the cut's loudest/exclamatory moments) -> ONE Claude call writing
the brief and three concepts (spec: two calls; merged to halve subscription use, both files
still written) -> per concept: best facecam frame near its moment, zy0x cutout, subject
(official art or a designed card; nguy: never a stream screen), a real in-game backdrop (the
place's wiki screenshot), layout in the 3D scene, accent auto-fixed against the background hue -> render -> deterministic checks -> report.md. Each run lands in
<run_dir>/thumb/v<k>/ so repeated runs sit side by side; later runs are told the earlier
hooks so they bet differently.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

TYPES = ("reaction", "subject", "reveal")
QUOTE_MAX = 26        # characters in a quote bubble (his: "I CAN'T BEAT THIS SH*T", 22)
LINE_MAX = 13         # hook characters per line, spaces included (refs: 6-13, ~10; nguy: "around 10")
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

Think before you design. First name what the video is about (brief.about): the one thing a fan
recognizes in a second, usually what the stream title and the central idea are about (a new
character, a boss or mode like Stygian Onslaught, the artifact that dropped, a tier list, a
region at 100%). EVERY concept's left element shows that thing or its sharpest evidence (the
character he built or pulled, the boss he fought, the artifact that dropped), never a character
only mentioned in passing. The three concepts may show it differently (official art / the team /
a designed in-game card) but all sell this video. What makes a fan click: novelty
(a new character, boss, area), value (5-star, max roll, #1, C6, 100%), stakes or mystery (hidden
behind the mosaic). Show what is chased or named (the stream title's "C6 VALERIY" -> Valeriy),
not the banner or menu around it. The three left elements must look different from each other
(never two of the same screen or map). The hook and the left element work as a pair: the hook asks about or names
what the left shows, never repeats it word for word. Write each concept's plan before its fields.

Concept types, use each exactly once:
- reaction: the strongest asset is his face at a moment.
- subject: the video is about a thing (an artifact, a character, a tier list); subject big.
- reveal: hide the answer (a stat, a tier placement, a pull result) behind a pixel mosaic with a
  "???" bubble; the hook is a question.

When frame sheets are given, Read every one before you plan: they show the gameplay at each beat,
labelled "b<beat> <source seconds>". Plan the left element from what was actually on screen: an art
query names a thing you saw (the character he played, the boss he fought, the item that dropped),
a designed card carries the exact values you read off a frame. For a team video (Spiral Abyss,
Stygian Onslaught, Imaginarium Theater) the team he cleared with is the strongest left element:
an art query of up to 3 names, comma separated, becomes one group (his EASIEST ABYSS? shows the
team's art); name first the one the hook is about: it stands in the middle, in front. It must read at phone size: bright, colorful official art or a designed card, NEVER a stream
screen (dark and busy at phone size). A reveal hides its answer inside a designed card (a result
card's "main" or rows, a stat_card's rows, a tier list's cells); nothing left visible may answer
the hook. An artifact you saw is ALWAYS a designed stat_card with the exact name, set, main stat and
substats you read off the frame; a clear time, a chamber result, an exploration %, a ranking is a
designed result card with the values you read off the frame. Names are spelled as in the game (the
given character list), never as the transcript heard them.
The background behind everything is a real in-game place, never the stream frame: brief.backdrop =
the exact wiki page name of the place the video happens in ("Spiral Abyss", "Land of Lower Setekh",
a region, an area, a domain), then after a comma the region it belongs to as a fallback (a mode or
boss without a place of its own, like Stygian Onslaught, has no screenshot: name where it is fought,
then its region, e.g. "Spiral Abyss, Mondstadt"); null only when nothing fits.

Pick face.t ONLY from the given moments (source seconds) and say the expression. The subject
must be deliberate, one item: either official art of ONE thing (source "art": query = one exact
in-game name: a character, weapon, material, enemy or boss, e.g. "Skirk", "Primogem", "Lord of the
Hidden Depths: Whisperer of Nightmares"), or a graphic you design (source "made": made =
{{"kind": "tierlist", "rows": [{{"label": "GOD TIER", "color": "#FF7F7F", "chars": ["Furina", ...]}}, ...],
"hidden": [[row, col]]}} with 3 rows x up to 4 Genshin characters (the ones the video is about), labels like the stream's own
(GOD TIER / OKAY / RUN AWAY or T0 / T0.5 / T67); or {{"kind": "stat_card", "title": exact artifact
piece name, "set": its artifact set, "subtitle": slot (Flower of Life / Plume of Death / Sands of Eon /
Goblet of Eonothem / Circlet of Logos), "main": [stat, value], "stars": 5, "level": "+20",
"rows": [[stat, value], ...] (4 substats), "hidden": [row indexes, the first rows: the lower ones sit
behind the hook]}}: rendered one-to-one as the in-game artifact card with its icon); or {{"kind": "result",
"title": the boss, chamber or area ("Floor 12 - Chamber 3", "Wavebinder: Stormfang", "Land of Lower
Setekh"), "subtitle": the mode ("Spiral Abyss", "Stygian Onslaught", "Exploration Progress"), "main":
[label, value] (the headline number: ["Time Elapsed", "117s"]), "stars": 0-3 (abyss stars, else 0),
"icon": one exact in-game name pictured in the header (the character he played or the boss), "rows":
up to 4 [label, value] facts you read off the screen, "hidden": ["main" and/or row indexes], "theme":
the header color: gold (rewards), purple (Stygian Onslaught), blue (Spiral Abyss), green (exploration),
red (danger)}}: the in-game card look. Never chat, never a stream screen, never a random shot.

Be creative with the left side: his newest thumbnails mix pieces, not always a card. Use them where
they sell the video (each optional):
- quote: a short line he actually SAID (from the moments or beat text), ALL CAPS, <= {QUOTE_MAX}
  characters, swears censored with * ("I CAN'T BEAT THIS SH*T"): a pixel speech bubble at the top,
  pointing at him. His own words when they are the funniest or most honest line; not on every concept.
- face.emote: "anger" puts the red anger mark over his head (tilted, betrayed, raging moments), else null.
- a reveal whose left element is character art gets big pixel ??? marks beside the character (his
  NUTTY SKIRK PULLS): use it when the question is who or what (the pull, the carry).
The three left elements must look different: never two cards of the same kind.
Title <= 60 characters, paired with the hook.

Reply with ONLY JSON:
{{"brief": {{"about": "...", "about_art": "one exact in-game name whose art shows it", "backdrop": "a place or null",
  "promise": "...", "entities": ["..."], "reveal": {{"what": "...", "t": 0, "beat_id": 0}} or null,
  "moments": [{{"t": 0, "kind": "reaction|subject|reveal", "why": "..."}}], "hook_ideas": ["..."]}},
 "concepts": [{{"type": "reaction|subject|reveal",
  "plan": {{"left": "what the left element shows and why a fan clicks it", "hook": "how hook and left pair up",
            "face": "why this face moment"}},
  "bet": "the one-line reason to click",
  "hook": {{"lines": ["LINE ONE", "LINE TWO"], "accent_phrase": "...", "accent_color": "gold|sky|green|mint|blue|cyan",
            "annotation": null}},
  "face": {{"t": 0, "expression": "shock|deadpan|skeptical|hype|laughing", "emote": null}},
  "quote": null,
  "subject": {{"source": "art|made", "query": null,
               "alt_query": "a second exact name showing the same thing, used if the first has no art",
               "made": null}},
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
    from ..auto import vod_info
    src = json.loads(mp.read_text(encoding="utf-8")).get("source") or ""
    title = vod_info(Path(src).with_suffix(".chat.json")).get("title", "") if src else ""
    try:
        from . import art
        names = sorted(art.characters(art.Fetcher()))          # canonical spellings (transcript: "Zabai")
    except Exception:                                          # offline: the glossary still helps
        names = []
    return {"stream_title": title, "characters": names,
            "central_idea": out.get("central_idea"), "viewer_promise": out.get("viewer_promise"),
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
        emote: str | None = None

    class Subject(BaseModel):
        source: str = "art"
        query: str | None = None
        alt_query: str | None = None
        made: dict | None = None

    class Concept(BaseModel):
        type: str
        plan: dict = {}
        bet: str = ""
        hook: Hook
        face: Face
        subject: Subject
        quote: str | None = None
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
        if c.hook.annotation and set(c.hook.annotation.upper().strip("()").split()) & set(words):
            errs.append(f"annotation repeats the hook: {c.hook.annotation} / {c.hook.lines}")
        long = [ln for ln in c.hook.lines if len(ln) > LINE_MAX]
        if long:
            errs.append(f"hook lines over {LINE_MAX} characters (aim for ~10): {long}")
        if any(b < 0 or b >= n_beats for b in c.payoff_beats):
            errs.append(f"payoff beat out of range: {c.payoff_beats}")
        if len(c.title) > 60:
            errs.append(f"title over 60 chars: {c.title}")
        if c.subject.source not in ("art", "made"):
            errs.append(f"subject source {c.subject.source!r}: only art or made (never a stream screen)")
        if c.quote and len(c.quote) > QUOTE_MAX:
            errs.append(f"quote over {QUOTE_MAX} characters: {c.quote}")
    made = [(c.subject.made or {}).get("kind", "tierlist") for c in out.concepts if c.subject.source == "made"]
    if len(made) != len(set(made)):
        errs.append(f"two left elements are the same kind of card ({made}): make the three look different")
    return errs


def direct(ctx: dict, previous_hooks: list[str], variant: int, backend: str = "cli",
           model: str = "claude-opus-5", effort: str = "medium", ask=None, sheets=None, cwd=None):
    """one call (plus one corrective retry) -> validated Out. sheets: frame sheet file names in
    `cwd` the model Reads first (cli only; ponytail: the api fallback goes without them)."""
    from .. import director
    Out = _models()
    note = (f"\n\nThis is run {variant}. Bet differently from these earlier hooks: {previous_hooks}"
            if previous_hooks else "")
    look = f"\n\nFrame sheets (Read them first): {', '.join(sheets)}" if sheets else ""
    user = json.dumps(ctx, ensure_ascii=False) + note + look

    def call(text):
        if ask:
            return ask(text, Out)
        content = [{"type": "text", "text": text}]
        if backend == "cli":
            err = None
            for _ in range(2):          # the cli now and then returns unparseable json: one more try first
                try:
                    return director._complete_cli(model, SYSTEM, content, Out, tag="thumb director", effort=effort,
                                                  live=False, tools="Read" if sheets else "",
                                                  cwd=cwd if sheets else None)
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

EDGE = 0.04          # a face box this close to the cam's side or top edge is cut off by the cam frame


def best_face(cams, sid: int, t: float, expression: str, window: float = 1.5, fps: float = 6.0):
    """score facecam frames around t (spec q = .35 sharp + .35 expression + .15 size + .15 open);
    -> (crop BGR, info). expression + open = mediapipe blendshapes (expression.py: the spec's recipes;
    open = eyes open, on the camera, head not tilted far); without mediapipe the old proxy
    (mouth-open darkness, yunet confidence)."""
    from . import expression as X
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
        ch, cw = crop.shape[:2]
        # leaning out of the cam: the cutout gets a flat sliced edge (and drags the wall along)
        clipped = fx < EDGE * cw or fx + fw > (1 - EDGE) * cw or fy < EDGE * ch
        cand.append({"crop": crop, "sharp": sharp, "size": fh, "conf": float(r[14]), "mouth": dark, "motion": motion,
                     "clipped": clipped})
    if not cand:
        raise ValueError(f"no face in the facecam around {t:.1f}s")
    cand = [c for c in cand if not c["clipped"]] or cand
    cut = np.percentile([c["motion"] for c in cand], 80) if len(cand) > 4 else float("inf")
    keep = [c for c in cand if c["motion"] <= cut] or cand

    def norm(k):
        v = np.array([c[k] for c in keep], float)
        return (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.ones_like(v)
    reads = [X.read(c["crop"]) for c in keep] if X.landmarker() is not None else []
    if reads and all(r is not None for r in reads):
        for c, r in zip(keep, reads):
            c["exp"], c["facing"] = X.score(expression, r["bs"]), X.facing(r)
        S, M, Op = norm("sharp"), norm("exp"), np.array([c["facing"] for c in keep])
    else:                                                # no mediapipe: the darkness proxy
        S, Op, M = norm("sharp"), norm("conf"), norm("mouth")
        if expression in ("deadpan", "skeptical"):
            M = 1.0 - M
    # his usual size in the window, not the biggest: leaning into the cam blows the face up and
    # breaks the 80-90 framing (the cap then leaves him short of the frame top)
    sz = np.array([c["size"] for c in keep], float)
    A = 1.0 - np.minimum(1.0, np.abs(sz - np.median(sz)) / max(1.0, np.median(sz)))
    q = 0.35 * S + 0.35 * M + 0.15 * A + 0.15 * Op
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


# hook size fitted on the refs (4-corner homography per ref, measured on the letters): letters
# average ~99 px (abyss 106, more tier lists 103, strongbox 94, skirk 80), lines ~730 px wide
# (widest 870). cap_frac sets the near end; the 3D plane (text.PLANE) shrinks the far end
CAP_FRAC, WIDTH_FRAC = 0.142, 0.68
# a slight black gradient up from the bottom edge, over zy0x and the scene, under the hook (his more
# tier lists: the shirt ~6% darker at 88-94% height, ~18% in the last strip; nguy: "slight")
SHADE = {"height": 0.2, "alpha": 0.3}
FOV = 50          # a wider lens than 30 so a tilted card actually reads as 3D (nguy: fix the focal length)
SPEEDLINES = Path(__file__).with_name("assets") / "speedlines.png"


def _fit(p: Path, max_w: float, max_h: float) -> float:
    """card width in px so it fits a max_w x max_h box at its own aspect."""
    from PIL import Image
    w, h = Image.open(p).size
    return round(min(max_w, max_h * w / h), 1)


# the left element (nguy): one clear item, ~50% of the left half, centered at mid-height, its
# edges blended into the scene. screen px; the card is set back in depth and swung toward zy0x.
FIG_TALL, FIG_KEEP = 1.2, 0.6   # a figure taller than 1.2x its width keeps its top 60%
FIG_THIN = 0.25                 # rows under 25% of the widest row's coverage above the body = effects
TEAM_STEP = 0.5                 # each team figure starts this far into the previous one's width
# nguy: the left element fills the left AND the middle so little background shows; the hook may
# cover part of it. his run from the left edge into the middle and behind zy0x (more tier lists to
# x ~770, the best tier list past 1000), from the top down behind the hook
ART_COVER = 0.32                # a figure's VISIBLE opaque pixels / frame (his left elements 24-42%)
ART_AT = (-30, 10)              # a figure hangs from the top-left, a little off the left edge (his bleed off
ART_MAX_W = 1100                # the edges) and may run on behind zy0x and off the bottom
ART_MAX_H = 920                 # the figure's upper half (its face) stays above the hook (~y 470)
ART_FACE = (320, 180)           # nguy: a character's face centered in the top-left quarter of the frame
LEFT_CENTER = (390, 310)
# per element: a tall card a bit higher so its top rows clear the hook; a cutout figure lower (it
# stands behind the hook and may run off the bottom)
CENTERS = {"stat_card": (390, 305), "result": (390, 305), "tierlist": (420, 300), "art": (380, 380)}
# his strongbox card is drawn ~0.85 as tall as the game's proportions (header 235 px for a 513 px
# wide card vs 283:492 in game): the free-transform squash every editor does, so the rows clear the hook
SQUASH = {"stat_card": 0.86, "result": 0.86}
LEFT_MAX = (840, 620)
LEFT_AREA = LEFT_MAX[0] * LEFT_MAX[1]   # = as big as fits: the cap decides
LEFT_RIGHT = 860          # right edge limit: runs behind zy0x's near side like his tier lists
CARD_MARGIN = (-40, 10)   # a card's soft left fade bleeds off the frame so its solid part hugs the edge
FADES = {"stat_card": (0.05, 0.02, 0.03, 0.18), "result": (0.05, 0.02, 0.03, 0.18),
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


def art_size(mask: np.ndarray, max_w: float, canvas=(1280, 720)) -> tuple[float, float]:
    """on-screen (w, h) for a cutout hung at ART_AT: grown until the part inside the frame covers
    ART_COVER of it (the rest runs off the bottom behind the hook), capped by max_w."""
    W, H = canvas
    mh, mw = mask.shape
    rows = mask.mean(axis=1)
    best = None
    for sh in range(300, ART_MAX_H + 1, 10):
        sw = sh * mw / mh
        if sw > max_w:
            break
        vis = min(sh, H - ART_AT[1])
        k = max(1, int(mh * vis / sh))
        best = (sw, sh)
        if rows[:k].mean() * sw * vis >= ART_COVER * W * H:
            break
    return best or (max_w, max_w * mh / mw)


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
    fk = "art" if kind == "art" else made_kind or "stat_card"
    soft = p.with_name(p.stem + "_soft.png")
    im = Image.open(p)
    if kind == "art":                   # the cut edge fades out behind the hook
        im = body(head_to_thigh(im))
    fx.fade(im, FADES.get(fk, FADES["stat_card"])).save(soft)
    w, h = im.size
    aspect = w / h / SQUASH.get(fk, 1.0)
    if kind == "art":       # a cutout figure: nearly flat on, may run behind the hook
        # sized by what the figure covers, not its box: a cutout is mostly transparent
        # a team stays in the left part so its middle (the hook's character) isn't behind zy0x's head
        sw, sh = art_size(np.asarray(im.getchannel("A")) > 128, LEFT_RIGHT + 30 if aspect > 1.4 else ART_MAX_W)
        x0, y0 = ART_AT
        face = art_face(im) if aspect <= 1.4 else None      # a team keeps its spread from the left edge
        if face:                                             # the face on ART_FACE, the figure around it
            x0, y0 = ART_FACE[0] - face[0] * sw, max(-0.5 * sh, min(80.0, ART_FACE[1] - face[1] * sh))
        box = (x0, y0, x0 + sw, y0 + sh)
        rot = [0, 8, 0]
    else:
        if fk == "tierlist":    # his tier lists run on behind zy0x (only ~4 icons per row show)
            sw, sh = screen_size((aspect, 1))
            box = left_box(sw, sh, CENTERS[fk], margin=CARD_MARGIN, right=LEFT_RIGHT + 60)
        else:
            sw, sh = screen_size((aspect, 1))
            box = left_box(sw, sh, CENTERS.get(fk, LEFT_CENTER), margin=CARD_MARGIN)
        rot = [3, 14 if c.type == "subject" else 16, -2]
    card = fit_card(f"assets/{soft.name}", aspect, box, -70, rot)
    if fk in SQUASH:
        card["h"] = round(card["w"] / aspect, 1)
    if kind == "art" and face:                           # screen px, for the critic's measurements
        card["face"] = [round(box[0] + face[0] * sw), round(box[1] + face[1] * sh)]
    return [card], box


def art_face(im, score: float = 0.3) -> tuple[float, float] | None:
    """the character's face center (0-1 of the figure) in official art: yunet finds anime faces at
    a low threshold (checked on 7 splash arts); the largest, surest one wins."""
    import cv2

    from ..facecam import YuNet
    a = np.asarray(im.convert("RGBA"))
    bgr = cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2BGR)
    bgr[a[:, :, 3] < 128] = 0
    k = 640 / max(bgr.shape[:2])
    faces = YuNet(score=score)(cv2.resize(bgr, (round(bgr.shape[1] * k), round(bgr.shape[0] * k))))
    if not faces:
        return None
    x, y, w, h, _ = max(faces, key=lambda f: f[2] * f[3] * f[4])
    return float(x + w / 2), float(y + h / 2)


def build_spec(c, d: Path, face_png: Path, subject_png, subject_kind: str, bg_rgb, seed: int,
               made_kind: str | None = None, hides: bool = False, backdrop: bool = False) -> dict:
    """subject_kind: art | made. hides: a made graphic that already hides its own values.
    backdrop: bg.jpg is a real in-game screenshot (BACKDROP grade), not the stream frame."""
    from PIL import Image

    from . import compose
    from . import fx as pfx
    from . import text as T
    im = Image.open(face_png)
    arr = np.asarray(im.convert("RGBA"))[:, :, [2, 1, 0, 3]]
    box = compose.face_box(arr)
    if box is None:
        raise ValueError("no face in the cutout")
    target = compose.FIGURE_TARGET                     # every layout: the 80-90 rule (was 0.80 for subject)
    import shutil
    shutil.copy(SPEEDLINES, d / "assets" / "speedlines.png")
    f = compose.place_face(im.size, box, compose.head_top(arr), target=target)
    fx, fy, fw, fh = f["face"]
    cards, sbox = cards_for(c, subject_png, subject_kind, made_kind)
    spec = {"version": 1, "layout": c.type, "canvas": [1280, 720], "background": {"vignette": 0.55},
            "shade": SHADE,
            "layers": [{"kind": "face", "src": f"assets/{face_png.name}", "x": f["x"], "y": f["y"], "w": f["w"],
                        "glow": "#ffffff", "face_box": f["face"]}],
            "scene": {"seed": seed, "fov": FOV, "bg": {"raw": "assets/bg.jpg", "z": -400,
                                                       **(BACKDROP if backdrop else STREAM_BG)},
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
                     "cap_frac": CAP_FRAC, "width_frac": WIDTH_FRAC, "plane": T.PLANE}}   # no skew: the slant is the 3D plane
    # a card that hides its own values already reads "???" beside its mosaic (his strongbox ref: no
    # bubble, which would sit on the card's title); a tier list's pixelated icons still get the bubble
    if c.type == "reveal" and cards and subject_kind != "art" and not (hides and made_kind in ("stat_card", "result")):
        # a made graphic hides its own values; one that doesn't gets the mosaic band
        compose.reveal(spec, d, boxes=None if hides else [(0.06, 0.30, 0.94, 0.62)],
                       bubble_at=None if c.quote else (round(min(sbox[2] - 70, 560)), round(max(12, sbox[1] - 40))))
    if c.type == "reveal" and subject_kind == "art":     # his NUTTY SKIRK PULLS: big pixel ??? by the character,
        x0 = fx - 40 - 3 * 108                           # ending short of zy0x's face
        compose.reveal(spec, d, glyphs=[("?", round(x0 + 108 * i), 170, 8) for i in range(3)])
    if c.face.emote == "anger":                          # his stygian refs: the anger mark over his head
        g = pfx.anger(round(0.62 * fw))
        g.save(d / "assets" / "emote.png")
        # on the hair at his left temple (his: over the head's top-left; ours fills 90%, so the head top is
        # at the frame edge and a mark above it would be cut)
        spec["layers"].append({"kind": "emote", "src": "assets/emote.png", "x": round(fx - 0.42 * fw),
                               "y": round(max(6, fy - 0.42 * fh)), "w": g.width})
    if c.quote:                                          # his words in a pixel speech bubble, pointing at him
        q = c.quote.upper()[:QUOTE_MAX]
        cell = max(3, min(6, 500 // pfx.bubble(q, cell=1).width))
        b = pfx.bubble(q, cell=cell, tail="right")
        b.save(d / "assets" / "quote.png")
        right = min(1264, fx + 0.8 * fw)                 # over his head, clear of a card's title
        spec["layers"].append({"kind": "quote", "src": "assets/quote.png", "x": round(max(16, right - b.width)),
                               "y": 10, "w": b.width})
    return spec


# --------------------------------------------------------------------------- run

SHEET_COLS, SHEET_ROWS, SHEET_TILE = 4, 3, (480, 270)


def beat_times(run_dir: Path) -> list[tuple[int, float]]:
    """(beat id, source s) worth a look: the middle of every segment of every outline beat."""
    beats = json.loads((Path(run_dir) / "outline.json").read_text(encoding="utf-8")).get("beats", [])
    return [(i, round((a + e) / 2, 1)) for i, b in enumerate(beats)
            for a, e in (b.get("segments") or [[b.get("start", 0), b.get("end", 0)]])]


def frame_sheets(cams, times: list[tuple[int, float]], root: Path, sid: int = 0) -> list[str]:
    """the gameplay at each beat, labelled "b<beat> <source s>", 12 to a sheet -> file names in
    root. what the director looks at before planning the left element: the transcript can't
    say what was on screen."""
    import cv2
    cam = cams.detect(sid)
    gx0, _, gx1, _ = gameplay((0, 0, 1, 1), cam.box if cam else None)
    tw, th = SHEET_TILE
    tiles, seen = [], []
    for b, t in times:
        got = cams.frames(sid, [t], width=0)
        if not got:
            continue
        seen.append(t)
        f = got[0][:, int(gx0 * got[0].shape[1]):int(gx1 * got[0].shape[1])]
        w = min(tw, round(f.shape[1] * th / f.shape[0]))
        tile = np.zeros((th, tw, 3), np.uint8)
        tile[:, (tw - w) // 2:(tw - w) // 2 + w] = cv2.resize(f, (w, th), interpolation=cv2.INTER_AREA)
        label = f"b{b} {t:.0f}s"
        for color, thick in (((0, 0, 0), 5), ((255, 255, 255), 2)):
            cv2.putText(tile, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thick, cv2.LINE_AA)
        tiles.append(tile)
    per = SHEET_COLS * SHEET_ROWS
    names = []
    for k in range(0, len(tiles), per):
        chunk = tiles[k:k + per] + [np.zeros((th, tw, 3), np.uint8)] * (per - len(tiles[k:k + per]))
        sheet = np.vstack([np.hstack(chunk[r * SHEET_COLS:(r + 1) * SHEET_COLS]) for r in range(SHEET_ROWS)])
        names.append(f"frames_{k // per + 1}.jpg")
        cv2.imwrite(str(Path(root) / names[-1]), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    (Path(root) / "frames.json").write_text(json.dumps(seen), encoding="utf-8")   # what the director saw
    return names


STREAM_BG = {"blur": 12, "brightness": 0.6, "saturate": 1.15}      # the stream frame: pushed back hard
# nguy: slightly blurred, a little darkened. measured on his EASIEST ABYSS? (the same Spiral Abyss
# screenshot): edges ~5 px soft on screen, sky ~15% darker and ~10% less saturated than ours at 0.85 / 1.1
BACKDROP = {"blur": 7, "brightness": 0.72, "saturate": 1.0}


def backdrop(names: str | None, log=print):
    """the first of the comma-separated places with a wiki screenshot (BGR, 16:9), or None (none
    named, none found, offline)."""
    import cv2

    from . import art
    for name in [n.strip() for n in (names or "").split(",") if n.strip()]:
        try:
            return cv2.imread(str(art.backdrop(name)))
        except art.ArtError as e:
            log(f"[thumb] backdrop {name!r}: {str(e)[:100]}")
    return None


def cover(img, aspect: float = 16 / 9):
    """center crop to the canvas aspect: the scene stretches its plate to 16:9."""
    h, w = img.shape[:2]
    cw, ch = min(w, round(h * aspect)), min(h, round(w / aspect))
    x, y = (w - cw) // 2, (h - ch) // 2
    return img[y:y + ch, x:x + cw]


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


def _figure(q: str, log=print):
    """one name -> (figure png cut out of its official art, sidecar json text) or None.
    a character -> its wish splash; anything else (weapon, item, enemy, boss) -> its wiki image."""
    from . import art
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
    return fig, src.with_suffix(".json").read_text(encoding="utf-8")


def head_to_thigh(im):
    """a full-body figure reads tiny: keep its top FIG_KEEP (his arlecchino, skirk), counted from
    where the body starts: thin effect rows above the head (a ribbon, a swoosh) are dropped first."""
    if im.height > FIG_TALL * im.width:
        rows = (np.asarray(im.getchannel("A")) > 128).mean(axis=1)
        top = int(np.argmax(rows >= FIG_THIN * rows.max()))
        im = im.crop((0, top, im.width, top + round((im.height - top) * FIG_KEEP)))
        im = im.crop(im.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox() or (0, 0, *im.size))
    return im


def body(im):
    """hang a figure by its body: crop to its solid pixels, then drop thin effect columns at either
    side (sucrose's wind ribbon, a trailing sash) that would push the body toward the middle."""
    im = im.crop(im.getchannel("A").point(lambda v: 255 if v > 128 else 0).getbbox() or (0, 0, *im.size))
    cols = (np.asarray(im.getchannel("A")) > 128).mean(axis=0)
    keep = np.where(cols >= FIG_THIN * cols.max())[0] if cols.max() > 0 else []
    return im.crop((int(keep[0]), 0, int(keep[-1]) + 1, im.height)) if len(keep) else im


def team(figs: list):
    """2-3 figures as one group (his EASIEST ABYSS?): same height, overlapping, the middle one
    larger and in front."""
    from PIL import Image
    H = 900
    ims = [head_to_thigh(f.convert("RGBA")) for f in figs]
    if len(ims) == 3:                                     # the first named (the hook's) stands in the middle
        ims = [ims[1], ims[0], ims[2]]
    mid = len(ims) // 2 if len(ims) == 3 else -1
    ims = [im.resize((round(im.width * H * (1.12 if i == mid else 0.92) / im.height),
                      round(H * (1.12 if i == mid else 0.92)))) for i, im in enumerate(ims)]
    xs, x = [], 0
    for im in ims:
        xs.append(x)
        x += round(im.width * TEAM_STEP)
    W, top = xs[-1] + ims[-1].width, max(im.height for im in ims)
    out = Image.new("RGBA", (W, top))
    order = [i for i in range(len(ims)) if i != mid] + ([mid] if mid >= 0 else [])
    for i in order:                                       # the middle one last = in front
        out.alpha_composite(ims[i], (xs[i], top - ims[i].height))
    return out.crop(out.getchannel("A").getbbox())


def art_card(query: str, d: Path, log=print) -> Path | None:
    """official art for the left element -> assets/art0.png: one name -> its figure; up to 3
    names, comma separated -> the team as one group."""
    import shutil

    from PIL import Image
    names = [n.strip() for n in (query or "").split(",") if n.strip()][:3]
    got = [g for g in (_figure(n, log) for n in names) if g]
    if not got:
        return None
    out = d / "assets" / "art0.png"
    if len(got) == 1:
        shutil.copy(got[0][0], out)
    else:
        team([Image.open(f) for f, _ in got]).save(out)
    for i, (_, meta) in enumerate(got):
        (d / "assets" / f"art{i}.art.json").write_text(meta, encoding="utf-8")
    return out


def characters_in(names: list[str]) -> list[str]:
    """which of these brief entities are genshin characters (resolved against the roster)."""
    from . import art
    try:
        roster = art.characters(art.Fetcher())
    except art.ArtError:
        return []
    return [r for r in (art.resolve(n, roster) for n in names) if r]


def left_queries(c, brief: dict) -> list[str]:
    """the art lookups that still show what the plan meant: its query, its backup, then the
    video's own subject (brief.about_art). never a character only mentioned in passing."""
    return list(dict.fromkeys(q for q in (c.subject.query, c.subject.alt_query, brief.get("about_art")) if q))


def assemble(c, k: str, root: Path, cams, browser, variant: int, brief: dict, log=print) -> None:
    """one concept -> <root>/<k>/{assets, spec.json, thumb.png}."""
    import cv2

    from . import render
    from .cutout import cutout
    sid = 0
    d = root / k
    (d / "assets").mkdir(parents=True, exist_ok=True)
    c.face.t = snap(cams, sid, c.face.t)
    crop, info = best_face(cams, sid, c.face.t, c.face.expression)
    cv2.imwrite(str(d / "assets" / "face.png"), cutout(crop))
    cam_box = cams.detect(sid).box
    sub, kind = None, c.subject.source
    if kind == "made" and c.subject.made:
        from . import made
        try:
            sub = made.render(c.subject.made.get("kind", "tierlist"), c.subject.made, d / "assets" / "made.png",
                              browser=browser)
        except Exception as e:                           # a bad made spec falls back to art
            log(f"[thumb] {k}: made graphic failed ({e}); trying character art")
    if sub is None:                                      # nguy: there is always a left element
        kind = "art"
        for q in left_queries(c, brief):
            if (sub := art_card(q, d, log)) is not None:
                break
        else:                                            # ponytail: off-plan last resort, flagged in the report
            sub = art_card(next(iter(characters_in(brief.get("entities", []))), ""), d, log)
            (d / "assets" / "off_plan.txt").write_text(f"none of {left_queries(c, brief)} had art", encoding="utf-8")
    bg = backdrop(brief.get("backdrop"), log)
    real = bg is not None
    if not real:                                         # no place named / no screenshot: the stream frame
        bg = cams.frames(sid, [c.face.t], width=0)[0]
        gx0, _, gx1, _ = gameplay((0, 0, 1, 1), cam_box)
        bg = cover(bg[:, int(gx0 * bg.shape[1]):int(gx1 * bg.shape[1])])
    cv2.imwrite(str(d / "assets" / "bg.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 90])
    bg_rgb = cv2.cvtColor(cv2.GaussianBlur(bg, (0, 0), 12), cv2.COLOR_BGR2RGB)
    made_kind = (c.subject.made or {}).get("kind", "tierlist") if kind == "made" else None
    spec = build_spec(c, d, d / "assets" / "face.png", sub, kind, bg_rgb, seed=variant * 10 + "ABC".index(k),
                      made_kind=made_kind, hides=made_kind is not None and bool(c.subject.made.get("hidden")),
                      backdrop=real)
    (d / "spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    render.render(d / "spec.json", browser=browser)
    log(f"[thumb] {k} {c.type}: {' / '.join(c.hook.lines)}  (face q {info['q']}, subject {kind if sub else 'none'})")


def _render_all(run_dir: Path, root: Path, concepts: list, variant: int, brief: dict, log=print,
                critic: bool = True) -> Path:
    from ..facecam import Facecams
    from . import render, report
    cams = Facecams(run_dir / "ae" / "manifest.json")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = render.launch(p)
        try:
            for k, c in zip("ABC", concepts):
                assemble(c, k, root, cams, b, variant, brief, log)
            if critic:                                   # one cheap look at placement, moves applied
                from . import critic as crit
                crit.review(root, browser=b, log=log)
        finally:
            b.close()
    return report.build(run_dir, root=root)


def run(run_dir, variant: int = 1, previous_hooks: list[str] | None = None, backend: str = "cli",
        model: str = "claude-opus-5", ask=None, log=print, critic: bool = True) -> Path:
    """one full candidate run -> <run_dir>/thumb/v<variant>/report.md."""
    run_dir = Path(run_dir)
    root = run_dir / "thumb" / f"v{variant}"
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    ctx = context(run_dir)
    log(f"[thumb] context: {len(ctx['beats'])} beats, {len(ctx['moments'])} moments ({time.monotonic() - t0:.0f}s)")
    from ..facecam import Facecams
    sheets = frame_sheets(Facecams(run_dir / "ae" / "manifest.json"), beat_times(run_dir), root)
    log(f"[thumb] frame sheets: {sheets}")
    t1 = time.monotonic()
    out = direct(ctx, previous_hooks or [], variant, backend, model, ask=ask, sheets=sheets, cwd=root)
    log(f"[thumb] director: {[c.hook.lines for c in out.concepts]} ({time.monotonic() - t1:.0f}s)")
    (root / "brief.json").write_text(json.dumps(out.brief, indent=1, ensure_ascii=False), encoding="utf-8")
    order = sorted(out.concepts, key=lambda c: TYPES.index(c.type))
    concepts = [{"id": k, **c.model_dump()} for k, c in zip("ABC", order)]
    (root / "concepts.json").write_text(json.dumps({"concepts": concepts}, indent=1, ensure_ascii=False), encoding="utf-8")
    rep = _render_all(run_dir, root, order, variant, out.brief, log, critic=critic)
    log(f"[thumb] v{variant}: {rep}  ({time.monotonic() - t0:.0f}s)")
    return rep


def recompose(run_dir, variant: int, log=print, critic: bool = False) -> Path:
    """rebuild a run's candidates from its saved brief + concepts (no director call): for
    layout/asset changes."""
    run_dir = Path(run_dir)
    root = run_dir / "thumb" / f"v{variant}"
    Concept = _models().model_fields["concepts"].annotation.__args__[0]
    concepts = [Concept.model_validate({k: v for k, v in c.items() if k != "id"})
                for c in json.loads((root / "concepts.json").read_text(encoding="utf-8"))["concepts"]]
    brief = json.loads((root / "brief.json").read_text(encoding="utf-8"))
    return _render_all(run_dir, root, concepts, variant, brief, log, critic=critic)
