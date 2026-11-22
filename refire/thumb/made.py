"""Left-side graphics built in code when the stream didn't show a clean panel (Nguy: "you have
the freedom to create an asset/panel yourself"). Rendered by the pinned Chromium to a
transparent PNG, then used as a card in the 3D scene like any panel.

  tierlist  {"rows": [{"label": "GOD TIER", "color": "#FF7F7F", "chars": ["Furina", ...]}]}
            (the refs: GOD TIER / OKAY / RUN AWAY, T0 / T0.5 / T67), official character icons
  stat_card {"title": "Bloom of the Mind's Desire", "set": "Night of the Sky's Unveiling",
             "subtitle": "Flower of Life", "main": ["HP", "717"], "stars": 5, "level": "+0",
             "rows": [["CRIT DMG", "23.3%"], ...], "hidden": [0, 1]}
            the in-game artifact card one-to-one (game font, measured layout, the piece's official
            icon + set bonus from yatta); hidden rows get a pixel mosaic + "???" (the reveal)
"""
from __future__ import annotations

import base64
import io
import json
import random
from html import escape
from pathlib import Path

from . import art

# his tier lists: 3 rows of big icons (MORE TIER LISTS: ~150 px cells, the row's tail behind zy0x)
TIER_ROWS, TIER_PER_ROW = 3, 4
TIER_COLORS = ["#FF7F7F", "#FFBF7F", "#FFDF7F", "#BFFF7F", "#7FFFFF", "#7FBFFF"]


def icon(name: str, f: art.Fetcher | None = None, cache: Path = art.CACHE) -> Path | None:
    """official avatar icon (enka's UI_AvatarIcon_X.png, ~256 px with alpha), cached."""
    f = f or art.Fetcher()
    roster = art.characters(f, cache)
    canon = art.resolve(name, roster)
    if not canon:
        return None
    out = cache / f"{art.slug(canon)}_icon.png"
    if out.exists():
        return out
    for url in (f"https://enka.network/ui/{roster[canon]}.png", f"{art.YATTA}/assets/UI/{roster[canon]}.png"):
        try:
            body, _, final = f.get(url, accept=art.IMAGES)
        except art.ArtError:
            continue
        cache.mkdir(parents=True, exist_ok=True)
        out.write_bytes(body)
        out.with_suffix(".json").write_text(json.dumps({"name": canon, "url": final}), encoding="utf-8")
        return out
    return None


def _pixelated(p: Path) -> Path:
    """an icon reduced to an 11x11 mosaic (nearest-neighbor), cached beside it."""
    from PIL import Image
    out = p.with_name(p.stem + "_px.png")
    if not out.exists():
        im = Image.open(p).convert("RGBA")
        im.resize((11, 11), Image.BOX).resize(im.size, Image.NEAREST).save(out)
    return out


def _data_uri(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:transparent;font-family:"Segoe UI",Arial,sans-serif}}
{css}</style></head><body>{body}</body></html>"""

TIER_CSS = """
.tl{display:inline-grid;grid-template-columns:170px auto;gap:5px;background:#1b1b1f;padding:5px;border-radius:6px}
.lab{display:flex;align-items:center;justify-content:center;font-weight:800;font-size:36px;color:#111;
     text-align:center;line-height:1.05;padding:0 8px;min-height:160px}
.row{display:flex;gap:5px;background:#2a2a30;padding:5px;min-width:160px}
.row img{width:150px;height:150px;object-fit:cover;background:radial-gradient(circle at 50% 35%,#5b4b7a,#2b2338);
         border-radius:4px}
"""

# --------------------------------------------------------------------------- artifact card
# the in-game artifact detail card, one-to-one: every position, size and color below was
# measured off a 1080p stream frame of the real card (strongbox stream, "Revelation's Toll"),
# card px at its native 492 px width. the header art (title bar, 5-star gradient, ornament) is
# that frame's header with its text and icon filled out (assets/artifact_head_5.png); the lock /
# favorite buttons are cropped from the same frame. font = the game's own (SDK_JP_Web Heavy,
# ja-jp.ttf). sizes come from ink heights: this face's caps run -40..745 units.
ASSETS = Path(__file__).with_name("assets")
GAME_FONT = "ja-jp.ttf"
SLOTS = {"flower": "EQUIP_BRACER", "plume": "EQUIP_NECKLACE", "feather": "EQUIP_NECKLACE", "sands": "EQUIP_SHOES",
         "goblet": "EQUIP_RING", "circlet": "EQUIP_DRESS"}
CARD = {"w": 492, "head": 283, "pad": 29.5, "body_bg": "#EBE3D8",
        "title": (32, 36.7), "slot": (20, 88), "label": (23, 171), "main": (42.5, 215),
        "stars": (41.5, 252, 33.5, 13.6), "icon": (263, 58, 225),
        "badge": (30, 309, 27, 22, 329), "buttons": (360, 292), "row0": 382, "pitch": 38.7, "size": 24, "line": 29,
        "track": -0.7, "wrap": 395,
        "text": "#4D5057", "unact": "#99978F", "green": "#5EBE53", "bonus": "#737371", "ring": "#B8B2AA"}


CARD_MAX_H = 560     # thumbnails show the header through the substats + set name, fading out (his strongbox card)


def game_font() -> Path:
    """the game's font subset to latin (the full file is 11 MB), cached."""
    from . import text
    src = text._installed(GAME_FONT)
    if not src.exists():
        raise FileNotFoundError(f"the game font {GAME_FONT} (SDK_JP_Web) isn't installed")
    out = art.CACHE.parent / "fonts" / "sdk_jp_web_latin.ttf"
    if not out.exists():
        from fontTools import subset
        from fontTools.ttLib import TTFont
        f = TTFont(src)
        s = subset.Subsetter(subset.Options())
        s.populate(unicodes=[*range(0x20, 0x7F), 0xB7, 0x2018, 0x2019, 0x201C, 0x201D, 0x2026, 0x2605])
        s.subset(f)
        out.parent.mkdir(parents=True, exist_ok=True)
        f.save(out)
    return out


def relic_sets(f: art.Fetcher, cache: Path = art.CACHE) -> dict[str, str]:
    """artifact set name -> yatta id, refreshed daily."""
    from datetime import date
    p = cache / "yatta_reliquary.json"
    if p.exists() and json.loads(p.read_text(encoding="utf-8")).get("date") == date.today().isoformat():
        return json.loads(p.read_text(encoding="utf-8"))["items"]
    body, _, _ = f.get(f"{art.YATTA}/api/v2/en/reliquary", accept=("application/json",))
    out = {v["name"]: str(k) for k, v in json.loads(body)["data"]["items"].items() if v.get("name")}
    cache.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"date": date.today().isoformat(), "items": out}, indent=1), encoding="utf-8")
    return out


def relic_set(sid: str, f: art.Fetcher, cache: Path = art.CACHE) -> dict:
    """one set's pieces + bonuses (cached for good: sets don't change)."""
    p = cache / f"yatta_reliquary_{sid}.json"
    if not p.exists():
        body, _, _ = f.get(f"{art.YATTA}/api/v2/en/reliquary/{sid}", accept=("application/json",))
        cache.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
    return json.loads(p.read_text(encoding="utf-8"))["data"]


def artifact(c: dict, f: art.Fetcher | None = None, cache: Path = art.CACHE) -> dict | None:
    """the card's piece: {"set", "bonus": [2pc, 4pc], "icon": Path} from its set name (fuzzy) or,
    failing that, its piece name across every set. None when it can't be found."""
    import difflib
    import re
    f = f or art.Fetcher()
    try:
        sets = relic_sets(f, cache)
        low = {n.lower(): n for n in sets}
        hit = (difflib.get_close_matches((c.get("set") or "").lower(), list(low), n=1, cutoff=0.75)
               or difflib.get_close_matches((c.get("title") or "").lower(), list(low), n=1, cutoff=0.85))
        order = [sets[low[hit[0]]]] if hit else list(sets.values())
        title = (c.get("title") or "").lower()
        slot = next((v for k, v in SLOTS.items() if k in (c.get("subtitle") or "").lower()), None)
        for sid in order:
            d = relic_set(sid, f, cache)
            suit = d.get("suit") or {}
            piece = next((p for p in suit.values() if p.get("name", "").lower() == title), None)
            if piece is None and hit:
                piece = suit.get(slot) or next(iter(suit.values()), None)
            if piece is None:
                continue
            out = cache / f"{piece['icon']}.png"
            if not out.exists():
                body, _, _ = f.get(f"{art.YATTA}/assets/UI/reliquary/{piece['icon']}.png", accept=art.IMAGES)
                out.write_bytes(body)
            bonus = [re.sub(r"<[^>]+>", "", v) for _, v in sorted((d.get("affixList") or {}).items())]
            return {"set": d.get("name"), "bonus": bonus, "icon": out, "piece": piece.get("name")}
    except art.ArtError:
        return None
    return None


def _wrap(s: str, m, size: float, width: float, track: float = 0.0) -> list[str]:
    lines, cur = [], ""
    for w in s.split():
        nxt = f"{cur} {w}".strip()
        if cur and m.width(nxt, size, track / size) > width:
            lines.append(cur)
            cur = w
        else:
            cur = nxt
    return lines + ([cur] if cur else [])


def _star(cx: float, cy: float, r: float) -> str:
    import math
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.5
        pts.append(f"{cx + rr * math.cos(a):.2f},{cy + rr * math.sin(a):.2f}")
    return (f'<polygon points="{" ".join(pts)}" fill="#F9CF39" stroke="#E2A72C" stroke-width="1.1" '
            f'stroke-linejoin="round"/>')


def _value(stat: str, val: str) -> str:
    v = str(val).strip()
    return f"{stat}{v if v.startswith(('+', '-')) else '+' + v}"


def stat_card_svg(c: dict, piece: dict | None = None, height: float | None = None, max_h: float = 1e9) -> str:
    """the in-game card as one svg page (baselines exact). c = the director's stat_card spec;
    rows are [stat, value] or [stat, value, "unactivated"]; hidden row values become a pixel
    mosaic + "???" (the strongbox ref's reveal)."""
    from . import text
    fp = game_font()
    m = text.metrics(str(fp))
    K = CARD
    W, x0 = K["w"], K["pad"]
    e = escape
    title = (piece or {}).get("piece") or c.get("title", "")          # the real piece name when known
    fit = (W - 2 * x0) / 1.06                    # metrics run ~5% under the drawn width
    ts = min(K["title"][0], fit * 1000 / max(1.0, m.width(title, 1000, 0)))
    out = [f'<image href="{_data_uri(ASSETS / "artifact_head_5.png")}" x="0" y="0" width="{W}" height="{K["head"]}"/>']
    ix, iy, isz = K["icon"]
    if piece and piece.get("icon"):
        out.append(f'<image href="{_data_uri(Path(piece["icon"]))}" x="{ix}" y="{iy}" width="{isz}" height="{isz}"/>')
    main = c.get("main") or ["", ""]
    out += [f'<text x="{x0}" y="{K["title"][1]}" font-size="{ts:.1f}" fill="#FFFFFF">{e(title)}</text>',
            f'<text x="{x0}" y="{K["slot"][1]}" font-size="{K["slot"][0]}" fill="#F2EDEA">{e(c.get("subtitle", ""))}</text>',
            f'<text x="{x0}" y="{K["label"][1]}" font-size="{K["label"][0]}" fill="#FFFFFF" fill-opacity="0.6">{e(main[0])}</text>',
            f'<text x="{x0}" y="{K["main"][1]}" font-size="{K["main"][0]}" fill="#FFFFFF">{e(str(main[1]))}</text>']
    sx, sy, sp, sr = K["stars"]
    out += [_star(sx + i * sp, sy, sr) for i in range(max(1, min(5, int(c.get("stars", 5)))))]
    # body
    bx, by, bh, bs, bb = K["badge"]
    lv = str(c.get("level", "+0"))
    bw = m.width(lv, bs, 0) + 25
    out += [f'<rect x="{bx}" y="{by}" width="{bw:.1f}" height="{bh}" rx="3" fill="#3A434E"/>',
            f'<text x="{bx + bw / 2:.1f}" y="{bb}" font-size="{bs}" fill="#FFFFFF" text-anchor="middle">{e(lv)}</text>',
            f'<image href="{_data_uri(ASSETS / "artifact_buttons.png")}" x="{K["buttons"][0]}" y="{K["buttons"][1]}" '
            f'width="116" height="58"/>']
    hidden = set(c.get("hidden") or [])
    y, size, tr = K["row0"], K["size"], K["track"]
    ls = f' letter-spacing="{tr}"'
    for i, row in enumerate((c.get("rows") or [])[:4]):
        stat, val = row[0], row[1] if len(row) > 1 else ""
        dim = len(row) > 2 and "unact" in str(row[2]).lower()
        col = K["unact"] if dim else K["text"]
        dot = f'<text x="{x0}" y="{y:.1f}" font-size="{size}" fill="{col}">·</text>'
        if i in hidden:
            lead = f"{stat}+"
            lx = x0 + 24 + m.width(lead, size, tr / size) + tr
            out += [dot, f'<text x="{x0 + 24}" y="{y:.1f}" font-size="{size}" fill="{col}"{ls}>{e(lead)}</text>',
                    _mosaic_svg(i + 1, lx + 2, y - 18, 7, 2, 9),
                    f'<text x="{lx + 2 + 7 * 9 + 2:.1f}" y="{y:.1f}" font-size="{size}" fill="{col}"{ls}>???</text>']
        else:
            tail = "  (unactivated)" if dim else ""
            out += [dot, f'<text x="{x0 + 24}" y="{y:.1f}" font-size="{size}" fill="{col}"{ls} xml:space="preserve">'
                         f'{e(_value(stat, val) + tail)}</text>']
        y += K["pitch"]
    if piece and piece.get("set"):
        out.append(f'<text x="{x0}" y="{y:.1f}" font-size="{size}" fill="{K["green"]}"{ls}>{e(piece["set"])}:</text>')
        y += K["pitch"] - 1
        for n, b in zip((2, 4), piece.get("bonus", [])[:2]):
            out.append(f'<circle cx="{x0 + 10.5}" cy="{y - 8:.1f}" r="8" fill="none" stroke="{K["ring"]}" stroke-width="1.8"/>'
                       f'<line x1="{x0 + 5}" y1="{y - 2.5:.1f}" x2="{x0 + 16}" y2="{y - 13.5:.1f}" stroke="{K["ring"]}" stroke-width="1.8"/>')
            for j, ln in enumerate(_wrap(f"{n}-Piece Set: {b}", m, size, K["wrap"], tr)):
                out.append(f'<text x="{x0 + 24}" y="{y + j * K["line"]:.1f}" font-size="{size}" fill="{K["bonus"]}"{ls}>{e(ln)}</text>')
            y += (j + 1) * K["line"] + (K["pitch"] - K["line"]) + 1
    H = round(height or min(max_h, y - K["pitch"] + 40))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="SDKJP" style="display:block">'
           f'<rect x="0" y="{K["head"]}" width="{W}" height="{H - K["head"]}" fill="{K["body_bg"]}"/>'
           f'<rect x="0" y="{K["head"]}" width="{W}" height="1" fill="#DED3C0"/>{"".join(out)}</svg>')
    return PAGE.format(css=text._face("SDKJP", fp), body=svg)


def _mosaic_svg(seed: int, x: float, y: float, cols: int, rows: int, cell: float) -> str:
    """pixel blocks standing in for a hidden value: a pixelated dark number on the card's beige."""
    rnd = random.Random(seed)
    return "".join(f'<rect x="{x + i * cell:.1f}" y="{y + j * cell:.1f}" width="{cell}" height="{cell}" '
                   f'fill="rgb({g},{g - 2},{g - 6})"/>'
                   for j in range(rows) for i in range(cols) for g in [rnd.randint(120, 225)])


def tierlist_html(rows: list[dict], icon_fn=icon, hidden=()) -> str:
    """hidden = [[row, col], ...]: those icons drawn as a coarse pixel mosaic (the reveal)."""
    out, hide = [], {tuple(h) for h in hidden or []}
    for i, r in enumerate(rows[:TIER_ROWS]):
        color = r.get("color") or TIER_COLORS[i % len(TIER_COLORS)]
        cells = []
        for j, p in enumerate(icon_fn(n) for n in r.get("chars", [])[:TIER_PER_ROW]):
            if not p:
                continue
            if (i, j) in hide:
                cells.append(f'<img class="px" src="{_data_uri(_pixelated(p))}">')
            else:
                cells.append(f'<img src="{_data_uri(p)}">')
        imgs = "".join(cells)
        out.append(f'<div class="lab" style="background:{escape(color)}">{escape(r.get("label", "?"))}</div>'
                   f'<div class="row">{imgs}</div>')
    return PAGE.format(css=TIER_CSS, body=f'<div class="tl">{"".join(out)}</div>')


def render(kind: str, data: dict, out: Path, browser=None, icon_fn=icon, piece_fn=artifact) -> Path:
    """made graphic -> transparent PNG cropped to its pixels (2x for crisp edges)."""
    from PIL import Image
    html = (tierlist_html(data.get("rows", []), icon_fn, data.get("hidden")) if kind == "tierlist"
            else stat_card_svg(data, piece_fn(data), max_h=CARD_MAX_H))

    def shoot(b):
        pg = b.new_page(viewport={"width": 1400, "height": 1000}, device_scale_factor=2)
        try:
            pg.set_content(html, wait_until="load")
            return pg.screenshot(type="png", omit_background=True, full_page=True)
        finally:
            pg.close()

    if browser is None:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            try:
                png = shoot(b)
            finally:
                b.close()
    else:
        png = shoot(browser)
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    im.crop(im.getchannel("A").getbbox() or (0, 0, 1, 1)).save(out)
    return out
